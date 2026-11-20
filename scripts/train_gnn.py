#!/usr/bin/env python
"""EXP04 + EXP05: does relational information help, and does a GNN beat plain aggregates?

    python scripts/build_graph.py --config configs/graph/cohort_graph.yaml
    python scripts/train_gnn.py   --config configs/graph/cohort_graph.yaml

Three arms, all scored on the same out-of-time window:

    A  tabular features only                        (the control)
    B  tabular + time-respecting cohort aggregates  (EXP04)
    C  GraphSAGE over sampled cohort neighbourhoods (EXP05)

**The GNN is not assumed to win.** If arm B captures the relational signal and arm C does not
improve on it, that is the finding, and it is a more useful one than a marginal AUC gain: it says
the value was in the aggregation, not in the learned message passing. The script writes whichever
verdict the numbers support.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.data.splits import test_window_masks  # noqa: E402
from credit_risk.evaluation.metrics import (  # noqa: E402
    compare_models,
    compute_metrics,
    paired_bootstrap_calibration_comparison,
)
from credit_risk.features.build import (  # noqa: E402
    FeatureBuilder,
    FeatureSpec,
    to_numeric_matrix,
)
from credit_risk.features.financial import add_derived_features  # noqa: E402
from credit_risk.graph.construction import TimeRespectingGraph  # noqa: E402
from credit_risk.graph.features import compute_macro_default_rate  # noqa: E402
from credit_risk.models.boosting import BoostingConfig, BoostingModel  # noqa: E402
from credit_risk.models.calibration import ProbabilityCalibrator  # noqa: E402
from credit_risk.utils.cli import (  # noqa: E402
    base_parser,
    detect_source,
    load_prepared,
    resolve,
    save_json,
    save_table,
    write_manifest,
    write_report,
)
from credit_risk.utils.runtime import get_logger  # noqa: E402

LOG = get_logger("scripts.gnn")


def _control_verdict(metrics, comp_macro, comp_shuffled) -> str:
    """The control verdict, derived from the intervals rather than asserted."""
    comps = [c for c in (comp_macro, comp_shuffled) if c is not None]
    survives = bool(comps) and all(c["delta_ece_ci"][1] < 0 for c in comps)
    text = (
        "**Control Verdict (single fitted run):** the paired-bootstrap interval on the ECE difference "
        + ("excludes zero against both controls." if survives
           else "does not exclude zero against every control.")
    )
    if "Control_shuffled_cohort" in metrics and "A_tabular" in metrics:
        text += (
            f" The shuffled-key control itself moves ECE from {metrics['A_tabular'].ece:.4f} to "
            f"{metrics['Control_shuffled_cohort'].ece:.4f}, so part of the gain is not specific to real "
            "cohort membership. The multi-seed comparison is `scripts/build_cohort_controls.py`."
        )
    return text


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/graph/cohort_graph.yaml")
    parser.add_argument("--skip-gnn", action="store_true", help="Run EXP04 only (no torch needed).")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = add_derived_features(load_prepared(paths).reset_index(drop=True))
    cohort_path = paths.data_processed / "cohort_features.parquet"
    graph_path = paths.data_processed / "graph.npz"
    if not cohort_path.exists() or not graph_path.exists():
        raise SystemExit(
            "Graph artefacts missing. Run:\n"
            "  python scripts/build_graph.py --config configs/graph/cohort_graph.yaml"
        )
    cohort = pd.read_parquet(cohort_path)
    df = pd.concat([df, cohort.set_index(df.index)], axis=1)

    masks = {s: (df["split"] == s).to_numpy() for s in ("train", "valid", "test")}
    train, valid, test = (df[masks[s]] for s in ("train", "valid", "test"))
    y = df["default"].astype("float32").to_numpy()
    y_train, y_valid, y_test = (y[masks[s]] for s in ("train", "valid", "test"))

    metrics: dict[str, object] = {}
    scores: dict[str, np.ndarray] = {}

    def run_boosting(label: str, extra: tuple[str, ...]) -> None:
        spec = FeatureSpec(feature_set="primary", extra_numeric=extra)
        builder = FeatureBuilder(spec).fit(train, valid)
        X = {k: builder.transform(d) for k, d in (("tr", train), ("va", valid), ("te", test))}
        model = BoostingConfig(n_estimators=2000, early_stopping_rounds=100)
        boosting = BoostingModel(builder.categorical_features, model).fit(
            X["tr"], y_train, X["va"], y_valid, seed=seed
        )
        calibrator = ProbabilityCalibrator("isotonic").fit(y_valid, boosting.predict_proba(X["va"]))
        scores[label] = calibrator.transform(boosting.predict_proba(X["te"]))
        metrics[label] = compute_metrics(y_test, scores[label])
        LOG.info("%-24s | %s", label, metrics[label].summary_line())
        if extra:
            importance = boosting.feature_importance()
            graph_share = importance[importance["feature"].str.startswith("cohort_")][
                "importance"
            ].sum() / max(importance["importance"].sum(), 1)
            LOG.info("  graph features account for %.1f%% of total gain", 100 * graph_share)
            metrics[label].extras["graph_gain_share"] = float(graph_share)
            save_table(importance.head(30), paths, f"{run_name}_{label}_importance", index=False)

    # -- arm A: tabular only ---------------------------------------------------
    run_boosting("A_tabular", ())

    # -- arm B: tabular + cohort aggregates (EXP04) ---------------------------
    cohort_columns = tuple(c for c in cohort.columns if c in df.columns)
    run_boosting("B_tabular_plus_graph", cohort_columns)

    # -- non-graph controls (P0-4 audit) --------------------------------------
    macro_rate = compute_macro_default_rate(df)
    df["macro_default_rate"] = macro_rate
    train, valid, test = (df[masks[s]] for s in ("train", "valid", "test"))
    run_boosting("Control_macro_default", ("macro_default_rate",))

    shuffled_path = paths.data_processed / "cohort_features_shuffled.parquet"
    if shuffled_path.exists():
        cohort_shuffled = pd.read_parquet(shuffled_path)
        shuf_cols = tuple(f"{c}_shuf" for c in cohort_columns)
        df_shuf = cohort_shuffled.copy()
        df_shuf.columns = list(shuf_cols)
        df_shuf_all = pd.concat([df, df_shuf.set_index(df.index)], axis=1)
        tr_s, va_s, te_s = (df_shuf_all[masks[s]] for s in ("train", "valid", "test"))

        spec_s = FeatureSpec(feature_set="primary", extra_numeric=shuf_cols)
        builder_s = FeatureBuilder(spec_s).fit(tr_s, va_s)
        X_s = {k: builder_s.transform(d) for k, d in (("tr", tr_s), ("va", va_s), ("te", te_s))}
        model_s = BoostingConfig(n_estimators=2000, early_stopping_rounds=100)
        boosting_s = BoostingModel(builder_s.categorical_features, model_s).fit(
            X_s["tr"], y_train, X_s["va"], y_valid, seed=seed
        )
        calibrator_s = ProbabilityCalibrator("isotonic").fit(y_valid, boosting_s.predict_proba(X_s["va"]))
        scores["Control_shuffled_cohort"] = calibrator_s.transform(boosting_s.predict_proba(X_s["te"]))
        metrics["Control_shuffled_cohort"] = compute_metrics(y_test, scores["Control_shuffled_cohort"])
        LOG.info("%-24s | %s", "Control_shuffled_cohort", metrics["Control_shuffled_cohort"].summary_line())

    # -- arm C: GraphSAGE (EXP05) ---------------------------------------------
    gnn_summary: dict[str, object] = {}
    device = "cpu"
    ckpt_path = paths.artifacts / f"{run_name}_graphsage.pt"
    should_run_gnn = not args.skip_gnn or ckpt_path.exists()
    if should_run_gnn:
        try:
            import torch

            from credit_risk.models.gnn import GNNConfig, GraphSAGE, GraphSAGETrainer
        except ImportError as exc:
            LOG.warning("skipping EXP05: %s", exc)
        else:
            gcfg = config.get("gnn", {})
            graph = TimeRespectingGraph.load(graph_path)

            base_spec = FeatureSpec(feature_set="primary")
            node_builder = FeatureBuilder(base_spec).fit(train, valid)
            X_all = node_builder.transform(df)
            _, _, means, stds = to_numeric_matrix(
                X_all[masks["train"]],
                node_builder.categorical_features,
                node_builder.state.categorical_levels,
            )
            node_features, node_names, _, _ = to_numeric_matrix(
                X_all,
                node_builder.categorical_features,
                node_builder.state.categorical_levels,
                means=means,
                stds=stds,
            )
            LOG.info("node feature matrix: %s", node_features.shape)
            idx = {s: np.flatnonzero(masks[s]) for s in masks}

            if args.skip_gnn and ckpt_path.exists():
                LOG.info("evaluating GraphSAGE from existing checkpoint: %s", ckpt_path)
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                cfg = ckpt.get("config", GNNConfig())
                trainer = GraphSAGETrainer(cfg)
                trainer._to_device(node_features, graph.neighbour_index, graph.neighbour_mask)
                trainer.model = GraphSAGE(
                    node_features.shape[1], graph.neighbour_index.shape[1], cfg
                ).to(trainer.device)
                trainer.model.load_state_dict(ckpt["state_dict"])
                trainer.model.eval()
                device = trainer.device
            else:
                trainer = GraphSAGETrainer(
                    GNNConfig(
                        hidden_dim=int(gcfg.get("hidden_dim", 64)),
                        embed_dim=int(gcfg.get("embed_dim", 32)),
                        n_layers=int(gcfg.get("n_layers", 2)),
                        dropout=float(gcfg.get("dropout", 0.2)),
                        fanout=tuple(gcfg.get("fanout", [10, 5])),
                        learning_rate=float(gcfg.get("learning_rate", 1e-3)),
                        weight_decay=float(gcfg.get("weight_decay", 1e-5)),
                        batch_size=int(gcfg.get("batch_size", 1024)),
                        max_epochs=int(gcfg.get("max_epochs", 30)),
                        patience=int(gcfg.get("patience", 5)),
                        l2_normalize=bool(gcfg.get("l2_normalize", True)),
                        balance_classes=bool(gcfg.get("balance_classes", False)),
                        device=str(gcfg.get("device", "auto")),
                        seed=seed,
                    )
                )
                device = trainer.device
                trainer.fit(
                    node_features,
                    graph.neighbour_index,
                    graph.neighbour_mask,
                    idx["train"], y[idx["train"]],
                    idx["valid"], y[idx["valid"]],
                )
                trainer.save(ckpt_path)
                save_table(pd.DataFrame(trainer.result.history), paths, f"{run_name}_gnn_history",
                           index=False)

            calibrator = ProbabilityCalibrator("isotonic").fit(
                y[idx["valid"]], trainer.predict(idx["valid"])
            )
            raw = trainer.predict(idx["test"])
            scores["C_graphsage"] = calibrator.transform(raw)
            metrics["C_graphsage"] = compute_metrics(y_test, scores["C_graphsage"])
            LOG.info("%-24s | %s", "C_graphsage", metrics["C_graphsage"].summary_line())
            gnn_summary = {
                "device": trainer.device,
                "best_epoch": getattr(trainer.result, "best_epoch", None),
                "best_valid_auc": getattr(trainer.result, "best_valid_auc", None),
                "n_node_features": int(node_features.shape[1]),
                "history": getattr(trainer.result, "history", []),
            }

    comparison = compare_models(metrics)
    save_table(comparison, paths, f"{run_name}_exp04_exp05")  # full 2015 (continuity)
    # CR-03: nothing here is selected on the test book (isotonic is fixed a priori, early
    # stopping uses validation), so the 2015-07..12 holdout is simply the headline window.
    holdout = test_window_masks(test["issue_period"])["holdout"]
    metrics_h2 = {k: compute_metrics(y_test[holdout], v[holdout]) for k, v in scores.items()}
    save_table(compare_models(metrics_h2), paths, f"{run_name}_exp04_exp05_h2")

    if not args.no_figures and scores:
        from credit_risk.evaluation.plots import reliability_diagram, roc_curves, save
        curves = {k: (y_test, v) for k, v in scores.items()}
        save(roc_curves(curves, title="Layer B arms (out-of-time)"),
             paths.results_figures / f"{run_name}_roc.png")
        save(reliability_diagram(curves), paths.results_figures / f"{run_name}_reliability.png")

    # -- verdicts --------------------------------------------------------------
    auc_a = metrics["A_tabular"].roc_auc
    auc_b = metrics["B_tabular_plus_graph"].roc_auc

    comp_macro = None
    comp_shuffled = None
    if "Control_macro_default" in scores:
        comp_macro = paired_bootstrap_calibration_comparison(
            y_test, scores["B_tabular_plus_graph"], scores["Control_macro_default"], seed=seed
        )
    if "Control_shuffled_cohort" in scores:
        comp_shuffled = paired_bootstrap_calibration_comparison(
            y_test, scores["B_tabular_plus_graph"], scores["Control_shuffled_cohort"], seed=seed
        )

    lines = [
        f"# {run_name}: EXP04 and EXP05",
        "",
        f"Out-of-time test window: {len(test):,} loans.",
        "",
        comparison.round(5).to_markdown(),
        "",
        "## EXP04 - do graph-derived cohort features add signal beyond tabular features?",
        "",
        f"Arm A (tabular) ROC-AUC {auc_a:.4f}; arm B (tabular + cohort aggregates) "
        f"{auc_b:.4f}; difference **{auc_b - auc_a:+.4f}**.",
        "",
    ]
    share = metrics["B_tabular_plus_graph"].extras.get("graph_gain_share")
    if share is not None:
        lines.append(
            f"The {len(cohort_columns)} cohort features account for {100 * share:.1f}% of total "
            f"LightGBM gain in arm B."
        )
    lines.append(
        "**Verdict.** "
        + (
            "Relational information adds measurable signal beyond the borrower's own file."
            if auc_b - auc_a > 0.002
            else "Relational information adds little or nothing beyond the borrower's own file "
            "on this dataset. That is a legitimate result, and the more likely one for a "
            "shared-attribute cohort graph: geography and employment are already partly "
            "encoded in the tabular features (`addr_state`, `emp_length`), so the cohort "
            "aggregates are largely redundant with them."
        )
    )

    if comp_macro is not None or comp_shuffled is not None:
        lines += [
            "",
            "### Non-Graph Controls & Paired Bootstrap Verification (P0-4 Audit)",
            "",
            "Does the calibration improvement of Arm B over Arm A survive non-graph controls?",
            "Two placebo arms rule out trivial non-graph explanations:",
            "1. **Control (a) - Macro default rate:** A single time-respecting recent portfolio-wide default rate column.",
            "2. **Control (b) - Shuffled cohort keys:** Cohort aggregates recomputed with `zip3` and `emp_norm` randomly permuted within issue month (preserving time and marginals but breaking cohort links).",
            "",
        ]
        if comp_macro is not None:
            lines.append(
                f"- **Arm B vs Macro Default:** Delta ECE {comp_macro['delta_ece']:+.5f} (95% CI: [{comp_macro['delta_ece_ci'][0]:.5f}, {comp_macro['delta_ece_ci'][1]:.5f}], p={comp_macro['p_val_ece']:.4f}); "
                f"Delta |O/E - 1|: {comp_macro['delta_oe_gap']:+.5f} (95% CI: [{comp_macro['delta_oe_gap_ci'][0]:.5f}, {comp_macro['delta_oe_gap_ci'][1]:.5f}], p={comp_macro['p_val_oe_gap']:.4f})."
            )
        if comp_shuffled is not None:
            lines.append(
                f"- **Arm B vs Shuffled Keys:** Delta ECE {comp_shuffled['delta_ece']:+.5f} (95% CI: [{comp_shuffled['delta_ece_ci'][0]:.5f}, {comp_shuffled['delta_ece_ci'][1]:.5f}], p={comp_shuffled['p_val_ece']:.4f}); "
                f"Delta |O/E - 1|: {comp_shuffled['delta_oe_gap']:+.5f} (95% CI: [{comp_shuffled['delta_oe_gap_ci'][0]:.5f}, {comp_shuffled['delta_oe_gap_ci'][1]:.5f}], p={comp_shuffled['p_val_oe_gap']:.4f})."
            )
        lines += [
            "",
            _control_verdict(metrics, comp_macro, comp_shuffled),
        ]


    if "C_graphsage" in metrics:
        auc_c = metrics["C_graphsage"].roc_auc
        lines += [
            "",
            "## EXP05 - does GraphSAGE add value beyond the aggregates?",
            "",
            f"Arm C (GraphSAGE) ROC-AUC {auc_c:.4f}, against arm B at {auc_b:.4f} "
            f"(**{auc_c - auc_b:+.4f}**) and arm A at {auc_a:.4f} (**{auc_c - auc_a:+.4f}**).",
            f"Trained on `{gnn_summary.get('device', device)}`, best epoch "
            f"{gnn_summary.get('best_epoch')}, {gnn_summary.get('n_node_features')} node features.",
            "",
            "**Verdict.** "
            + (
                "The GNN improves on the hand-built aggregates, so the learned message passing "
                "is capturing structure the fixed aggregation does not."
                if auc_c - auc_b > 0.002
                else "The GNN does **not** beat the hand-built cohort aggregates. This is the "
                "expected outcome and worth stating plainly: mean-aggregation over a "
                "one-hop cohort is close to what the aggregate features already compute in "
                "closed form, so the GNN is re-learning a known function from far less "
                "signal, with more parameters. On a genuine transaction network -- where "
                "multi-hop structure carries information no aggregate captures -- the "
                "comparison could well go the other way. This dataset has no such network."
            ),
            "",
            "Practical note: arm B is also the more deployable of the two. Its contribution is "
            "legible in a SHAP plot and can be given to an applicant as a reason; the GNN's "
            "neighbourhood aggregation cannot, which matters under adverse-action requirements.",
        ]

    write_report(lines, paths, f"{run_name}_exp04_exp05", source)
    save_json(
        {
            "arms": {k: v.to_dict() for k, v in metrics.items()},
            "cohort_features": list(cohort_columns),
            "gnn": gnn_summary,
        },
        paths, f"{run_name}_exp04_exp05",
    )
    write_manifest(run_name, config, seed, paths, data_source=source, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
