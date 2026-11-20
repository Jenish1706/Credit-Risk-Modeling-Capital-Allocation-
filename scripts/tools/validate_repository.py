#!/usr/bin/env python
"""Repository validation: every reported number must trace to a current artefact.

    python scripts/tools/validate_repository.py

Run this before submitting or after any pipeline change. It is deliberately separate from the
pytest suite: the tests check that the code is correct, this checks that the *results on disk*
were produced by that code and that the README does not quote a number nothing produces.

Confirms that every number quoted in the README traces to an artefact produced by the current
code, that the pipeline chain is intact, and that the leakage controls still hold on real data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.data import schema  # noqa: E402
from credit_risk.data.splits import assert_temporal_ordering  # noqa: E402
from credit_risk.decision.expected_loss import realised_profit  # noqa: E402
from credit_risk.features.build import FeatureBuilder  # noqa: E402
from credit_risk.graph.construction import TimeRespectingGraph, assert_no_future_edges  # noqa: E402
from credit_risk.utils.cli import load_prepared  # noqa: E402
from credit_risk.utils.config import Paths  # noqa: E402

paths = Paths()
ok, fail = [], []


def check(name, condition, detail=""):
    (ok if condition else fail).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


print("=" * 78)
print("1. PIPELINE CHAIN")
print("=" * 78)
df = load_prepared(paths)
check("prepared table present", len(df) > 0, f"{len(df):,} rows")
check("graph artefact present", (paths.data_processed / "graph.npz").exists())
check("cohort features present", (paths.data_processed / "cohort_features.parquet").exists())

splits = df["split"].value_counts().to_dict()
check("all five splits present", set(splits) >= {"history", "train", "valid", "test", "monitor"},
      str({k: f"{v:,}" for k, v in splits.items()}))

print()
print("=" * 78)
print("2. LEAKAGE CONTROLS (real data)")
print("=" * 78)
assert_temporal_ordering(df)
check("temporal split ordering", True)

builder = FeatureBuilder.load(paths.artifacts / "boosting_features.pkl")
feature_names = set(builder.feature_names)
check("no post-origination column in features",
      not (feature_names & schema.FORBIDDEN_AS_FEATURES),
      f"{len(feature_names)} features")
check("primary set excludes LC underwriting",
      not (feature_names & {"grade", "sub_grade", "int_rate", "installment"}))

graph = TimeRespectingGraph.load(paths.data_processed / "graph.npz")
assert_no_future_edges(graph, df)
check("graph has no future edges", True, f"{graph.n_edges:,} edges")

resolved = df["outcome_observed_period"].notna()
issue = pd.PeriodIndex(df.loc[resolved, "issue_period"])
observed = pd.PeriodIndex(df.loc[resolved, "outcome_observed_period"])
check("no outcome resolves before issue", bool((observed > issue).all()))

monitor = df[df["split"] == "monitor"]
check("monitoring window unlabelled", bool(monitor["default"].isna().all()),
      f"{len(monitor):,} rows")

print()
print("=" * 78)
print("3. SIGNAL ARTEFACTS ALIGN WITH THE TEST SPLIT")
print("=" * 78)
test = df[df["split"] == "test"]
for name in ("boosting_pd_test", "anomaly_test", "disagreement_test"):
    path = paths.artifacts / f"{name}.npy"
    if path.exists():
        values = np.load(path)
        check(f"{name} aligned", len(values) == len(test), f"{len(values):,} vs {len(test):,}")
    else:
        check(f"{name} present", False)

print()
print("=" * 78)
print("4. README NUMBERS TRACE TO ARTEFACTS")
print("=" * 78)

exp01 = json.loads((paths.results_tables / "boosting_exp01_exp02.json").read_text())
check("EXP01 headline (H2) and full-2015 tables both exist",
      (paths.results_tables / "boosting_exp01_models_h2.csv").exists()
      and (paths.results_tables / "boosting_exp01_models.csv").exists())

cal = exp01["calibration"]
choice = json.loads((paths.results_tables / "boosting_calibration_choice.json").read_text())
check("EXP02 calibration leaves AUC unchanged (non-isotonic methods)",
      abs(cal["identity"]["roc_auc"] - cal[choice["method"]]["roc_auc"]) < 5e-4
      or choice["method"] == "isotonic",
      f"{choice['method']}: ECE {cal['identity']['ece']:.4f} -> {cal[choice['method']]['ece']:.4f}")
check("EXP02 production calibrator was selected on 2015-01..06",
      "2015-01..06" in choice["selected_on"], choice["selected_on"])

# KEEP/REJECT is decided on H1 only; the holdout confirms or not.
decisions = {}
for run in ("exp08_blindspots", "exp09_vgae", "exp10_disagreement"):
    rec = json.loads((paths.results_tables / f"{run}.json").read_text())
    h1 = rec["windows"]["selection_h1"]
    decisions[run] = rec["decision"]
    check(f"{run}: decision equals the H1 rule",
          (rec["decision"] == "KEEP") == bool(h1["keep"]) and "2015-01..06" in rec["decision_basis"],
          f"{rec['decision']} on H1 ({h1['n_strata_positive']}/{h1['n_strata']}, p={h1['sign_test_p_twosided']:.4f}); "
          f"holdout confirms: {rec['holdout_confirms']}")

exp10 = json.loads((paths.results_tables / "exp10_disagreement.json").read_text())
check("EXP10 log-odds less PD-confounded than probability scale",
      exp10["scale"] == "logodds"
      and abs(exp10["spearman_with_pd_logodds"]) < abs(exp10["spearman_with_pd_probability"]),
      f"{exp10['spearman_with_pd_logodds']:+.3f} vs {exp10['spearman_with_pd_probability']:+.3f}")

exp11 = json.loads((paths.results_tables / "exp11_swaps.json").read_text())
a = next(p for p in exp11["portfolios"] if p["policy"] == "A_pd_ranked")
b = next(p for p in exp11["portfolios"] if p["policy"] == "B_blindspot_aware")
kept = [n for n, run in (("anomaly", "exp08_blindspots"), ("disagreement", "exp10_disagreement"))
        if decisions[run] == "KEEP"]
check("EXP11 acts only on signals that were KEEP on H1",
      sorted(exp11["signals_used"]) == sorted(kept), f"used {exp11['signals_used']}, kept {kept}")
check("EXP11 return-on-capital change is immaterial (<50bp)",
      abs(b["return_on_capital"] - a["return_on_capital"]) < 0.005,
      f"{(b['return_on_capital'] - a['return_on_capital']):+.4f}")
check("EXP11 default rate is lower for the signal book",
      b["default_rate"] < a["default_rate"], f"{a['default_rate']:.4f} -> {b['default_rate']:.4f}")

exp06 = json.loads((paths.results_tables / "exp06_rl_exp03_exp06.json").read_text())
conditions = exp06["exp06"]
check("EXP06 trained on a disjoint pool",
      exp06.get("training_pool") == "validation window", str(exp06.get("training_pool")))
sweep = pd.read_csv(paths.results_tables / "exp06_rl_exp06_sweep.csv")
check("EXP06 reports approval rate and capital for every policy",
      {"approval_rate", "capital_deployed"} <= set(sweep.columns) and sweep["approval_rate"].notna().all())
check("EXP06 has no policy called an oracle ceiling",
      not sweep["policy"].str.contains("oracle_ceiling").any())
unc = pd.read_csv(paths.results_tables / "exp06_rl_exp06_diff_unconstrained.csv", index_col=0)
learners = [n for n in unc.index if any(m in n for m in ("q(", "linucb", "thompson"))]
check("EXP06 control: no learner significantly beats the myopic rule with unconstrained capital",
      bool((unc.loc[learners, "ci_low"] <= 0).all()), f"{len(learners)} learners checked")

print()
print("=" * 78)
print("5. ECONOMICS ARE OBSERVED, NOT ASSUMED")
print("=" * 78)
profit = realised_profit(test)
funded = pd.to_numeric(test["funded_amnt"], errors="coerce")
check("observed book profit computable", np.isfinite(profit.sum()),
      f"${profit.sum():,.0f} on ${funded.sum():,.0f}")

defaulted = pd.to_numeric(test["default"], errors="coerce") > 0.5
back = (pd.to_numeric(test["total_rec_prncp"], errors="coerce").fillna(0)
        + pd.to_numeric(test["recoveries"], errors="coerce").fillna(0)
        - pd.to_numeric(test["collection_recovery_fee"], errors="coerce").fillna(0))
lgd = (1 - back / funded).clip(0, 1)[defaulted].mean()
check("empirical LGD ~0.518 (measured, not assumed)", abs(lgd - 0.518) < 0.01, f"{lgd:.4f}")

cashflow = exp11["cashflow_model"]
check("cashflow model fitted from data",
      cashflow["n_good"] > 1000 and cashflow["n_bad"] > 1000,
      f"n_good={cashflow['n_good']:,} n_bad={cashflow['n_bad']:,}")

print()
print("=" * 78)
print("6. NO STALE OUTPUT PRESENTED AS CURRENT")
print("=" * 78)
for removed in ("cohort_graph_exp09_uncertainty.md", "boosting_exp11_swaps.md",
                "boosting_exp08_blindspots.md", "boosting_exp10_disagreement.md"):
    check(f"superseded {removed} removed",
          not (ROOT / "results/reports" / removed).exists())

manifests = list((ROOT / "results/reports").glob("*_manifest.json"))
sources = {m.stem: json.loads(m.read_text())["data_source"] for m in manifests}
synthetic = [k for k, v in sources.items() if v == "synthetic"]
check("no committed run manifest is synthetic", not synthetic, str(synthetic))

print()
print("=" * 78)
print(f"RESULT: {len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILURES:")
    for f in fail:
        print("  -", f)
print("=" * 78)
sys.exit(1 if fail else 0)
