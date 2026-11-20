# 06 · Runbook

Everything here is for Windows PowerShell with a conda environment. Run every command from the
repository root. Times were measured on the development machine (20 CPU threads, RTX 3060 Laptop
GPU) and will differ on yours.

## 1. Environment

The reference environment is Python 3.13 with PyTorch 2.9 (CUDA 12.9), LightGBM 4.6, scikit-learn
1.9, pandas 3.0 and numpy 2.4.

```powershell
conda create -n credit-risk python=3.13 numpy pandas scipy scikit-learn pyarrow lightgbm shap matplotlib pyyaml pytest numba
conda activate credit-risk
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install -e . --no-deps
```

Torch is optional. Without it the code runs and the two torch-dependent tests skip with a reason.
CPU-only torch is enough for the tests; the GNN and VGAE are much faster on a GPU.

Check which interpreter you are using before anything else:

```powershell
python --version
python -c "import credit_risk, lightgbm, pandas; print(credit_risk.__file__)"
```

## 2. Tests

```powershell
python -m pytest -q
```

About 25 seconds. Torch-dependent tests skip when torch is missing (see
[07_TESTING.md](07_TESTING.md)). Run from the repository root; some tests import `scripts.*`.

Fresh-clone check (proves the committed tree passes without your local data):

```powershell
git clone --branch audit/stage-a . $env:TEMP\credit-risk-fresh
$env:PYTHONPATH = "$env:TEMP\credit-risk-fresh\src"
python -m pytest -q $env:TEMP\credit-risk-fresh\tests
Remove-Item Env:PYTHONPATH
```

`PYTHONPATH` is needed because an editable install points at the original checkout, so a clone
would otherwise import the wrong code.

## 3. Re-render the tables (no data needed)

```powershell
python scripts/render_readme.py
```

Regenerates the README results table and prose and every named BEGIN/END marker block in
`docs/guide/` from `results/tables/`. A second run changes nothing. A test fails if a rendered region
and the artifacts disagree.

## 4. The full pipeline

Needs the raw data. Either set a Kaggle token (`%USERPROFILE%\.kaggle\kaggle.json`) and download the
real file, or build a synthetic extract (synthetic results are stamped and never reported as
findings).

```powershell
python scripts/download_data.py --config configs/data/lending_club.yaml
python scripts/download_data.py --synthetic --n-loans 200000   # alternative: no credentials
```

Warning: both write into `data/raw/`, and `prepare_data.py` overwrites
`data/processed/loans.parquet`. Do not run the synthetic path in a checkout whose processed data you
want to keep.

Commands, in dependency order, with measured run time and what each produces:

| # | Command | Time | Produces |
|---|---|---|---|
| 1 | `python scripts/prepare_data.py --config configs/data/lending_club.yaml --max-threads 10` | 1.2 min | `data/processed/loans.parquet`, `results/reports/data_profile.md`, `results/tables/data_profile.json` |
| 2 | `python scripts/build_graph.py --config configs/graph/cohort_graph.yaml` | 16 min | `data/processed/graph.npz`, `cohort_features.parquet` |
| 3 | `python scripts/train_baseline.py --config configs/models/boosting.yaml` | 3 min | `artifacts/boosting_*`, `boosting_pd_test.npy`, EXP01/EXP02 tables, `boosting_calibration_choice.json` |
| 4 | `python scripts/train_gnn.py --config configs/graph/cohort_graph.yaml` | 10 min | `cohort_graph_exp04_exp05*.csv`, GraphSAGE weights |
| 5 | `python scripts/build_cohort_controls.py` | 23 min | `cohort_controls_exp04*.csv/json` (five seeds) |
| 6 | `python scripts/analyze_blind_spots.py --config configs/experiments/exp08_blindspots.yaml` | 6 min | `exp08_blindspots*`, `anomaly_test.npy` |
| 7 | `python scripts/train_vgae.py --config configs/experiments/exp09_vgae.yaml` | **94 min** | `exp09_vgae*`, `vgae_sigma_test.npy` |
| 8 | `python scripts/analyze_disagreement.py --config configs/experiments/exp10_disagreement.yaml` | 1.5 min | `exp10_disagreement*`, `disagreement_test.npy` |
| 9 | `python scripts/analyze_decision_swaps.py --config configs/experiments/exp11_swaps.yaml` | 1.5 min | `exp11_swaps*` |
| 10 | `python scripts/train_policy.py --config configs/experiments/exp06_rl.yaml` | 12 min | `exp06_rl_*` (headline) |
| 11 | `python scripts/train_policy.py --config configs/experiments/exp06_rl.yaml --multi-exposure` | 14 min | `exp06_rl_sens_*` (sensitivity) |
| 12 | `python scripts/evaluate.py --config configs/models/boosting.yaml` | 1.5 min | `boosting_exp07_*`, fairness, SHAP |
| 13 | `python scripts/analyze_deployment_realism.py` | 1 min | `deployment_realism_p1_5.csv` |
| 14 | `python scripts/analyze_profit_calibration.py --config configs/experiments/exp06_rl.yaml` | 3 min | `profit_calibration_*.csv` |
| 15 | `python scripts/analyze_apr_in_pd.py --config configs/experiments/exp06_rl.yaml` | 5 min | `apr_in_pd_*.csv` |
| 16 | `python scripts/render_readme.py` | seconds | README and guide blocks |

Total for steps 1 to 16: about 3.2 hours, of which the VGAE is about 90 minutes. Steps 6 to 9 need steps 2 and 3
first (graph artifacts and `boosting_pd_test.npy`); steps 10, 11, 14 and 15 need step 3. Step 9 reads the KEEP/REJECT
decision files written by steps 6 and 8, so run it after them. `--max-threads N` stops a long run
monopolising the machine.

Every selection reads only 2015-01…06 rows (see [04_METHODOLOGY.md](04_METHODOLOGY.md)), and each script writes a
manifest (`results/reports/*_manifest.json`) recording config, seed and data source.

Steps 1 and 2 were re-run from the raw file in a fresh clone and reproduced `loans.parquet` (1,609,607
rows by 162 columns), `cohort_features.parquet` and every array in `graph.npz` exactly.

## 5. Verification status of the commands above

Each command was run once on this branch (2026-11-19 to 2026-11-20), except where marked.

| Command | Status |
|---|---|
| `python -m pytest -q`, fresh-clone test with `PYTHONPATH` (PowerShell) | executed |
| `python scripts/render_readme.py` | executed |
| `python scripts/check_docs.py` | executed |
| steps 1 and 2 (`prepare_data`, `build_graph`) | executed in a fresh clone with a copy of the raw file; outputs identical |
| steps 3 to 15 | executed with the current code (steps 6 to 9 and 12 in one chain run; 10, 11, 14 and 15 individually) |
| `conda create ...`, `pip install torch ...`, `pip install -e . --no-deps` | **not re-run**: the environment already exists; needs conda and network access |
| `python scripts/download_data.py` (real data) | **requires a Kaggle token**; would write the raw CSV and `checksums.json` |
| `python scripts/download_data.py --synthetic --n-loans 200000` | **not run**, because it writes into `data/raw/` and would replace the processed data of this checkout; it produces a synthetic raw file |
| `python scripts/validate_repository.py` | executed (needs the real processed data and the artifacts; all checks pass) |

## 6. Where each README number comes from

| README row | Script | Artifact |
|---|---|---|
| EXP01 | `train_baseline.py` | `boosting_exp01_models_h2.csv`, `boosting_exp01_models.csv` |
| EXP02 | `train_baseline.py` | `boosting_exp02_calibration_h2.csv`, `boosting_calibration_choice.json` |
| EXP03 | `train_policy.py`, `analyze_profit_calibration.py` | `exp06_rl_exp03_differences.csv`, `profit_calibration_rules.csv` |
| EXP04 | `build_cohort_controls.py`, `train_gnn.py` | `cohort_controls_exp04.json`, `cohort_controls_exp04_seed_*.csv` |
| EXP05 | `train_gnn.py` | `cohort_graph_exp04_exp05_h2.csv` |
| EXP06 | `train_policy.py` | `exp06_rl_exp06_sweep.csv`, `exp06_rl_exp06_diff_*.csv`, `exp06_rl_exp06_diffthr_*.csv` |
| EXP07 | `evaluate.py` | `boosting_exp07_calibration_drift*.csv` |
| EXP08/09/10 | `analyze_blind_spots.py`, `train_vgae.py`, `analyze_disagreement.py` | `exp08_blindspots.json`, `exp09_vgae.json`, `exp10_disagreement.json` |
| EXP11 | `analyze_decision_swaps.py` | `exp11_swaps_portfolios.csv`, `exp11_swaps.json` |
| Profit audit | `analyze_profit_calibration.py` | `profit_calibration_*.csv` |
| APR ablation | `analyze_apr_in_pd.py` | `apr_in_pd_*.csv` |
| Data tables | `prepare_data.py` | `data_profile.json`, `data/processed/loans.parquet` |

## 7. Troubleshooting

Each entry is a problem that was actually hit on this project.

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: DLL load failed` for scipy, scikit-learn or LightGBM | Windows Smart App Control blocks unsigned `.pyd` files from PyPI wheels | use a **conda** environment; Anaconda's packages are signed |
| conda cannot install `pytorch` (path too long) | Windows long paths are disabled | install torch from pip (`pip install torch --index-url ...`) |
| OpenMP "duplicate library" abort when torch and MKL meet | two OpenMP runtimes | handled by `KMP_DUPLICATE_LIB_OK` in `src/credit_risk/__init__.py`; do not remove it |
| `ModuleNotFoundError: lightgbm` (or pandas) | `python` resolved to the system Python 3.9, not the conda env | `conda activate credit-risk`, or call the env's `python.exe` by full path |
| a fresh clone runs the old code | the editable install points at the original checkout | set `PYTHONPATH` to the clone's `src` (section 2) |
| `fatal: Unable to create '.git/index.lock'` | an earlier git process was interrupted | confirm no git process is running, then delete `.git/index.lock` |
| `warning: CRLF will be replaced by LF` on commit | files are CRLF on disk, LF in the repository | harmless |
| a script says an artifact is missing and names another script | pipeline order (section 4) | run the named script first |
| torch-dependent tests report `SKIPPED` | torch not installed | expected; install torch to run them |
| `ModuleNotFoundError: scripts` when running a test | pytest started outside the repository root | run from the repository root |
| README test fails with "differs from rendered artifact values" | a table or an artifact changed | run `python scripts/render_readme.py` and review the diff |
