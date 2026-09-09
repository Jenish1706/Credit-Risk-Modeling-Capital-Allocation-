# Graph-Based Credit Risk & Reinforcement Learning Lending Decision System

This repository implements an end-to-end Machine Learning credit decision system. It is designed to answer a specific thesis:

> Can borrower-level credit-risk modelling be improved by incorporating relational financial information, and can calibrated risk predictions subsequently be used by a decision policy to optimise lending outcomes rather than merely predict default?

The system is structured in three primary layers, plus a cross-cutting trust layer, each addressing a specific part of the credit lifecycle. The governing philosophy is a *small theoretical surface area, fully explainable in an interview*. It is not a survey of techniques, but a rigorously tested pipeline.

## Repository Architecture

- **Layer A — Risk**: Predicts how likely a borrower is to default. Generates a calibrated Probability of Default (PD) using a LightGBM model compared against a Logistic Regression baseline.
- **Layer B — Graph**: Evaluates whether relational information adds signal. Uses a time-respecting cohort graph (`zip3` and `emp_norm` based) and a custom PyTorch GraphSAGE implementation to incorporate neighbour information without leaking future states.
- **Layer C — Decision**: Given a PD and economic constraints, decides the *right action*. Implements a lending simulator with conditional coupling, moving from static threshold policies to budget-aware Reinforcement Learning (Q-learning) under capital constraints.
- **Layer D — Trust**: Ensures honesty in evaluation. Features strict out-of-time splits, SHAP explanations, fairness slicing, and calibration/PSI drift monitors.

## Results Table

*(This table will be populated after running the pipeline on the full Kaggle dataset.)*

| Experiment | Focus | Key Finding |
|------------|-------|-------------|
| **EXP01** | Non-linear vs Baseline | ... |
| **EXP02** | Calibration Impact | ... |
| **EXP03** | Expected Profit vs Fixed PD | ... |
| **EXP04** | Graph Features | ... |
| **EXP05** | GraphSAGE | ... |
| **EXP06** | Budget-aware RL | ... |
| **EXP07** | Vintage Drift | ... |

## Quickstart

**Environment Warning:**
- **Smart App Control (Windows)** blocks unsigned `.pyd` files. You **must** use a Conda environment for authentic-coded packages (scipy/scikit-learn/LightGBM).
- **Windows long paths** are disabled by default. PyTorch is installed via `pip` to avoid Conda extraction errors.
- **KMP_DUPLICATE_LIB_OK** is handled automatically in `src/credit_risk/__init__.py`. Do not remove it.

**1. Setup Environment**
```bash
conda create -n credit-risk python=3.13 pandas numpy scikit-learn lightgbm shap
conda activate credit-risk
pip install torch
```

**2. Download Data**
You need a Kaggle API token (`~/.kaggle/kaggle.json`). The primary dataset is `wordsforthewise/lending-club`.
```bash
python scripts/download_data.py --config configs/data/lending_club.yaml
```
*Note: A synthetic dataset generator is included to test the pipeline without downloading the full dataset.*

**3. Run Pipeline End-to-End**
```bash
python scripts/prepare_data.py
python scripts/build_graph.py
python scripts/train_baseline.py
python scripts/train_gnn.py
python scripts/train_policy.py
python scripts/evaluate.py
```

## Documentation Map

- [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md): The project contract, dataset choices, and layer definitions. **Read this first.**
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md): Column-by-column decision rulings for leakage control.
- [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md): The Q&A companion explaining the "why" behind every technical decision.
- [`docs/GRAPH_DESIGN.md`](docs/GRAPH_DESIGN.md): Time-respecting graph and node relationships.
- [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md): Economic assumptions for the decision simulator.
- [`docs/RL_FORMULATION.md`](docs/RL_FORMULATION.md): Why capital constraints turn lending into a sequential RL problem.
