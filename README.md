# BIOGAS RS-PIM Engine

Research/academic Python engine for actuarial risk modeling of biogas plant
output, built around a 13-model comparison pipeline, extreme value theory
(EVT) tail fitting, and Solvency II-style capital metrics.

**No proprietary or client data is included in this repository.** The code
expects the user to supply their own CSV time series (see `config.py` for
the expected paths); none are shipped here.

## Project overview

- **Stochastic process modeling**: Ornstein-Uhlenbeck (OU) SDE calibration
  for mean-reverting plant output.
- **Regime detection**: Hidden Markov Models (HMM/GMM) with covariance-type
  and k-value sensitivity sweeps (k = 2..8), scored with BIC/ICL/AICc/CAIC/HQIC
  and a weighted MCDA ranking.
- **Tail risk**: Extreme Value Theory (Peaks-Over-Threshold / GPD) compared
  against Lognormal, Gamma, Weibull, and Burr XII fits, with VaR/TVaR and
  bootstrap confidence intervals.
- **Capital modeling**: Solvency II-style metrics derived from the fitted
  loss distributions.
- **Reporting**: exports to Excel (multi-sheet), PDF, DOCX, LaTeX, and a
  self-contained HTML report with embedded Plotly charts.

## Architecture

| File | Role |
|---|---|
| `config.py` | Frozen dataclass — single source of truth for all hyperparameters (OU params, k-sweep range, MCDA weights, file paths) |
| `academic_data_io.py` | Dual-dataset loader with fuzzy column detection, IQR outlier removal, provenance tests |
| `model_physics_ml.py` | Physics-informed ML models for plant output |
| `model_actuarial_evt.py` | EVT tail fitting (5-distribution comparison), Solvency II metrics |
| `academic_reporter.py` | Excel/PDF/DOCX/LaTeX/HTML report generation |
| `app_defense_dashboard.py` | Streamlit dashboard |

## Quick start

```bash
git clone https://github.com/000MAK/biogas-rspim-engine.git
cd biogas-rspim-engine
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt
# Supply your own data at the paths configured in config.py, then:
streamlit run app_defense_dashboard.py
```

## Verification status

- All modules compile cleanly (`python -m py_compile *.py`).
- Three broken multi-line string literals in `academic_reporter.py`'s LaTeX
  export (unterminated strings / invalid `\u` escape sequences from
  non-raw strings containing LaTeX commands) were found and fixed before
  publishing.
- Full runtime execution against real data has not been verified in this
  repository (no sample dataset is included). Treat this as research code
  — review and test before relying on its output.

## License

MIT — see `LICENSE`.
