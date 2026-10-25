<div align="center">

# BIOGAS RS-PIM Engine

**Research pipeline for biogas output modeling, regime analysis, tail-risk estimation, and reporting**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Compile check](https://github.com/000MAK/biogas-rspim-engine/actions/workflows/compile.yml/badge.svg)](https://github.com/000MAK/biogas-rspim-engine/actions/workflows/compile.yml)

</div>

---

## Overview

BIOGAS RS-PIM is a research/academic Python project for exploring biogas plant output data through statistical modeling, regime detection, extreme-value analysis, and report generation.

The repository contains code and configuration only; it does **not** include proprietary or client data. Users must supply their own CSV time series and check that they have permission to use it.

## Research components

- **Stochastic-process modeling:** Ornstein–Uhlenbeck process parameterization.
- **Regime analysis:** HMM/GMM-based model comparisons, sensitivity sweeps, information criteria, and weighted ranking.
- **Tail-risk analysis:** Peaks-over-threshold / Generalized Pareto Distribution (GPD) compared with Lognormal, Gamma, Weibull, and Burr XII fits.
- **Risk measures:** VaR/TVaR-style estimates and bootstrap confidence intervals.
- **Reporting and exploration:** Streamlit dashboard and report exports to Excel, PDF, DOCX, LaTeX, and self-contained HTML with Plotly charts.

These components describe the project's research scope; they do not imply that every model is validated for operational or regulatory decisions.

## Repository map

| File | Responsibility |
|---|---|
| **config.py** | Central configuration: data paths, model settings, random seed, thresholds, and report names |
| **academic_data_io.py** | CSV ingestion, column detection, basic cleaning, and feature/target helpers |
| **model_physics_ml.py** | Physics-informed and ML model components |
| **model_actuarial_evt.py** | Extreme-value distribution fitting and risk metrics |
| **academic_reporter.py** | Report generation across supported formats |
| **app_defense_dashboard.py** | Streamlit research dashboard |

## Requirements

- Python 3.10 or later is recommended.
- Install the pinned dependencies from requirements.txt.
- Supply a suitable CSV time series at the path configured in config.py.
- No private or production dataset is bundled in this repository.

## Setup

Clone the repository and create a virtual environment:

    git clone https://github.com/000MAK/biogas-rspim-engine.git
    cd biogas-rspim-engine
    python -m venv .venv

Activate the environment.

**Windows PowerShell:**

    .\.venv\Scripts\Activate.ps1

**macOS / Linux:**

    source .venv/bin/activate

Install dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

## Configure data

The default paths in config.py are:

- Empirical data: data/power_ch4_30_later.csv
- Optional synthetic data: data/biogas_extended_synthetic.csv
- Output directory: output/

These files are not included. Update the configuration to match your authorized local data before running the application. The ingestion module attempts to identify common column names related to methane, power, pH, temperature, FOS/VFA, TAC/alkalinity, COD, total solids, and volatile solids. Confirm the detected columns and cleaned values before interpreting the output.

Launch the dashboard after configuration:

    streamlit run app_defense_dashboard.py

The dashboard and report pipeline may require additional data-specific preparation. A successful launch alone does not validate the statistical assumptions or the resulting risk estimates.

## Verification status

- The repository README previously documented a successful Python syntax compilation check.
- No sample dataset is provided, and end-to-end execution against real data has not been verified in the repository.
- A syntax-check workflow is included to catch Python compilation errors; it is not a substitute for unit tests, data validation, or full model evaluation.

## Interpretation and limitations

This is a research prototype, not a certified engineering or actuarial system, and not investment, operational, or regulatory advice. Results depend on data quality, preprocessing, parameter choices, model assumptions, and validation design. Before relying on a result, document data provenance, validate domain assumptions, test out-of-sample behavior, and review the full analysis with qualified domain experts.

## Related work

- [GitHub profile](https://github.com/000MAK)
- [Quantix Python utilities and ML/DL examples](https://github.com/000MAK/quantix-100-new-mini-projects)
- [Portfolio optimization research artifact](https://github.com/000MAK/PXU030Portfoliooptimization)

## License

MIT. See [LICENSE](LICENSE).
