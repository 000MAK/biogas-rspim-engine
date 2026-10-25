"""src/config.py - Frozen BiogasConfig dataclass. Single source of truth for all hyperparameters."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import pathlib

@dataclass(frozen=True)
class BiogasConfig:
    # PATHS
    EMPIRICAL_DATA: str = "data/power_ch4_30_later.csv"
    SYNTHETIC_DATA: str = "data/biogas_extended_synthetic.csv"
    OUTPUT_DIR: str = "output"

    # REPRODUCIBILITY
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.20

    # OU SDE PARAMETERS (calibrated via MLE on empirical ACF)
    OU_THETA: float = 0.35
    OU_MU: float = 58.5
    OU_SIGMA: float = 4.2
    OU_DT: float = 1.0

    # FULL K-VALUE SENSITIVITY ANALYSIS
    K_MIN: int = 2
    K_MAX: int = 8
    K_CANDIDATES: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8])

    # HMM/GMM PARAMETERS
    HMM_COVARIANCE_TYPES: List[str] = field(default_factory=lambda: ["full", "diag", "spherical"])
    HMM_N_ITER: int = 250
    HMM_TOL: float = 1e-4
    GMM_N_INIT: int = 5
    GMM_MAX_ITER: int = 200

    # BAYESIAN GMM (Dirichlet Process)
    BGM_MAX_ITER: int = 300
    BGM_WEIGHT_CONCENTRATION_PRIOR: float = 1.0

    # DBSCAN PARAMETERS
    DBSCAN_EPS: float = 0.5
    DBSCAN_MIN_SAMPLES: int = 5

    # SPECTRAL CLUSTERING
    SPECTRAL_N_NEIGHBORS: int = 10
    SPECTRAL_AFFINITY: str = "rbf"

    # TIME-SERIES K-MEANS (DTW)
    TSKM_MAX_ITER: int = 50
    TSKM_N_INIT: int = 3

    # BIOLOGICAL HYSTERESIS
    HYSTERESIS_PH_THRESHOLD: float = 6.8
    HYSTERESIS_PH_CRITICAL: float = 6.5
    MONOD_ALPHA: float = 2.0

    # EVT PARAMETERS
    EVT_THRESHOLD_QUANTILE: float = 0.90
    EVT_CONFIDENCE_LEVEL: float = 0.995
    EVT_MIN_EXCESSES: int = 30
    EVT_BOOTSTRAP_ITERATIONS: int = 2000
    EVT_DISTRIBUTIONS: List[str] = field(default_factory=lambda: ["gpd", "lognormal", "gamma", "weibull", "burr"])

    # MCDA WEIGHTS (sum=1.0)
    MCDA_WEIGHT_BIC: float = 0.30
    MCDA_WEIGHT_ICL: float = 0.25
    MCDA_WEIGHT_SILHOUETTE: float = 0.20
    MCDA_WEIGHT_PREDICTIVE_LL: float = 0.15
    MCDA_WEIGHT_CONVERGENCE: float = 0.10

    # KDE PARAMETERS
    KDE_BANDWIDTH_METHODS: List[str] = field(default_factory=lambda: ["scott", "silverman"])
    KDE_KERNEL: str = "gaussian"

    # REPORTING FILENAMES
    REPORT_EXCEL: str = "Validation_Core.xlsx"
    REPORT_PDF: str = "Methodology_Guide.pdf"
    REPORT_DOCX: str = "Executive_Summary.docx"
    REPORT_LATEX: str = "LaTeX_Methodology_Appendix.tex"
    REPORT_HTML: str = "Interactive_HTML_Report.html"

    STATE_COLORS: List[str] = field(default_factory=lambda: [
        "#2ecc71", "#27ae60", "#f1c40f", "#f39c12",
        "#e67e22", "#e74c3c", "#c0392b", "#8e44ad"
    ])

    def get_state_labels(self, k: int) -> List[str]:
        base = ["Optimal","Healthy","Mild-Stress","Stress","Severe-Stress","Critical","Failure","Catastrophic"]
        return base[:k]

    def output_path(self, filename: str) -> pathlib.Path:
        p = pathlib.Path(self.OUTPUT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p / filename

    def __post_init__(self):
        assert self.K_MIN >= 2
        assert self.K_MAX <= 8
        assert abs(self.MCDA_WEIGHT_BIC + self.MCDA_WEIGHT_ICL +
                   self.MCDA_WEIGHT_SILHOUETTE + self.MCDA_WEIGHT_PREDICTIVE_LL +
                   self.MCDA_WEIGHT_CONVERGENCE - 1.0) < 1e-6
        assert 0.0 < self.EVT_THRESHOLD_QUANTILE < 1.0
        assert self.OU_THETA > 0.0 and self.OU_SIGMA > 0.0

CONFIG = BiogasConfig()
