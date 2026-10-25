"""src/academic_data_io.py - Dual dataset loader with dynamic column detection and provenance tests."""
from __future__ import annotations
import re, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from typing import Dict, List, Optional, Tuple
from config import CONFIG, BiogasConfig

# Fuzzy column keyword maps
_COL_PATTERNS = {
    "ch4":   re.compile(r"ch4|methane|metan|biogas_ch4|gas_ch4", re.I),
    "power": re.compile(r"power|watt|kw|elektrik|energy|elec", re.I),
    "ph":    re.compile(r"^ph$|ph_val|reaktor_ph|ph_mes", re.I),
    "fos":   re.compile(r"fos|vfa|volatile_acid", re.I),
    "tac":   re.compile(r"tac|alkalinity|bicarb", re.I),
    "temp":  re.compile(r"temp|sicaklik|temperature", re.I),
    "cod":   re.compile(r"cod|chemical_oxygen", re.I),
    "ts":    re.compile(r"ts|total_solid", re.I),
    "vs":    re.compile(r"vs|volatile_solid", re.I),
}


class AcademicDataIngestion:
    """Dual-dataset loader with fuzzy column detection, IQR outlier removal, and provenance testing."""

    def __init__(self, cfg: BiogasConfig = CONFIG):
        self.cfg = cfg
        self.empirical_df: Optional[pd.DataFrame] = None
        self.synthetic_df: Optional[pd.DataFrame] = None
        self.emp_cols: Dict[str, str] = {}
        self.syn_cols: Dict[str, str] = {}

    # ---- Column detection -------------------------------------------------
    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for role, pat in _COL_PATTERNS.items():
            for col in df.columns:
                if pat.search(str(col)):
                    mapping[role] = col; break
        return mapping

    # ---- Physical constraints + imputation --------------------------------
    def _clean(self, df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
        df = df.copy()
        for role in ("ch4", "power", "fos", "tac", "cod", "ts", "vs"):
            c = cols.get(role)
            if c and c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df.loc[df[c] < 0, c] = np.nan
        numeric_cols = df.select_dtypes(include=np.number).columns
        for c in numeric_cols:
            Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 3.0*IQR, Q3 + 3.0*IQR
            df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
        df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(df[numeric_cols].median())
        return df

    # ---- Loaders ----------------------------------------------------------
    def load_empirical(self) -> pd.DataFrame:
        path = Path(self.cfg.EMPIRICAL_DATA)
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python")
                if len(df.columns) > 1: break
            except Exception:
                continue
        self.emp_cols = self._detect_columns(df)
        df = self._clean(df, self.emp_cols)
        self.empirical_df = df.reset_index(drop=True)
        return self.empirical_df

    def load_synthetic(self) -> Optional[pd.DataFrame]:
        path = Path(self.cfg.SYNTHETIC_DATA)
        if not path.exists():
            self.synthetic_df = None; return None
        try:
            df = pd.read_csv(path, engine="python")
            self.syn_cols = self._detect_columns(df)
            df = self._clean(df, self.syn_cols)
            self.synthetic_df = df.reset_index(drop=True)
        except Exception:
            self.synthetic_df = None
        return self.synthetic_df

    def load_both(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        return self.load_empirical(), self.load_synthetic()

    # ---- Feature matrix helpers ------------------------------------------
    def get_feature_matrix(self, which: str = "empirical") -> np.ndarray:
        df = self.empirical_df if which == "empirical" else self.synthetic_df
        cols_map = self.emp_cols if which == "empirical" else self.syn_cols
        feature_roles = ["ch4", "power", "fos", "ph", "temp", "tac", "cod", "ts", "vs"]
        use_cols = [cols_map[r] for r in feature_roles if r in cols_map and cols_map[r] in df.columns]
        if not use_cols:
            use_cols = list(df.select_dtypes(include=np.number).columns)
        X = df[use_cols].values.astype(float)
        return X

    def get_target_series(self, which: str = "empirical") -> np.ndarray:
        df = self.empirical_df if which == "empirical" else self.synthetic_df
        cols_map = self.emp_cols if which == "empirical" else self.syn_cols
        for role in ("ch4", "power"):
            c = cols_map.get(role)
            if c and c in df.columns:
                return df[c].values.astype(float)
        return df.select_dtypes(include=np.number).iloc[:, 0].values

    def get_ph_series(self, which: str = "empirical") -> Optional[np.ndarray]:
        df = self.empirical_df if which == "empirical" else self.synthetic_df
        cols_map = self.emp_cols if which == "empirical" else self.syn_cols
        c = cols_map.get("ph")
        if c and c in df.columns:
            return df[c].values.astype(float)
        return None

    def get_train_test_split(self, which: str = "empirical") -> Tuple[np.ndarray, np.ndarray]:
        X = self.get_feature_matrix(which)
        n = len(X); split = int(n * (1 - self.cfg.TEST_SIZE))
        return X[:split], X[split:]

    # ---- Summary stats ---------------------------------------------------
    def summary_stats(self, which: str = "empirical") -> pd.DataFrame:
        df = self.empirical_df if which == "empirical" else self.synthetic_df
        if df is None: return pd.DataFrame()
        num = df.select_dtypes(include=np.number)
        rows = []
        for col in num.columns:
            v = num[col].dropna().values
            if len(v) < 4: continue
            try: jb_p = float(stats.jarque_bera(v).pvalue)
            except: jb_p = float("nan")
            rows.append({"Column": col, "N": len(v), "Mean": round(v.mean(),4),
                         "Std": round(v.std(),4), "Skewness": round(float(stats.skew(v)),4),
                         "Kurtosis": round(float(stats.kurtosis(v)),4), "JB_pvalue": round(jb_p,4),
                         "Min": round(v.min(),4), "Max": round(v.max(),4)})
        return pd.DataFrame(rows)

    # ---- Quality audit ---------------------------------------------------
    def get_quality_report(self) -> pd.DataFrame:
        if self.empirical_df is None: return pd.DataFrame()
        rows = []
        for col in self.empirical_df.select_dtypes(include=np.number).columns:
            orig = self.empirical_df[col]
            rows.append({"Column": col, "N_total": len(orig),
                         "N_missing": int(orig.isna().sum()),
                         "Pct_missing": round(orig.isna().mean()*100,2),
                         "Detected_role": next((r for r,c in self.emp_cols.items() if c==col),"unknown")})
        return pd.DataFrame(rows)

    # ---- Provenance tests ------------------------------------------------
    def _hist_kl(self, p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
        lo = min(p.min(), q.min()); hi = max(p.max(), q.max())
        hp, _ = np.histogram(p, bins=bins, range=(lo,hi), density=True)
        hq, _ = np.histogram(q, bins=bins, range=(lo,hi), density=True)
        eps = 1e-10; hp = hp + eps; hq = hq + eps
        hp /= hp.sum(); hq /= hq.sum()
        return float(np.sum(hp * np.log(hp/hq)))

    def provenance_tests(self) -> pd.DataFrame:
        if self.empirical_df is None or self.synthetic_df is None:
            return pd.DataFrame({"test": ["No synthetic data"], "statistic": [None], "p_value": [None]})
        emp_roles = ["ch4","power","ph","temp","fos"]
        rows = []
        for role in emp_roles:
            ec = self.emp_cols.get(role); sc = self.syn_cols.get(role)
            if not ec or not sc: continue
            if ec not in self.empirical_df.columns or sc not in self.synthetic_df.columns: continue
            E = self.empirical_df[ec].dropna().values.astype(float)
            S = self.synthetic_df[sc].dropna().values.astype(float)
            if len(E)<10 or len(S)<10: continue
            # KS test
            ks = stats.ks_2samp(E, S); rows.append({"feature":role,"test":"KS-2samp","stat":round(ks.statistic,4),"p_value":round(ks.pvalue,4)})
            # Anderson-Darling
            try:
                ad = stats.anderson_ksamp([E,S]); rows.append({"feature":role,"test":"Anderson-Darling","stat":round(ad.statistic,4),"p_value":round(float(ad.significance_level),4)})
            except: pass
            # Mann-Whitney
            mw = stats.mannwhitneyu(E, S, alternative="two-sided"); rows.append({"feature":role,"test":"Mann-Whitney","stat":round(mw.statistic,4),"p_value":round(mw.pvalue,4)})
            # Levene
            lev = stats.levene(E, S); rows.append({"feature":role,"test":"Levene","stat":round(lev.statistic,4),"p_value":round(lev.pvalue,4)})
            # KL divergence
            kl = self._hist_kl(E, S); rows.append({"feature":role,"test":"KL-divergence","stat":round(kl,4),"p_value":None})
            # JS divergence
            lo=min(E.min(),S.min()); hi=max(E.max(),S.max())
            hp,_=np.histogram(E,bins=50,range=(lo,hi),density=True); hq,_=np.histogram(S,bins=50,range=(lo,hi),density=True)
            js=float(jensenshannon(hp+1e-10,hq+1e-10)); rows.append({"feature":role,"test":"JS-divergence","stat":round(js,4),"p_value":None})
            # Wasserstein
            ws=float(wasserstein_distance(E,S)); rows.append({"feature":role,"test":"Wasserstein-1","stat":round(ws,4),"p_value":None})
        return pd.DataFrame(rows)
