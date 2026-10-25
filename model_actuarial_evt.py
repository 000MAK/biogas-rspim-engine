"""src/model_actuarial_evt.py - 5-distribution EVT engine: GPD/LN/Gamma/Weibull/Burr + Solvency II."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from config import CONFIG, BiogasConfig


# ===========================================================================
# LOSS PROFILE GENERATORS
# ===========================================================================
def generate_loss_profile(series: np.ndarray, profile_type: str = "downside") -> np.ndarray:
    """Convert raw series to a non-negative loss vector."""
    if profile_type == "negative_returns":
        ret = np.diff(series) / (np.abs(series[:-1]) + 1e-8)
        return np.maximum(-ret, 0)
    elif profile_type == "residuals":
        mu = series.mean()
        return np.abs(series - mu)
    elif profile_type == "capacity_deficit":
        cap = np.percentile(series, 95)
        return np.maximum(cap - series, 0)
    else:  # downside
        mu = series.mean()
        return np.maximum(mu - series, 0)


# ===========================================================================
# INDIVIDUAL DISTRIBUTION FITTERS
# ===========================================================================
def _gof_tests(exceedances: np.ndarray, dist_cls, params: tuple) -> dict:
    """KS + Anderson-Darling + Cramer-von Mises goodness-of-fit."""
    try: ks_stat, ks_p = stats.kstest(exceedances, lambda x: dist_cls.cdf(x, *params))
    except: ks_stat, ks_p = float("nan"), float("nan")
    try:
        ad_res = stats.anderson(exceedances)
        ad_stat = float(ad_res.statistic)
    except: ad_stat = float("nan")
    try:
        cvm = stats.cramervonmises(exceedances, lambda x: dist_cls.cdf(x, *params))
        cvm_stat, cvm_p = float(cvm.statistic), float(cvm.pvalue)
    except: cvm_stat, cvm_p = float("nan"), float("nan")
    return {"KS_stat": round(ks_stat,4), "KS_p": round(ks_p,4),
            "AD_stat": round(ad_stat,4), "CvM_stat": round(cvm_stat,4), "CvM_p": round(cvm_p,4)}


class ActuarialEVTBackend:
    """5-distribution POT-EVT engine with bootstrap VaR/TVaR CIs and Solvency II metrics."""

    def __init__(self, cfg: BiogasConfig = CONFIG):
        self.cfg = cfg
        self.losses_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None
        self.exceedances_: Optional[np.ndarray] = None
        self.gpd_params_: Optional[tuple] = None

    def set_losses(self, losses: np.ndarray) -> None:
        self.losses_ = losses
        self.threshold_ = float(np.quantile(losses, self.cfg.EVT_THRESHOLD_QUANTILE))
        self.exceedances_ = losses[losses > self.threshold_] - self.threshold_

    # ---- Individual distribution fits ----------------------------------------
    def fit_gpd(self) -> dict:
        exc = self.exceedances_
        xi, loc, sigma = stats.genpareto.fit(exc, floc=0)
        self.gpd_params_ = (xi, loc, sigma)
        gof = _gof_tests(exc, stats.genpareto, (xi, loc, sigma))
        n_params = 2
        ll = float(np.sum(stats.genpareto.logpdf(exc, xi, loc, sigma)))
        return {"distribution":"GPD","xi":round(xi,4),"sigma":round(sigma,4),
                "log_ll":round(ll,4),"n_params":n_params,
                "AIC":round(2*n_params-2*ll,2),"BIC":round(n_params*np.log(len(exc))-2*ll,2),
                **gof}

    def fit_lognormal(self) -> dict:
        exc = self.exceedances_; params = stats.lognorm.fit(exc); gof = _gof_tests(exc, stats.lognorm, params)
        ll=float(np.sum(stats.lognorm.logpdf(exc,*params))); n_p=len(params)
        return {"distribution":"Lognormal","log_ll":round(ll,4),"n_params":n_p,
                "AIC":round(2*n_p-2*ll,2),"BIC":round(n_p*np.log(len(exc))-2*ll,2),
                "xi":float("nan"),"sigma":float("nan"),**gof}

    def fit_gamma(self) -> dict:
        exc = self.exceedances_; params = stats.gamma.fit(exc); gof = _gof_tests(exc, stats.gamma, params)
        ll=float(np.sum(stats.gamma.logpdf(exc,*params))); n_p=len(params)
        return {"distribution":"Gamma","log_ll":round(ll,4),"n_params":n_p,
                "AIC":round(2*n_p-2*ll,2),"BIC":round(n_p*np.log(len(exc))-2*ll,2),
                "xi":float("nan"),"sigma":float("nan"),**gof}

    def fit_weibull(self) -> dict:
        exc = self.exceedances_; params = stats.weibull_min.fit(exc); gof = _gof_tests(exc, stats.weibull_min, params)
        ll=float(np.sum(stats.weibull_min.logpdf(exc,*params))); n_p=len(params)
        return {"distribution":"Weibull","log_ll":round(ll,4),"n_params":n_p,
                "AIC":round(2*n_p-2*ll,2),"BIC":round(n_p*np.log(len(exc))-2*ll,2),
                "xi":float("nan"),"sigma":float("nan"),**gof}

    def fit_burr(self) -> dict:
        exc = self.exceedances_
        try: params = stats.burr12.fit(exc)
        except: return {"distribution":"Burr","log_ll":float("nan"),"n_params":4,
                        "AIC":float("nan"),"BIC":float("nan"),"xi":float("nan"),"sigma":float("nan"),
                        "KS_stat":float("nan"),"KS_p":float("nan"),"AD_stat":float("nan"),
                        "CvM_stat":float("nan"),"CvM_p":float("nan")}
        gof = _gof_tests(exc, stats.burr12, params)
        ll=float(np.sum(stats.burr12.logpdf(exc,*params))); n_p=len(params)
        return {"distribution":"Burr","log_ll":round(ll,4),"n_params":n_p,
                "AIC":round(2*n_p-2*ll,2),"BIC":round(n_p*np.log(len(exc))-2*ll,2),
                "xi":float("nan"),"sigma":float("nan"),**gof}

    def compare_tail_models(self) -> pd.DataFrame:
        """Fit all 5 distributions, return ranked comparison DataFrame."""
        rows = []
        for fn in [self.fit_gpd, self.fit_lognormal, self.fit_gamma, self.fit_weibull, self.fit_burr]:
            try: rows.append(fn())
            except Exception as e: rows.append({"distribution":fn.__name__,"status":str(e)})
        df = pd.DataFrame(rows)
        df["AIC_rank"] = pd.to_numeric(df.get("AIC",float("nan")),errors="coerce").rank()
        df["KS_rank"] = pd.to_numeric(df.get("KS_stat",float("nan")),errors="coerce").rank()
        return df

    # ---- VaR/TVaR calculations -----------------------------------------------
    def _gpd_var(self, p: float) -> float:
        xi, _, sigma = self.gpd_params_
        u = self.threshold_; n = len(self.losses_); n_u = len(self.exceedances_)
        if abs(xi) < 1e-10:
            return u + sigma * np.log(n*(1-p)/n_u)
        return u + (sigma/xi)*((n*(1-p)/n_u)**(-xi) - 1)

    def _gpd_tvar(self, var: float) -> float:
        xi, _, sigma = self.gpd_params_; u = self.threshold_
        return var/(1-xi) + (sigma - xi*u)/(1-xi)

    def _generic_var(self, dist_cls, params: tuple, p: float) -> float:
        exc = self.exceedances_
        return self.threshold_ + float(dist_cls.ppf(p, *params))

    def _generic_tvar(self, dist_cls, params: tuple, p: float, var_exc: float) -> float:
        xs = np.linspace(var_exc, dist_cls.ppf(0.9999, *params), 1000)
        try: return self.threshold_ + float(np.trapz(xs*dist_cls.pdf(xs,*params),xs)/(1-p))
        except: return float("nan")

    def calculate_var_tvar(self, p: Optional[float] = None) -> dict:
        p = p or self.cfg.EVT_CONFIDENCE_LEVEL
        var = self._gpd_var(p); tvar = self._gpd_tvar(var)
        return {"confidence":p,"VaR":round(var,4),"TVaR":round(tvar,4),
                "threshold":round(self.threshold_,4),"n_exceedances":len(self.exceedances_)}

    # ---- Bootstrap VaR/TVaR CI -----------------------------------------------
    def bootstrap_var_tvar_ci(self, B: Optional[int] = None, p: Optional[float] = None) -> dict:
        B = B or self.cfg.EVT_BOOTSTRAP_ITERATIONS; p = p or self.cfg.EVT_CONFIDENCE_LEVEL
        rng = np.random.default_rng(self.cfg.RANDOM_SEED)
        exc = self.exceedances_; u = self.threshold_; n = len(self.losses_); n_u = len(exc)
        vars_b, tvars_b = [], []
        for _ in range(B):
            samp = rng.choice(exc, size=len(exc), replace=True)
            try:
                xi, _, sig = stats.genpareto.fit(samp, floc=0)
                if abs(xi) < 1e-10: v = u + sig*np.log(n*(1-p)/n_u)
                else: v = u + (sig/xi)*((n*(1-p)/n_u)**(-xi)-1)
                t = v/(1-xi) + (sig - xi*u)/(1-xi)
                if np.isfinite(v): vars_b.append(v)
                if np.isfinite(t): tvars_b.append(t)
            except: continue
        def ci(arr, conf=0.95):
            if len(arr)<10: return (float("nan"),float("nan"))
            a=(1-conf)/2
            return (round(float(np.quantile(arr,a)),4), round(float(np.quantile(arr,1-a)),4))
        return {"B":B,"VaR_95CI":ci(vars_b),"TVaR_95CI":ci(tvars_b),
                "VaR_mean":round(float(np.mean(vars_b)),4) if vars_b else float("nan"),
                "TVaR_mean":round(float(np.mean(tvars_b)),4) if tvars_b else float("nan")}

    # ---- Mean Excess Function ------------------------------------------------
    def mean_excess_function(self, n_thresholds: int = 50) -> pd.DataFrame:
        losses = self.losses_
        u_vals = np.linspace(np.quantile(losses,0.5), np.quantile(losses,0.99), n_thresholds)
        rows = []
        for u in u_vals:
            exc = losses[losses>u]-u
            if len(exc)>=5:
                rows.append({"threshold":round(float(u),4),"mean_excess":round(float(exc.mean()),4),
                             "n_exceed":len(exc)})
        return pd.DataFrame(rows)

    # ---- Hill Estimator -------------------------------------------------------
    def hill_estimator(self, k_range: Optional[range] = None) -> pd.DataFrame:
        """Hill estimator for tail index stability."""
        losses_sorted = np.sort(self.losses_)[::-1]
        n = len(losses_sorted)
        if k_range is None: k_range = range(10, min(n//2, 200))
        rows = []
        for k in k_range:
            if k>=n: continue
            hill = float(np.mean(np.log(losses_sorted[:k]/losses_sorted[k])))
            rows.append({"k":k, "hill_alpha":round(1.0/hill,4) if hill>0 else float("nan")})
        return pd.DataFrame(rows)

    # ---- Pickands Estimator ---------------------------------------------------
    def pickands_estimator(self, k_range: Optional[range] = None) -> pd.DataFrame:
        losses_sorted = np.sort(self.losses_)[::-1]
        n = len(losses_sorted)
        if k_range is None: k_range = range(10, min(n//4, 100))
        rows = []
        for k in k_range:
            if 4*k>=n: continue
            try:
                gamma = (1/np.log(2))*np.log((losses_sorted[k]-losses_sorted[2*k])/(losses_sorted[2*k]-losses_sorted[4*k]))
                rows.append({"k":k,"pickands_gamma":round(float(gamma),4)})
            except: pass
        return pd.DataFrame(rows)

    # ---- Solvency II ---------------------------------------------------------
    def calculate_solvency_premium(self, pure_loss: float, tvar: float) -> float:
        return pure_loss + 0.06 * (tvar - pure_loss)

    def calculate_epd_ratio(self, var: float) -> float:
        """Expected Policyholder Deficit = E[max(L-VaR,0)] / E[L]."""
        losses = self.losses_
        epd = float(np.mean(np.maximum(losses - var, 0)))
        mu = float(np.mean(losses))
        return round(epd/max(mu,1e-8), 6)

    def full_solvency_report(self) -> dict:
        """Complete Solvency II metrics: SCR, premium, EPD, bootstrap CIs."""
        p = self.cfg.EVT_CONFIDENCE_LEVEL
        var = self._gpd_var(p); tvar = self._gpd_tvar(var)
        pure_loss = float(np.mean(self.losses_))
        premium = self.calculate_solvency_premium(pure_loss, tvar)
        epd = self.calculate_epd_ratio(var)
        ci = self.bootstrap_var_tvar_ci()
        xi, _, sigma = self.gpd_params_
        return {"threshold": round(self.threshold_,4),
                "n_exceedances": len(self.exceedances_),
                "xi_GPD": round(xi,4), "sigma_GPD": round(sigma,4),
                f"VaR_{p*100:.1f}%": round(var,4),
                f"TVaR_{p*100:.1f}%": round(tvar,4),
                "SCR": round(var,4),
                "pure_expected_loss": round(pure_loss,4),
                "solvency_premium": round(premium,4),
                "EPD_ratio": epd,
                "VaR_95CI": ci["VaR_95CI"],
                "TVaR_95CI": ci["TVaR_95CI"]}
