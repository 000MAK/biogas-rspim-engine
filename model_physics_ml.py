"""src/model_physics_ml.py - OU Filter (Kalman-RTS), HysteresisPenalizer, ModelComparisonEngine (13 models x k=2..8)."""
from __future__ import annotations
import time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats, signal
from sklearn import metrics as sk_metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from typing import Dict, List, Optional, Tuple
try:
    from hmmlearn import hmm as hmmlib
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
try:
    from tslearn.clustering import TimeSeriesKMeans
    HAS_TSLEARN = True
except ImportError:
    HAS_TSLEARN = False
from config import CONFIG, BiogasConfig


# ===========================================================================
# ORNSTEIN-UHLENBECK FILTER (Kalman-RTS)
# ===========================================================================
class OrnsteinUhlenbeckFilter:
    """Exact analytical OU + Kalman Forward Filter + Rauch-Tung-Striebel Backward Smoother."""

    def __init__(self, cfg: BiogasConfig = CONFIG):
        self.cfg = cfg
        self.theta = cfg.OU_THETA
        self.mu = cfg.OU_MU
        self.sigma = cfg.OU_SIGMA
        self.dt = cfg.OU_DT
        self.phi = float(np.exp(-self.theta * self.dt))
        self.proc_var = float(self.sigma**2 / (2*self.theta) * (1 - np.exp(-2*self.theta*self.dt)))
        self.smoothed_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.filter_var_: Optional[np.ndarray] = None

    def fit_transform(self, series: np.ndarray, obs_var: Optional[float] = None) -> np.ndarray:
        """Kalman forward filter + RTS backward smoother. Returns smoothed signal."""
        n = len(series)
        obs_var = obs_var if obs_var is not None else float(np.var(series) * 0.1)
        x_filt = np.empty(n); P_filt = np.empty(n)
        x_pred = np.empty(n); P_pred = np.empty(n)
        x_filt[0] = series[0]; P_filt[0] = self.proc_var
        for t in range(1, n):
            xp = self.phi * x_filt[t-1] + (1 - self.phi) * self.mu
            Pp = self.phi**2 * P_filt[t-1] + self.proc_var
            x_pred[t] = xp; P_pred[t] = Pp
            K = Pp / (Pp + obs_var)
            x_filt[t] = xp + K * (series[t] - xp)
            P_filt[t] = (1 - K) * Pp
        x_pred[0] = x_filt[0]; P_pred[0] = P_filt[0]
        x_smooth = x_filt.copy(); P_smooth = P_filt.copy()
        for t in range(n-2, -1, -1):
            G = P_filt[t] * self.phi / max(P_pred[t+1], 1e-12)
            x_smooth[t] = x_filt[t] + G * (x_smooth[t+1] - x_pred[t+1])
            P_smooth[t] = P_filt[t] + G**2 * (P_smooth[t+1] - P_pred[t+1])
        self.smoothed_ = x_smooth
        self.residuals_ = series - x_smooth
        self.filter_var_ = P_smooth
        return x_smooth

    def ljung_box_residuals(self, lags: int = 20) -> dict:
        if self.residuals_ is None: raise ValueError("Call fit_transform first.")
        r = self.residuals_; n = len(r); mean_r = r.mean()
        r0 = np.sum((r - mean_r)**2)
        acfs = [np.sum((r[h:]-mean_r)*(r[:-h]-mean_r))/r0 for h in range(1, lags+1)]
        LB = n*(n+2)*sum(ac**2/(n-h) for h, ac in enumerate(acfs, 1))
        pval = float(stats.chi2.sf(LB, df=lags))
        return {"lb_stat": round(LB, 4), "lb_pvalue": round(pval, 4), "white_noise": pval > 0.05}

    def welch_periodogram(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.smoothed_ is None: raise ValueError("Call fit_transform first.")
        fs = 1.0 / self.dt
        f_raw, Pxx_raw = signal.welch(series, fs=fs, nperseg=min(256, len(series)//2))
        _, Pxx_sm = signal.welch(self.smoothed_, fs=fs, nperseg=min(256, len(self.smoothed_)//2))
        return f_raw, Pxx_raw, Pxx_sm

    def monte_carlo_ergodic(self, n_paths: int = 100, n_steps: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(self.cfg.RANDOM_SEED)
        final_vals = []
        for _ in range(n_paths):
            x = self.mu
            for _ in range(n_steps):
                x = x*self.phi + (1-self.phi)*self.mu + np.sqrt(self.proc_var)*rng.standard_normal()
            final_vals.append(x)
        final = np.array(final_vals)
        theoretical_var = self.sigma**2 / (2*self.theta)
        return pd.DataFrame({"stat": ["MC_mean","MC_std","Theoretical_mean","Theoretical_std","MC_skew","MC_kurt"],
                             "value": [round(final.mean(),4), round(final.std(),4),
                                       round(self.mu,4), round(np.sqrt(theoretical_var),4),
                                       round(float(stats.skew(final)),4), round(float(stats.kurtosis(final)),4)]})


# ===========================================================================
# BIOLOGICAL HYSTERESIS PENALIZER
# ===========================================================================
class HysteresisPenalizer:
    """Generalized hysteresis kernel + Monod kinetics + correction audit."""

    def __init__(self, cfg: BiogasConfig = CONFIG):
        self.cfg = cfg
        self.correction_log_: List[dict] = []
        self.transmat_before_: Optional[np.ndarray] = None
        self.transmat_after_: Optional[np.ndarray] = None

    def monod_inhibition(self, ph: np.ndarray) -> np.ndarray:
        """I(pH) = exp(-alpha*(pH_crit - pH)^2) for pH < pH_crit, else 1.0."""
        alpha = self.cfg.MONOD_ALPHA; ph_crit = self.cfg.HYSTERESIS_PH_CRITICAL
        I = np.ones_like(ph, dtype=float); below = ph < ph_crit
        I[below] = np.exp(-alpha * (ph_crit - ph[below])**2)
        return I

    def monod_growth_rate(self, substrate: np.ndarray, ph: np.ndarray,
                          mu_max: float = 0.9, Ks: float = 0.5) -> np.ndarray:
        return mu_max * substrate / (Ks + substrate + 1e-12) * self.monod_inhibition(ph)

    def thermodynamic_check(self, ph: np.ndarray) -> pd.DataFrame:
        deltaG0 = -31.0; R = 8.314e-3; T_K = 310.0
        H_conc = 10.0**(-ph)
        deltaG = deltaG0 + R * T_K * np.log(H_conc + 1e-20)
        feasible = deltaG < 0
        n = min(len(ph), 20)
        return pd.DataFrame({"pH": np.round(ph[:n],3), "deltaG_kJ_mol": np.round(deltaG[:n],3),
                             "methanogenesis_feasible": feasible[:n]})

    def _transition_matrix(self, states: np.ndarray, k: int) -> np.ndarray:
        T = np.zeros((k, k))
        for t in range(1, len(states)):
            s0, s1 = int(states[t-1]), int(states[t])
            if 0 <= s0 < k and 0 <= s1 < k: T[s0, s1] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        return T / np.maximum(row_sums, 1)

    def apply(self, states: np.ndarray, ph: np.ndarray, k: int) -> np.ndarray:
        out = states.copy().astype(int); n = min(len(out), len(ph))
        self.transmat_before_ = self._transition_matrix(out, k)
        self.correction_log_ = []
        worst_state = k - 1
        for t in range(1, n):
            if out[t-1] == worst_state and out[t] == 0 and ph[t] < self.cfg.HYSTERESIS_PH_THRESHOLD:
                corrected = max(k - 2, 1)
                self.correction_log_.append({"t": t, "pH": round(float(ph[t]),3),
                                              "before": int(out[t]), "after": corrected})
                out[t] = corrected
        self.transmat_after_ = self._transition_matrix(out, k)
        return out

    def correction_rate(self) -> float:
        return round(len(self.correction_log_) / max(len(self.correction_log_)+1, 1), 4)

    def audit_trail_df(self) -> pd.DataFrame:
        if not self.correction_log_: return pd.DataFrame({"message":["No hysteresis corrections applied"]})
        return pd.DataFrame(self.correction_log_)


# ===========================================================================
# INFORMATION CRITERIA HELPERS
# ===========================================================================
def _aic(ll, k): return 2.0*k - 2.0*ll
def _bic(ll, k, n): return k*float(np.log(max(n,2))) - 2.0*ll
def _aicc(ll, k, n):
    d = n - k - 1
    return _aic(ll,k) + 2.0*k*(k+1)/d if d > 0 else float("inf")
def _caic(ll, k, n): return -2.0*ll + k*(float(np.log(max(n,2)))+1.0)
def _hqic(ll, k, n): return -2.0*ll + 2.0*k*float(np.log(max(float(np.log(max(n,3))),0.001)))
def _icl(ll, gamma, k_p, n):
    g = np.clip(gamma, 1e-300, 1.0)
    ent = -float(np.sum(g * np.log(g)))
    return _bic(ll, k_p, n) - 2.0*ent
def _state_entropy(labels, n_states):
    c = np.bincount(labels.astype(int), minlength=n_states).astype(float)
    p = c/max(c.sum(),1); p = p[p>0]
    return float(-np.sum(p*np.log(p)))
def _cluster_metrics(X, labels):
    u = np.unique(labels[labels>=0])
    if len(u) < 2: return {"silhouette":-1.0,"calinski_harabasz":0.0,"davies_bouldin":99.0}
    mask = labels >= 0; Xm, Lm = X[mask], labels[mask]
    if len(np.unique(Lm)) < 2: return {"silhouette":-1.0,"calinski_harabasz":0.0,"davies_bouldin":99.0}
    try: sil=float(sk_metrics.silhouette_score(Xm,Lm,sample_size=min(len(Xm),2000)))
    except: sil=-1.0
    try: ch=float(sk_metrics.calinski_harabasz_score(Xm,Lm))
    except: ch=0.0
    try: db=float(sk_metrics.davies_bouldin_score(Xm,Lm))
    except: db=99.0
    return {"silhouette":sil,"calinski_harabasz":ch,"davies_bouldin":db}
def _null_row(name, k, reason):
    return {"model":name,"k":k,"log_ll":float("nan"),"AIC":float("nan"),"BIC":float("nan"),
            "AICc":float("nan"),"CAIC":float("nan"),"HQIC":float("nan"),"ICL":float("nan"),
            "silhouette":float("nan"),"calinski_harabasz":float("nan"),"davies_bouldin":float("nan"),
            "entropy":float("nan"),"pred_ll":float("nan"),"train_time_s":0.0,"conv_rate":0.0,
            "n_params":0,"status":f"FAILED:{reason}","labels":np.array([]),"mean_state_duration":float("nan")}
def _state_durations(labels):
    if len(labels)==0: return 0.0
    durations=[]; cur=labels[0]; cnt=1
    for s in labels[1:]:
        if s==cur: cnt+=1
        else: durations.append(cnt); cur=s; cnt=1
    durations.append(cnt)
    return float(np.mean(durations))


# ===========================================================================
# MODEL COMPARISON ENGINE
# ===========================================================================
class ModelComparisonEngine:
    """Fits 13 model families for k in cfg.K_CANDIDATES. Computes 12 criteria. MCDA rank."""

    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, cfg: BiogasConfig = CONFIG):
        self.X_train = X_train; self.X_test = X_test; self.cfg = cfg
        self.results_df_: Optional[pd.DataFrame] = None
        self.mcda_df_: Optional[pd.DataFrame] = None

    def _fit_hmm(self, k: int, cov_type: str) -> dict:
        if not HAS_HMMLEARN: return _null_row(f"HMM-{cov_type}", k, "hmmlearn missing")
        X,n,d=self.X_train,len(self.X_train),self.X_train.shape[1]
        best_ll=-np.inf; best_model=None; converged=0; n_init=max(self.cfg.GMM_N_INIT//2,2)
        t0=time.perf_counter()
        for off in range(n_init):
            try:
                m=hmmlib.GaussianHMM(n_components=k,covariance_type=cov_type,
                    n_iter=self.cfg.HMM_N_ITER,tol=self.cfg.HMM_TOL,
                    random_state=self.cfg.RANDOM_SEED+off)
                m.fit(X); ll=m.score(X)
                if np.isfinite(ll) and ll>best_ll: best_ll=ll; best_model=m; converged+=1
            except: continue
        if best_model is None: return _null_row(f"HMM-{cov_type}",k,"all inits failed")
        if cov_type=="full": k_p=k*(k-1)+(k-1)+k*d+k*d*(d+1)//2
        elif cov_type=="diag": k_p=k*(k-1)+(k-1)+k*d*2
        else: k_p=k*(k-1)+(k-1)+k*d+k
        try: gamma=best_model.predict_proba(X)
        except: gamma=np.ones((n,k))/k
        try: _,labels=best_model.decode(X,algorithm="viterbi")
        except: labels=np.zeros(n,dtype=int)
        cm=_cluster_metrics(X,labels)
        try: pred_ll=float(best_model.score(self.X_test))
        except: pred_ll=float("nan")
        return {"model":f"HMM-{cov_type}","k":k,"log_ll":round(best_ll,4),
                "AIC":round(_aic(best_ll,k_p),2),"BIC":round(_bic(best_ll,k_p,n),2),
                "AICc":round(_aicc(best_ll,k_p,n),2),"CAIC":round(_caic(best_ll,k_p,n),2),
                "HQIC":round(_hqic(best_ll,k_p,n),2),"ICL":round(_icl(best_ll,gamma,k_p,n),2),
                "silhouette":round(cm["silhouette"],4),"calinski_harabasz":round(cm["calinski_harabasz"],2),
                "davies_bouldin":round(cm["davies_bouldin"],4),"entropy":round(_state_entropy(labels,k),4),
                "pred_ll":round(pred_ll,4) if np.isfinite(pred_ll) else float("nan"),
                "train_time_s":round(time.perf_counter()-t0,3),"conv_rate":round(converged/n_init,3),
                "n_params":k_p,"status":"OK","labels":labels,
                "mean_state_duration":round(_state_durations(labels),2)}

    def _fit_gmm(self, k: int, cov_type: str) -> dict:
        X,n,d=self.X_train,len(self.X_train),self.X_train.shape[1]; t0=time.perf_counter()
        try:
            m=GaussianMixture(n_components=k,covariance_type=cov_type,n_init=self.cfg.GMM_N_INIT,
                max_iter=self.cfg.GMM_MAX_ITER,random_state=self.cfg.RANDOM_SEED)
            m.fit(X); ll=float(m.score(X)*n)
            if cov_type=="full": k_p=(k-1)+k*d+k*d*(d+1)//2
            elif cov_type=="diag": k_p=(k-1)+k*d*2
            elif cov_type=="tied": k_p=(k-1)+k*d+d*(d+1)//2
            else: k_p=(k-1)+k*d+k
            gamma=m.predict_proba(X); labels=m.predict(X); cm=_cluster_metrics(X,labels)
            try: pred_ll=float(m.score(self.X_test)*len(self.X_test))
            except: pred_ll=float("nan")
            return {"model":f"GMM-{cov_type}","k":k,"log_ll":round(ll,4),
                    "AIC":round(_aic(ll,k_p),2),"BIC":round(_bic(ll,k_p,n),2),
                    "AICc":round(_aicc(ll,k_p,n),2),"CAIC":round(_caic(ll,k_p,n),2),
                    "HQIC":round(_hqic(ll,k_p,n),2),"ICL":round(_icl(ll,gamma,k_p,n),2),
                    "silhouette":round(cm["silhouette"],4),"calinski_harabasz":round(cm["calinski_harabasz"],2),
                    "davies_bouldin":round(cm["davies_bouldin"],4),"entropy":round(_state_entropy(labels,k),4),
                    "pred_ll":round(pred_ll,4) if np.isfinite(pred_ll) else float("nan"),
                    "train_time_s":round(time.perf_counter()-t0,3),"conv_rate":1.0,
                    "n_params":k_p,"status":"OK","labels":labels,
                    "mean_state_duration":round(_state_durations(labels),2)}
        except Exception as e: return _null_row(f"GMM-{cov_type}",k,str(e))

    def _fit_bayesian_gmm(self, k: int) -> dict:
        X,n,d=self.X_train,len(self.X_train),self.X_train.shape[1]; t0=time.perf_counter()
        try:
            m=BayesianGaussianMixture(n_components=k,covariance_type="full",
                max_iter=self.cfg.BGM_MAX_ITER,
                weight_concentration_prior=self.cfg.BGM_WEIGHT_CONCENTRATION_PRIOR,
                random_state=self.cfg.RANDOM_SEED)
            m.fit(X); n_eff=int(np.sum(m.weights_>0.01))
            gamma=m.predict_proba(X); labels=m.predict(X)
            ll=float(np.sum(m.score_samples(X))); k_p=max(n_eff,1)*(d+d*(d+1)//2+1)
            cm=_cluster_metrics(X,labels)
            try: pred_ll=float(np.sum(m.score_samples(self.X_test)))
            except: pred_ll=float("nan")
            return {"model":"BayesianGMM-DP","k":k,"log_ll":round(ll,4),
                    "AIC":round(_aic(ll,k_p),2),"BIC":round(_bic(ll,k_p,n),2),
                    "AICc":round(_aicc(ll,k_p,n),2),"CAIC":round(_caic(ll,k_p,n),2),
                    "HQIC":round(_hqic(ll,k_p,n),2),"ICL":round(_icl(ll,gamma,k_p,n),2),
                    "silhouette":round(cm["silhouette"],4),"calinski_harabasz":round(cm["calinski_harabasz"],2),
                    "davies_bouldin":round(cm["davies_bouldin"],4),"entropy":round(_state_entropy(labels,k),4),
                    "pred_ll":round(pred_ll,4) if np.isfinite(pred_ll) else float("nan"),
                    "train_time_s":round(time.perf_counter()-t0,3),"conv_rate":1.0,
                    "n_params":k_p,"status":f"OK(n_eff={n_eff})","labels":labels,
                    "mean_state_duration":round(_state_durations(labels),2)}
        except Exception as e: return _null_row("BayesianGMM-DP",k,str(e))

    def _fit_kmeans(self, k: int) -> dict:
        X,n,d=self.X_train,len(self.X_train),self.X_train.shape[1]; t0=time.perf_counter()
        try:
            m=KMeans(n_clusters=k,n_init=5,max_iter=300,random_state=self.cfg.RANDOM_SEED)
            labels=m.fit_predict(X); ll=-float(m.inertia_)/2.0; k_p=k*d+k
            cm=_cluster_metrics(X,labels)
            try:
                tl=m.predict(self.X_test)
                pred_ll=float(-np.sum((self.X_test-m.cluster_centers_[tl])**2)/2.0)
            except: pred_ll=float("nan")
            return {"model":"KMeans","k":k,"log_ll":round(ll,4),
                    "AIC":round(_aic(ll,k_p),2),"BIC":round(_bic(ll,k_p,n),2),
                    "AICc":round(_aicc(ll,k_p,n),2),"CAIC":round(_caic(ll,k_p,n),2),
                    "HQIC":round(_hqic(ll,k_p,n),2),"ICL":round(_bic(ll,k_p,n),2),
                    "silhouette":round(cm["silhouette"],4),"calinski_harabasz":round(cm["calinski_harabasz"],2),
                    "davies_bouldin":round(cm["davies_bouldin"],4),"entropy":round(_state_entropy(labels,k),4),
                    "pred_ll":round(pred_ll,4) if np.isfinite(pred_ll) else float("nan"),
                    "train_time_s":round(time.perf_counter()-t0,3),"conv_rate":1.0,
                    "n_params":k_p,"status":"OK","labels":labels,
                    "mean_state_duration":round(_state_durations(labels),2)}
        except Exception as e: return _null_row("KMeans",k,str(e))

    def _fit_agglomerative(self, k: int) -> dict:
        X,n,d=self.X_train,len(self.X_train),self.X_train.shape[1]; t0=time.perf_counter()
        try:
            labels=AgglomerativeClustering(n_clusters=k,linkage="ward").fit_predict(X)
            k_p=k*d; cm=_cluster_metrics(X,labels)
            ll=sum(float(np.sum(stats.multivariate_normal.logpdf(
                X[labels==c],mean=X[labels==c].mean(0),cov=np.cov(X[labels==c].T)+1e-6*np.eye(d))))
                for c in range(k) if (labels==c).sum()>1)
            return {"model":"Agglomerative-Ward","k":k,"log_ll":round(ll,4),
                    "AIC":round(_aic(ll,k_p),2),"BIC":round(_bic(ll,k_p,n),2),
                    "AICc":round(_aicc(ll,k_p,n),2),"CAIC":round(_caic(ll,k_p,n),2),
                    "HQIC":round(_hqic(ll,k_p,n),2),"ICL":round(_bic(ll,k_p,n),2),
                    "silhouette":round(cm["silhouette"],4),"calinski_harabasz":round(cm["calinski_harabasz"],2),
                    "davies_bouldin":round(cm["davies_bouldin"],4),"entropy":round(_state_entropy(labels,k),4),
                    "pred_ll":float("nan"),"train_time_s":round(time.perf_counter()-t0,3),
                    "conv_rate":1.0,"n_params":k_p,"status":"OK","labels":labels,
                    "mean_state_duration":round(_state_durations(labels),2)}
        except Exception as e: return _null_row("Agglomerative-Ward",k,str(e))

    def _fit_dbscan(self) -> dict:
        X=self.X_train; t0=time.perf_counter()
        try:
            Xs=(X-X.mean(0))/(X.std(0)+1e-8)
            labels=DBSCAN(eps=self.cfg.DBSCAN_EPS,min_samples=self.cfg.DBSCAN_MIN_SAMPLES).fit_predict(Xs)
            n_cl=len(set(labels))-(1 if -1 in labels else 0)
            n_noise=int((labels==-1).sum()); cm=_cluster_metrics(Xs,labels)
            return {"model":"DBSCAN","k":n_cl,"log_ll":float("nan"),
                    "AIC":float("nan"),"BIC":float("nan"),"AICc":float("nan"),
                    "CAIC":float("nan"),"HQIC":float("nan"),"ICL":float("nan"),
                    "silhouette":round(cm["silhouette"],4),"calinski_harabasz":round(cm["calinski_harabasz"],2),
                    "davies_bouldin":round(cm["davies_bouldin"],4),
                    "entropy":round(_state_entropy(np.clip(labels,0,99),max(n_cl,1)),4),
                    "pred_ll":float("nan"),"train_time_s":round(time.perf_counter()-t0,3),
                    "conv_rate":1.0,"n_params":2,"status":f"OK(cl={n_cl},noise={n_noise})",
                    "labels":labels,"mean_state_duration":float("nan")}
        except Exception as e: return _null_row("DBSCAN",-1,str(e))

    def _fit_spectral(self, k: int) -> dict:
        X,n,d=self.X_train,len(self.X_train),self.X_train.shape[1]
        n_sub=min(n,1500); t0=time.perf_counter()
        try:
            rng=np.random.default_rng(self.cfg.RANDOM_SEED)
            idx=rng.choice(n,n_sub,replace=False) if n>n_sub else np.arange(n)
            Xs=X[idx]
            ls=SpectralClustering(n_clusters=k,affinity=self.cfg.SPECTRAL_AFFINITY,
                random_state=self.cfg.RANDOM_SEED,n_init=3).fit_predict(Xs)
            centroids=np.array([Xs[ls==c].mean(0) if (ls==c).any() else X.mean(0) for c in range(k)])
            labels=np.argmin(np.linalg.norm(X[:,None,:]-centroids[None,:,:],axis=2),axis=1)
            cm=_cluster_metrics(X,labels); k_p=k*d
            ll=sum(float(np.sum(stats.norm.logpdf(X[labels==c]-centroids[c])))
                for c in range(k) if (labels==c).any())
            return {"model":"Spectral-RBF","k":k,"log_ll":round(ll,4),
                    "AIC":round(_aic(ll,k_p),2),"BIC":round(_bic(ll,k_p,n),2),
                    "AICc":round(_aicc(ll,k_p,n),2),"CAIC":round(_caic(ll,k_p,n),2),
                    "HQIC":round(_hqic(ll,k_p,n),2),"ICL":round(_bic(ll,k_p,n),2),
                    "silhouette":round(cm["silhouette"],4),"calinski_harabasz":round(cm["calinski_harabasz"],2),
                    "davies_bouldin":round(cm["davies_bouldin"],4),"entropy":round(_state_entropy(labels,k),4),
                    "pred_ll":float("nan"),"train_time_s":round(time.perf_counter()-t0,3),
                    "conv_rate":1.0,"n_params":k_p,"status":"OK","labels":labels,
                    "mean_state_duration":round(_state_durations(labels),2)}
        except Exception as e: return _null_row("Spectral-RBF",k,str(e))

    def _fit_student_hmm(self, k: int, nu: float = 4.0) -> dict:
        if not HAS_HMMLEARN: return _null_row("HMM-StudentT",k,"hmmlearn missing")
        X,n,d=self.X_train,len(self.X_train),self.X_train.shape[1]; t0=time.perf_counter()
        try:
            m=hmmlib.GaussianHMM(n_components=k,covariance_type="full",
                n_iter=min(self.cfg.HMM_N_ITER,60),random_state=self.cfg.RANDOM_SEED)
            m.fit(X); _,labels=m.decode(X,algorithm="viterbi")
            means=m.means_.copy(); covars=m.covars_.copy()
            for _ in range(5):
                for i in range(k):
                    mask=(labels==i)
                    if mask.sum()<2: continue
                    Xc=X[mask]; mu=means[i]; Sig=covars[i]
                    try:
                        Si=np.linalg.inv(Sig+1e-5*np.eye(d))
                        diff=Xc-mu; delta2=np.einsum("ni,ij,nj->n",diff,Si,diff)
                        w=(nu+d)/(nu+delta2); W=w.sum()
                        means[i]=(w@Xc)/W; diff2=Xc-means[i]
                        covars[i]=(w[:,None]*diff2).T@diff2/W+1e-5*np.eye(d)
                    except: pass
                m.means_=means; m.covars_=covars
                try: _,labels=m.decode(X,algorithm="viterbi")
                except: break
            ll=float(m.score(X)); k_p=k*(k-1)+(k-1)+k*d+k*d*(d+1)//2
            try: gamma=m.predict_proba(X)
            except: gamma=np.ones((n,k))/k
            cm=_cluster_metrics(X,labels)
            try: pred_ll=float(m.score(self.X_test))
            except: pred_ll=float("nan")
            return {"model":"HMM-StudentT","k":k,"log_ll":round(ll,4),
                    "AIC":round(_aic(ll,k_p),2),"BIC":round(_bic(ll,k_p,n),2),
                    "AICc":round(_aicc(ll,k_p,n),2),"CAIC":round(_caic(ll,k_p,n),2),
                    "HQIC":round(_hqic(ll,k_p,n),2),"ICL":round(_icl(ll,gamma,k_p,n),2),
                    "silhouette":round(cm["silhouette"],4),"calinski_harabasz":round(cm["calinski_harabasz"],2),
                    "davies_bouldin":round(cm["davies_bouldin"],4),"entropy":round(_state_entropy(labels,k),4),
                    "pred_ll":round(pred_ll,4) if np.isfinite(pred_ll) else float("nan"),
                    "train_time_s":round(time.perf_counter()-t0,3),"conv_rate":1.0,
                    "n_params":k_p,"status":"OK(IRLS)","labels":labels,
                    "mean_state_duration":round(_state_durations(labels),2)}
        except Exception as e: return _null_row("HMM-StudentT",k,str(e))

    def _fit_tskmeans(self, k: int, window_len: int = 20) -> dict:
        X,n,d=self.X_train,len(self.X_train),self.X_train.shape[1]; t0=time.perf_counter()
        try:
            if HAS_TSLEARN:
                n_w=n//window_len
                if n_w<k: return _null_row("TSKMeans-DTW",k,f"n_windows={n_w}<k")
                Xw=X[:n_w*window_len].reshape(n_w,window_len,d)
                m=TimeSeriesKMeans(n_clusters=k,metric="dtw",max_iter=self.cfg.TSKM_MAX_ITER,
                    n_init=self.cfg.TSKM_N_INIT,random_state=self.cfg.RANDOM_SEED,verbose=False)
                wl=m.fit_predict(Xw); labels=np.repeat(wl,window_len)[:n]; method="DTW"
            else:
                labels=KMeans(n_clusters=k,n_init=3,random_state=self.cfg.RANDOM_SEED).fit_predict(X)
                method="Euclidean(fallback)"
            cm=_cluster_metrics(X,labels); k_p=k*d
            ll=sum(float(np.sum(stats.norm.logpdf(X[labels==c]-X[labels==c].mean(0))))
                for c in range(k) if (labels==c).any())
            return {"model":f"TSKMeans-{method}","k":k,"log_ll":round(ll,4),
                    "AIC":round(_aic(ll,k_p),2),"BIC":round(_bic(ll,k_p,n),2),
                    "AICc":round(_aicc(ll,k_p,n),2),"CAIC":round(_caic(ll,k_p,n),2),
                    "HQIC":round(_hqic(ll,k_p,n),2),"ICL":round(_bic(ll,k_p,n),2),
                    "silhouette":round(cm["silhouette"],4),"calinski_harabasz":round(cm["calinski_harabasz"],2),
                    "davies_bouldin":round(cm["davies_bouldin"],4),"entropy":round(_state_entropy(labels,k),4),
                    "pred_ll":float("nan"),"train_time_s":round(time.perf_counter()-t0,3),
                    "conv_rate":1.0,"n_params":k_p,"status":f"OK({method})","labels":labels,
                    "mean_state_duration":round(_state_durations(labels),2)}
        except Exception as e: return _null_row("TSKMeans-DTW",k,str(e))

    def run_full_comparison(self, progress_callback=None) -> pd.DataFrame:
        rows = []
        model_fns = [
            ("HMM-full",    lambda k: self._fit_hmm(k,"full")),
            ("HMM-diag",    lambda k: self._fit_hmm(k,"diag")),
            ("HMM-spherical",lambda k: self._fit_hmm(k,"spherical")),
            ("GMM-full",    lambda k: self._fit_gmm(k,"full")),
            ("GMM-diag",    lambda k: self._fit_gmm(k,"diag")),
            ("GMM-spherical",lambda k: self._fit_gmm(k,"spherical")),
            ("BayesianGMM", lambda k: self._fit_bayesian_gmm(k)),
            ("KMeans",      lambda k: self._fit_kmeans(k)),
            ("Agglomerative",lambda k: self._fit_agglomerative(k)),
            ("Spectral",    lambda k: self._fit_spectral(k)),
            ("HMM-StudentT",lambda k: self._fit_student_hmm(k)),
            ("TSKMeans",    lambda k: self._fit_tskmeans(k)),
        ]
        tasks = [(name,k,fn) for name,fn in model_fns for k in self.cfg.K_CANDIDATES]
        total = len(tasks) + 1
        for step,(name,k,fn) in enumerate(tasks):
            if progress_callback: progress_callback(step,total,f"{name} k={k}")
            r = fn(k)
            rows.append({kk:v for kk,v in r.items() if kk!="labels"})
        if progress_callback: progress_callback(len(tasks),total,"DBSCAN")
        r = self._fit_dbscan()
        rows.append({kk:v for kk,v in r.items() if kk!="labels"})
        self.results_df_ = pd.DataFrame(rows)
        return self.results_df_

    def get_mcda_ranking(self) -> pd.DataFrame:
        if self.results_df_ is None: raise ValueError("Run run_full_comparison() first.")
        df = self.results_df_.copy()
        def nl(col):
            s=pd.to_numeric(col,errors="coerce"); lo,hi=s.min(),s.max()
            return pd.Series(0.5,index=col.index) if hi==lo else 1.0-(s-lo)/(hi-lo)
        def nh(col):
            s=pd.to_numeric(col,errors="coerce"); lo,hi=s.min(),s.max()
            return pd.Series(0.5,index=col.index) if hi==lo else (s-lo)/(hi-lo)
        c=self.cfg
        df["MCDA_Score"]=(c.MCDA_WEIGHT_BIC*nl(df["BIC"]).fillna(0)+
                          c.MCDA_WEIGHT_ICL*nl(df["ICL"]).fillna(0)+
                          c.MCDA_WEIGHT_SILHOUETTE*nh(df["silhouette"]).fillna(0)+
                          c.MCDA_WEIGHT_PREDICTIVE_LL*nh(df["pred_ll"]).fillna(0)+
                          c.MCDA_WEIGHT_CONVERGENCE*nh(df["conv_rate"]).fillna(0)).round(4)
        df["MCDA_Rank"]=df["MCDA_Score"].rank(ascending=False,method="min").astype(int)
        self.mcda_df_=df.sort_values("MCDA_Rank")
        return self.mcda_df_

    def get_optimal_model_k(self) -> Tuple[str, int, str]:
        if self.mcda_df_ is None: self.get_mcda_ranking()
        top = self.mcda_df_.iloc[0]
        return (str(top["model"]), int(top["k"]),
                f"MCDA={top['MCDA_Score']:.4f} BIC={top['BIC']:.2f} Sil={top['silhouette']:.4f}")

    def get_mcda_sensitivity(self) -> pd.DataFrame:
        if self.results_df_ is None: return pd.DataFrame()
        df=self.results_df_.copy()
        def nl(col):
            s=pd.to_numeric(col,errors="coerce"); lo,hi=s.min(),s.max()
            return pd.Series(0.5,index=col.index) if hi==lo else 1.0-(s-lo)/(hi-lo)
        def nh(col):
            s=pd.to_numeric(col,errors="coerce"); lo,hi=s.min(),s.max()
            return pd.Series(0.5,index=col.index) if hi==lo else (s-lo)/(hi-lo)
        rows=[]
        for label,wb,wi,ws,wp,wc in [
            ("Base",0.30,0.25,0.20,0.15,0.10),
            ("BIC+10%",0.40,0.225,0.175,0.125,0.075),
            ("BIC-10%",0.20,0.275,0.225,0.175,0.125),
            ("Sil+10%",0.275,0.225,0.30,0.125,0.075),
            ("Sil-10%",0.325,0.25,0.10,0.175,0.15)]:
            s=(wb*nl(df["BIC"]).fillna(0)+wi*nl(df["ICL"]).fillna(0)+
               ws*nh(df["silhouette"]).fillna(0)+wp*nh(df["pred_ll"]).fillna(0)+
               wc*nh(df["conv_rate"]).fillna(0))
            best=df.loc[s.idxmax()]
            rows.append({"Config":label,"Top_model":best["model"],"Top_k":int(best["k"]),"Score":round(s.max(),4)})
        return pd.DataFrame(rows)
