"""PAT Multivariate Statistical Process Control (PCA T2/Q-SPE) and NIPALS PLS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, f as f_dist


@dataclass(frozen=True)
class PATScoreSample:
    """Per-observation MSPC diagnostic scores and fault isolation output."""

    sample_index: int
    t2: float
    spe: float
    t2_alarm: bool
    spe_alarm: bool
    dominant_variable: str
    variable_contributions: Dict[str, float]


@dataclass
class PATBatchAudit:
    """Batch-level multivariate PAT audit summary."""

    samples: List[PATScoreSample]
    t2_ucl: float
    spe_ucl: float
    t2_exceedance_pct: float
    spe_exceedance_pct: float
    fault_detected: bool
    top_fault_drivers: List[Tuple[str, float]]


class MultivariatePATMonitor:
    """SVD-based PCA MSPC monitor computing Hotelling's T^2, Q-residual (SPE), and contributions.

    Upper control limits:
    - Hotelling's ``T^2_UCL = (A (N^2 - 1)) / (N (N - A)) * F_{alpha, A, N - A}``
    - Jackson-Mudholkar ``Q_UCL`` (or scaled ``g * chi2_{h, 1 - alpha}`` Box approximation
      matching Nomikos & MacGregor batch MSPC).
    """

    def __init__(self, n_components: int = 3, alpha: float = 0.01) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if not (0.0 < alpha < 0.5):
            raise ValueError("alpha must be in (0, 0.5)")
        self.n_components = n_components
        self.alpha = alpha

        self.feature_names: List[str] = []
        self.mean_: Optional[NDArray[np.float64]] = None
        self.std_: Optional[NDArray[np.float64]] = None
        self.loadings_: Optional[NDArray[np.float64]] = None  # shape (P, A)
        self.eigenvalues_: Optional[NDArray[np.float64]] = None  # shape (A,)
        self.residual_eigvals_: Optional[NDArray[np.float64]] = None
        self.t2_ucl_: float = 0.0
        self.spe_ucl_: float = 0.0

    def fit(self, x_train: NDArray[np.float64], feature_names: Optional[Sequence[str]] = None) -> "MultivariatePATMonitor":
        """Calibrate PCA control model on in-control golden-batch observations ``(N, P)``."""
        x = np.asarray(x_train, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x_train must be 2D (n_samples, n_features)")
        n, p = x.shape
        a = min(self.n_components, n - 1, p)
        if a < 1 or n <= a + 1:
            raise ValueError("Insufficient training samples/features for PCA calibration")

        self.feature_names = (
            list(feature_names) if feature_names is not None else [f"var_{j}" for j in range(p)]
        )
        if len(self.feature_names) != p:
            raise ValueError("feature_names length must match x_train columns")

        self.mean_ = np.mean(x, axis=0)
        std = np.std(x, axis=0, ddof=1)
        self.std_ = np.where(std < 1e-9, 1.0, std)
        z = (x - self.mean_) / self.std_

        # SVD of standardized matrix Z = U S V^T
        _u, s, vt = np.linalg.svd(z, full_matrices=False)
        eigvals = (s ** 2) / max(1, n - 1)
        self.loadings_ = vt[:a, :].T  # (P, A)
        self.eigenvalues_ = np.maximum(eigvals[:a], 1e-10)
        self.residual_eigvals_ = eigvals[a:]

        # Hotelling T^2 UCL using F-distribution
        f_crit = float(f_dist.ppf(1.0 - self.alpha, a, n - a))
        self.t2_ucl_ = (a * (n ** 2 - 1.0)) / (n * (n - a)) * f_crit

        # SPE / Q-residual UCL via Jackson-Mudholkar normal approximation when residual variance > 0
        if len(self.residual_eigvals_) > 0 and float(np.sum(self.residual_eigvals_)) > 1e-12:
            theta1 = float(np.sum(self.residual_eigvals_))
            theta2 = float(np.sum(self.residual_eigvals_ ** 2))
            theta3 = float(np.sum(self.residual_eigvals_ ** 3))
            if theta2 > 1e-14:
                g = theta2 / theta1
                h = (theta1 ** 2) / theta2
                self.spe_ucl_ = float(g * chi2.ppf(1.0 - self.alpha, max(1.0, h)))
            else:
                self.spe_ucl_ = float(theta1 * 3.0)
            _ = theta3
        else:
            self.spe_ucl_ = 1e-3

        return self

    def score(self, x_obs: NDArray[np.float64]) -> PATBatchAudit:
        """Score observations against calibrated golden-batch control limits."""
        if self.mean_ is None or self.std_ is None or self.loadings_ is None or self.eigenvalues_ is None:
            raise RuntimeError("Monitor must be fit() before calling score()")

        x = np.asarray(x_obs, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        z = (x - self.mean_) / self.std_
        scores = z @ self.loadings_                 # (M, A)
        z_hat = scores @ self.loadings_.T           # (M, P)
        residuals = z - z_hat                       # (M, P)

        t2_vals = np.sum((scores ** 2) / self.eigenvalues_[None, :], axis=1)
        spe_vals = np.sum(residuals ** 2, axis=1)

        # Variable contribution to T^2 + SPE (combined standardized squared anomaly)
        t2_contrib_mat = ((scores / np.sqrt(self.eigenvalues_[None, :])) @ self.loadings_.T) ** 2
        spe_contrib_mat = residuals ** 2
        total_contrib = t2_contrib_mat + spe_contrib_mat

        samples: List[PATScoreSample] = []
        driver_accum: Dict[str, float] = {name: 0.0 for name in self.feature_names}

        for idx in range(x.shape[0]):
            t2_v = float(t2_vals[idx])
            spe_v = float(spe_vals[idx])
            t2_alarm = t2_v > self.t2_ucl_
            spe_alarm = spe_v > self.spe_ucl_
            c_vec = total_contrib[idx]
            c_dict = {name: float(c_vec[j]) for j, name in enumerate(self.feature_names)}
            dom_var = max(c_dict.items(), key=lambda kv: kv[1])[0]
            if t2_alarm or spe_alarm:
                for name, val in c_dict.items():
                    driver_accum[name] += val
            samples.append(
                PATScoreSample(
                    sample_index=idx,
                    t2=t2_v,
                    spe=spe_v,
                    t2_alarm=t2_alarm,
                    spe_alarm=spe_alarm,
                    dominant_variable=dom_var,
                    variable_contributions=c_dict,
                )
            )

        n_samples = max(1, len(samples))
        t2_exc = 100.0 * sum(1 for s in samples if s.t2_alarm) / n_samples
        spe_exc = 100.0 * sum(1 for s in samples if s.spe_alarm) / n_samples
        fault_detected = (t2_exc > 5.0) or (spe_exc > 5.0)
        sorted_drivers = sorted(driver_accum.items(), key=lambda kv: kv[1], reverse=True)

        return PATBatchAudit(
            samples=samples,
            t2_ucl=self.t2_ucl_,
            spe_ucl=self.spe_ucl_,
            t2_exceedance_pct=t2_exc,
            spe_exceedance_pct=spe_exc,
            fault_detected=fault_detected,
            top_fault_drivers=sorted_drivers[:3],
        )


class PLSQualityPredictor:
    """NIPALS Partial Least Squares (PLS1/PLS2) soft-sensor for CQA prediction."""

    def __init__(self, n_components: int = 3, max_iter: int = 200, tol: float = 1e-8) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.x_mean_: Optional[NDArray[np.float64]] = None
        self.x_std_: Optional[NDArray[np.float64]] = None
        self.y_mean_: Optional[NDArray[np.float64]] = None
        self.y_std_: Optional[NDArray[np.float64]] = None
        self.coef_: Optional[NDArray[np.float64]] = None

    def fit(self, x_train: NDArray[np.float64], y_train: NDArray[np.float64]) -> "PLSQualityPredictor":
        """Fit NIPALS PLS model mapping process trajectory features X to CQAs Y."""
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        n, p = x.shape
        m = y.shape[1]
        a_comp = min(self.n_components, n - 1, p)

        self.x_mean_ = np.mean(x, axis=0)
        self.x_std_ = np.where(np.std(x, axis=0, ddof=1) < 1e-9, 1.0, np.std(x, axis=0, ddof=1))
        self.y_mean_ = np.mean(y, axis=0)
        self.y_std_ = np.where(np.std(y, axis=0, ddof=1) < 1e-9, 1.0, np.std(y, axis=0, ddof=1))

        xk = (x - self.x_mean_) / self.x_std_
        yk = (y - self.y_mean_) / self.y_std_

        w_mat = np.zeros((p, a_comp), dtype=np.float64)
        p_mat = np.zeros((p, a_comp), dtype=np.float64)
        q_mat = np.zeros((m, a_comp), dtype=np.float64)

        for a in range(a_comp):
            # Initialize u as column of yk with largest variance
            col_idx = int(np.argmax(np.sum(yk ** 2, axis=0)))
            u = yk[:, col_idx].copy()
            if np.linalg.norm(u) < 1e-12:
                break
            w = np.zeros(p, dtype=np.float64)
            t = np.zeros(n, dtype=np.float64)
            q = np.zeros(m, dtype=np.float64)

            for _ in range(self.max_iter):
                w = xk.T @ u
                w_norm = float(np.linalg.norm(w))
                if w_norm < 1e-12:
                    break
                w /= w_norm
                t_new = xk @ w
                t_dot = float(t_new @ t_new)
                if t_dot < 1e-12:
                    t = t_new
                    break
                q = (yk.T @ t_new) / t_dot
                q_dot = float(q @ q)
                u_new = (yk @ q) / max(q_dot, 1e-12)
                diff = float(np.linalg.norm(t_new - t))
                t = t_new
                u = u_new
                if diff < self.tol:
                    break

            t_dot = max(float(t @ t), 1e-12)
            p_vec = (xk.T @ t) / t_dot
            xk = xk - np.outer(t, p_vec)
            yk = yk - np.outer(t, q)

            w_mat[:, a] = w
            p_mat[:, a] = p_vec
            q_mat[:, a] = q

        # Regression coefficients in standardized space: B = W (P^T W)^-1 Q^T
        ptw = p_mat.T @ w_mat
        self.coef_ = w_mat @ np.linalg.pinv(ptw) @ q_mat.T
        return self

    def predict(self, x_obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict Critical Quality Attributes from process observations."""
        if self.coef_ is None or self.x_mean_ is None or self.x_std_ is None or self.y_mean_ is None or self.y_std_ is None:
            raise RuntimeError("PLSQualityPredictor must be fit() before predict()")
        x = np.asarray(x_obs, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        z = (x - self.x_mean_) / self.x_std_
        y_scaled = z @ self.coef_
        return y_scaled * self.y_std_ + self.y_mean_
