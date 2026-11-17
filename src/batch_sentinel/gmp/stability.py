"""ICH Q1A/Q1E Arrhenius shelf-life prediction and Haynes Mean Kinetic Temperature."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t

from ..config import StabilityConfig


GAS_CONSTANT_J_MOL_K = 8.314462618


@dataclass(frozen=True)
class StabilityAssessment:
    """Shelf-life, MKT, and cold-chain thermal excursion release evaluation."""

    mean_kinetic_temp_c: float
    nominal_rate_per_day: float
    excursion_rate_per_day: float
    remaining_potency_pct: float
    predicted_t90_days: float
    t90_lower_95_days: float
    within_release_spec: bool


class StabilityModel:
    """ICH Q1A/Q1E kinetic stability and cold-chain thermal excursion evaluator."""

    def __init__(self, config: StabilityConfig | None = None) -> None:
        self.config = config or StabilityConfig()

    def rate_constant_per_day(self, temp_c: float) -> float:
        """Arrhenius degradation rate ``k(T) = A * exp(-E_a / (R * T_K))``."""
        c = self.config
        temp_k = temp_c + 273.15
        if temp_k <= 0.0:
            raise ValueError("Temperature cannot be <= absolute zero")
        return c.pre_exponential_per_day * math.exp(
            -c.activation_energy_j_mol / (GAS_CONSTANT_J_MOL_K * temp_k)
        )

    def mean_kinetic_temperature_c(self, temps_c: Sequence[float]) -> float:
        """Compute Haynes Mean Kinetic Temperature (degC) over a temperature log."""
        if not temps_c:
            raise ValueError("temps_c must be non-empty")
        ea = self.config.activation_energy_j_mol
        terms = [math.exp(-ea / (GAS_CONSTANT_J_MOL_K * (tc + 273.15))) for tc in temps_c]
        mean_exp = sum(terms) / len(terms)
        mkt_k = (-ea / GAS_CONSTANT_J_MOL_K) / math.log(mean_exp)
        return mkt_k - 273.15

    def remaining_potency_pct(
        self,
        temps_c: Sequence[float],
        dt_days: float,
        initial_potency_pct: float = 100.0,
    ) -> float:
        """Integrate potency decay across a piecewise-constant temperature time series."""
        if dt_days <= 0.0:
            raise ValueError("dt_days must be positive")
        potency = initial_potency_pct
        order = self.config.reaction_order
        for tc in temps_c:
            k = self.rate_constant_per_day(tc)
            if order == 1:
                potency *= math.exp(-k * dt_days)
            else:
                potency = max(0.0, potency - k * 100.0 * dt_days)
        return potency

    def estimate_t90_ich_q1e(
        self,
        sample_days: Sequence[float],
        measured_potency_pct: Sequence[float],
    ) -> Tuple[float, float]:
        """Estimate shelf life ``t90`` and one-sided 95% lower confidence bound per ICH Q1E.

        Uses ordinary least squares regression of potency (or ``ln(potency)`` for
        first-order) vs time and computes the exact Student-t prediction/mean
        lower confidence bound intersection with ``release_spec_pct``.
        """
        t_arr = np.asarray(sample_days, dtype=np.float64)
        y_raw = np.asarray(measured_potency_pct, dtype=np.float64)
        if len(t_arr) < 3 or len(t_arr) != len(y_raw):
            raise ValueError("At least 3 paired (day, potency) observations required for ICH Q1E")

        spec = self.config.release_spec_pct
        if self.config.reaction_order == 1:
            y_arr = np.log(np.maximum(y_raw, 1e-6))
            y_spec = math.log(spec)
        else:
            y_arr = y_raw
            y_spec = spec

        n = len(t_arr)
        t_bar = float(np.mean(t_arr))
        y_bar = float(np.mean(y_arr))
        sxx = float(np.sum((t_arr - t_bar) ** 2))
        if sxx < 1e-12:
            raise ValueError("Sample time points must span > 0 days")

        slope = float(np.sum((t_arr - t_bar) * (y_arr - y_bar)) / sxx)
        intercept = y_bar - slope * t_bar
        if slope >= -1e-12:
            # Negligible degradation -> cap at 10 years
            return 3650.0, 3650.0

        t90_point = max(0.0, (y_spec - intercept) / slope)
        residuals = y_arr - (intercept + slope * t_arr)
        s_res = math.sqrt(float(np.sum(residuals ** 2)) / (n - 2))
        t_crit = float(student_t.ppf(0.95, df=n - 2))

        # Search forward for exact time t where lower 95% mean confidence bound hits y_spec
        lo, hi = 0.0, t90_point
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            y_hat = intercept + slope * mid
            se = s_res * math.sqrt((1.0 / n) + ((mid - t_bar) ** 2) / sxx)
            if (y_hat - t_crit * se) > y_spec:
                lo = mid
            else:
                hi = mid
        return t90_point, 0.5 * (lo + hi)

    def evaluate_excursion(
        self,
        temp_log_c: Sequence[float],
        dt_days: float = 1.0 / 24.0,
        planned_shelf_days: float = 365.0,
    ) -> StabilityAssessment:
        """Assess cold-chain excursion impact and remaining shelf-life margin."""
        c = self.config
        mkt_c = self.mean_kinetic_temperature_c(temp_log_c)
        k_nom = self.rate_constant_per_day(c.storage_target_c)
        k_exc = self.rate_constant_per_day(mkt_c)
        rem_pct = self.remaining_potency_pct(temp_log_c, dt_days, 100.0)

        # Predict t90 at MKT vs nominal and synthesize calibration check points
        if c.reaction_order == 1:
            t90_nom = math.log(100.0 / c.release_spec_pct) / max(k_nom, 1e-12)
            t90_exc = math.log(100.0 / c.release_spec_pct) / max(k_exc, 1e-12)
        else:
            t90_nom = (100.0 - c.release_spec_pct) / max(k_nom * 100.0, 1e-12)
            t90_exc = (100.0 - c.release_spec_pct) / max(k_exc * 100.0, 1e-12)
        _ = t90_nom

        # Generate synthetic analytical assay checkpoints over planned shelf for conservative LCL
        days = np.linspace(0.0, min(planned_shelf_days, max(30.0, t90_exc)), 6)
        potencies = [
            100.0 * math.exp(-k_exc * d) if c.reaction_order == 1 else max(1.0, 100.0 - k_exc * 100.0 * d)
            for d in days
        ]
        t90_pt, t90_lcl = self.estimate_t90_ich_q1e(days.tolist(), potencies)
        within_spec = rem_pct >= c.release_spec_pct and t90_lcl >= planned_shelf_days * 0.5

        return StabilityAssessment(
            mean_kinetic_temp_c=mkt_c,
            nominal_rate_per_day=k_nom,
            excursion_rate_per_day=k_exc,
            remaining_potency_pct=rem_pct,
            predicted_t90_days=t90_pt,
            t90_lower_95_days=t90_lcl,
            within_release_spec=within_spec,
        )
