"""Method-of-Moments Population Balance cooling crystallizer digital twin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ..config import CrystallizerParams
from .solvers import integrate_rkf45


@dataclass(frozen=True)
class CrystallizerState:
    """Instantaneous state of the cooling crystallizer PBE."""

    time_min: float
    temp_c: float
    jacket_temp_c: float
    concentration_g_kg: float
    solubility_g_kg: float
    supersaturation_ratio: float
    mu0_per_kg: float
    l43_mean_um: float
    cv_size: float
    crystal_mass_kg: float


@dataclass
class CrystallizerResult:
    """Full cooling crystallization trajectory and mass-closure metrics."""

    states: List[CrystallizerState]
    final_l43_um: float
    final_cv: float
    crystal_yield_kg: float
    yield_pct: float
    solute_mass_closure_error_pct: float


class CrystallizerTwin:
    """Population Balance Equation (Method of Moments) cooling crystallizer twin.

    Integrates the 7-state ODE system ``[mu_0, mu_1, mu_2, mu_3, mu_4, C, T_c]``
    where ``mu_k = int_0^inf L^k n(L, t) dL`` (units ``um^k / kg_solvent``) under
    size-independent McCabe ``Delta L`` crystal growth and secondary nucleation.
    """

    def __init__(self, params: Optional[CrystallizerParams] = None) -> None:
        self.params = params or CrystallizerParams()

    def solubility(self, temp_c: float) -> float:
        """Equilibrium solubility ``C*(T)`` in ``g solute / kg solvent``."""
        p = self.params
        return p.sol_a + p.sol_b * math.exp(p.sol_c * temp_c)

    def simulate_cooling_batch(
        self,
        duration_min: float = 180.0,
        step_min: float = 3.0,
        t_start_c: float = 65.0,
        t_end_c: float = 10.0,
        cooling_profile: str = "cubic",  # "linear" | "cubic" | "crash"
        seed_mass_kg: float = 0.45,
        seed_mean_um: float = 55.0,
    ) -> CrystallizerResult:
        """Simulate a seeded cooling crystallization campaign via RKF45."""
        p = self.params
        # Initial solute slightly supersaturated at t_start_c
        c_init = self.solubility(t_start_c) * 1.04
        # Convert seed mass (kg) -> mu_3 (um^3 / kg_solvent):
        # M_cryst = rho_crystal (kg/m^3) * k_v * mu_3 * 1e-18 (m^3/um^3) * solvent_mass_kg
        mu3_init = (seed_mass_kg / p.solvent_mass_kg) / (p.rho_crystal * p.k_v_shape * 1e-18)
        # Monodisperse-approx seed moments around seed_mean_um
        mu0_init = mu3_init / (seed_mean_um ** 3)
        mu1_init = mu0_init * seed_mean_um
        mu2_init = mu0_init * (seed_mean_um ** 2)
        mu4_init = mu0_init * (seed_mean_um ** 4)

        initial_total_solute_kg = p.solvent_mass_kg * (c_init / 1000.0) + seed_mass_kg

        y = np.array(
            [mu0_init, mu1_init, mu2_init, mu3_init, mu4_init, c_init, t_start_c],
            dtype=np.float64,
        )
        states: List[CrystallizerState] = []
        n_steps = max(1, int(math.ceil(duration_min / step_min)))
        t_cur = 0.0

        def target_temp(t_m: float) -> float:
            frac = min(1.0, max(0.0, t_m / max(duration_min, 1e-6)))
            if cooling_profile == "cubic":
                # Slow initial cooling in metastable zone, faster later (controlled nucleation)
                return t_start_c - (t_start_c - t_end_c) * (frac ** 3)
            if cooling_profile == "crash":
                # Rapid exponential crash cooling creating high early supersaturation
                return t_end_c + (t_start_c - t_end_c) * math.exp(-6.0 * frac)
            return t_start_c - (t_start_c - t_end_c) * frac

        for step_idx in range(n_steps + 1):
            mu0, mu1, mu2, mu3, mu4, c_val, temp_c = [float(v) for v in y]
            c_star = self.solubility(temp_c)
            ss_ratio = c_val / max(c_star, 1e-6)
            l43 = (mu4 / mu3) if mu3 > 1e-9 else 0.0
            mean_l = (mu1 / mu0) if mu0 > 1e-9 else 0.0
            var_l = max(0.0, (mu2 / mu0) - mean_l ** 2) if mu0 > 1e-9 else 0.0
            cv = (math.sqrt(var_l) / mean_l) if mean_l > 1e-9 else 0.0
            cryst_mass_kg = p.rho_crystal * p.k_v_shape * mu3 * 1e-18 * p.solvent_mass_kg
            t_j = target_temp(t_cur) - 2.0

            states.append(
                CrystallizerState(
                    time_min=t_cur,
                    temp_c=temp_c,
                    jacket_temp_c=t_j,
                    concentration_g_kg=c_val,
                    solubility_g_kg=c_star,
                    supersaturation_ratio=ss_ratio,
                    mu0_per_kg=mu0,
                    l43_mean_um=l43,
                    cv_size=cv,
                    crystal_mass_kg=cryst_mass_kg,
                )
            )

            if step_idx == n_steps:
                break

            dt = min(step_min, duration_min - t_cur)
            t_sp = target_temp(t_cur + 0.5 * dt)

            def rhs(_t: float, st: NDArray[np.float64]) -> NDArray[np.float64]:
                m0, m1, m2, m3, _m4, c_i, temp_i = st
                c_eq = self.solubility(float(temp_i))
                s_rat = float(c_i) / max(c_eq, 1e-6)
                rel_ss = s_rat - 1.0

                if rel_ss >= 0.0:
                    growth = p.k_g * (rel_ss ** p.g_growth)
                    magma_kg_kg = max(1e-6, p.rho_crystal * p.k_v_shape * float(m3) * 1e-18)
                    nuc = p.k_b * (rel_ss ** p.b_nuc) * (magma_kg_kg ** p.j_magma)
                else:
                    # Undersaturated fines dissolution
                    growth = -p.k_d_fines * abs(rel_ss)
                    nuc = 0.0

                dm0 = nuc
                dm1 = growth * m0
                dm2 = 2.0 * growth * m1
                dm3 = 3.0 * growth * m2
                dm4 = 4.0 * growth * m3

                # dC/dt in g/(kg_solvent * min): 3 * rho * k_v * G * mu_2 * 1e-18 * 1000
                dc = -3.0 * p.rho_crystal * p.k_v_shape * growth * m2 * 1e-15
                # Jacket proportional thermal tracking (sec -> min factor 60)
                t_jacket_cmd = t_sp - 2.5 * (float(temp_i) - t_sp)
                dtemp = 60.0 * p.ua_jacket_w_k * (t_jacket_cmd - float(temp_i)) / p.heat_cap_j_k

                return np.array([dm0, dm1, dm2, dm3, dm4, dc, dtemp], dtype=np.float64)

            res = integrate_rkf45(
                rhs,
                (t_cur, t_cur + dt),
                y,
                rtol=1e-6,
                atol=1e-8,
                non_negative_indices=(0, 1, 2, 3, 4, 5),
            )
            y = res.y[-1]
            t_cur += dt

        final = states[-1]
        final_total_solute_kg = p.solvent_mass_kg * (final.concentration_g_kg / 1000.0) + final.crystal_mass_kg
        closure_err_pct = (
            100.0
            * abs(initial_total_solute_kg - final_total_solute_kg)
            / max(initial_total_solute_kg, 1e-6)
        )
        yield_pct = 100.0 * (final.crystal_mass_kg - seed_mass_kg) / max(initial_total_solute_kg - seed_mass_kg, 1e-6)

        return CrystallizerResult(
            states=states,
            final_l43_um=final.l43_mean_um,
            final_cv=final.cv_size,
            crystal_yield_kg=final.crystal_mass_kg,
            yield_pct=max(0.0, yield_pct),
            solute_mass_closure_error_pct=closure_err_pct,
        )
