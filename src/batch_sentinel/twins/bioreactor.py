"""Fed-batch / perfusion bioreactor digital twin with cascade DO/pH/T PID & MPC feed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ..config import BioreactorParams
from ..plc.blocks import PID_FB
from .solvers import integrate_rkf45


@dataclass
class BioreactorState:
    """Instantaneous bioreactor state vector."""

    time_h: float
    volume_l: float
    biomass_g_l: float
    substrate_g_l: float
    lactate_g_l: float
    product_g_l: float
    dissolved_o2_mg_l: float
    temperature_c: float
    ph: float
    feed_rate_l_h: float
    bleed_rate_l_h: float
    agitation_rpm: float
    aeration_vvm: float
    jacket_temp_c: float
    base_flow_l_h: float


@dataclass
class BioreactorTrajectory:
    """Full simulated bioreactor campaign with mass-balance verification."""

    states: List[BioreactorState]
    total_glucose_fed_g: float
    total_product_harvested_g: float
    final_titer_g_l: float
    min_do_mg_l: float
    max_lactate_g_l: float
    carbon_mass_closure_error_pct: float
    alarm_events: List[str]


class BioreactorTwin:
    """Nonlinear fed-batch/perfusion bioreactor simulator with cascade PID & MPC control.

    State vector ``y`` (8 coupled ODEs):
        ``[V, X, S, L_inh, P, C_O2, T_c, pH]``
    where:
    - ``V``: liquid working volume (L)
    - ``X``: viable cell / biomass concentration (g/L)
    - ``S``: limiting glucose substrate (g/L)
    - ``L_inh``: inhibitory lactate byproduct (g/L)
    - ``P``: recombinant monoclonal antibody titer (g/L)
    - ``C_O2``: dissolved oxygen concentration (mg/L)
    - ``T_c``: broth temperature (degC)
    - ``pH``: culture broth pH
    """

    def __init__(
        self,
        params: Optional[BioreactorParams] = None,
        do_setpoint_mg_l: float = 3.6,
        temp_setpoint_c: float = 37.0,
        ph_setpoint: float = 7.15,
        substrate_setpoint_g_l: float = 2.5,
    ) -> None:
        self.params = params or BioreactorParams()
        self.do_setpoint_mg_l = do_setpoint_mg_l
        self.temp_setpoint_c = temp_setpoint_c
        self.ph_setpoint = ph_setpoint
        self.substrate_setpoint_g_l = substrate_setpoint_g_l

        # Industrial anti-windup PID loops tuned for 0.1 h supervisory sub-steps
        self.do_pid = PID_FB(
            KP=22.0, KI=40.0, KD=0.0, OUT_MIN=80.0, OUT_MAX=420.0, SP=do_setpoint_mg_l, _integral=160.0
        )
        self.temp_pid = PID_FB(
            KP=2.2, KI=1.2, KD=0.0, OUT_MIN=15.0, OUT_MAX=42.0, SP=temp_setpoint_c, _integral=temp_setpoint_c
        )
        self.ph_pid = PID_FB(
            KP=0.5, KI=0.4, KD=0.0, OUT_MIN=0.0, OUT_MAX=0.6, SP=ph_setpoint
        )

    def thermal_factor(self, temp_c: float) -> float:
        """Bell-shaped Arrhenius growth/denaturation thermal modifier peaking at t_opt_c."""
        dt_opt = temp_c - self.params.t_opt_c
        return math.exp(-0.045 * dt_opt * dt_opt)

    def specific_growth_rate(self, s_g_l: float, c_o2_mg_l: float, lac_g_l: float, temp_c: float) -> float:
        """Monod + Haldane substrate inhibition + DO Monod + lactate inhibition kinetics."""
        p = self.params
        s = max(0.0, s_g_l)
        o2 = max(0.0, c_o2_mg_l)
        lac = max(0.0, lac_g_l)
        f_sub = s / (p.k_s + s + (s * s) / p.k_i_sub)
        f_o2 = o2 / (p.k_o2 + o2)
        f_lac = p.k_i_lac / (p.k_i_lac + lac)
        f_temp = self.thermal_factor(temp_c)
        return p.mu_max * f_sub * f_o2 * f_lac * f_temp

    def compute_kla(self, agitation_rpm: float, aeration_vvm: float) -> float:
        """Van't Riet style volumetric oxygen mass-transfer correlation kLa (1/h)."""
        p = self.params
        rpm_ratio = max(0.2, agitation_rpm / 180.0)
        vvm_ratio = max(0.1, aeration_vvm / 0.5)
        return p.kla_base * (rpm_ratio ** 1.15) * (vvm_ratio ** 0.55)

    def mpc_substrate_feed(
        self,
        volume_l: float,
        biomass_g_l: float,
        substrate_g_l: float,
        mu: float,
        dt_h: float,
        perfusion_bleed_l_h: float,
    ) -> float:
        """One-step receding-horizon MPC feed calculation holding glucose near setpoint.

        Solves analytically for the non-negative feed rate ``F_in`` bounded by remaining
        headspace and pump capacity so ``S(t + dt) -> substrate_setpoint_g_l``.
        """
        p = self.params
        if volume_l >= 0.96 * p.v_max:
            return perfusion_bleed_l_h  # Match bleed in perfusion hold
        q_s = mu / p.y_xs + p.m_s
        consumption_g_h = q_s * biomass_g_l * volume_l
        deficit_g_h = (self.substrate_setpoint_g_l - substrate_g_l) * volume_l / max(dt_h, 0.25)
        required_g_h = consumption_g_h + deficit_g_h + perfusion_bleed_l_h * substrate_g_l
        feed_l_h = required_g_h / max(p.s_feed - substrate_g_l, 1.0)
        max_headroom_l_h = max(0.0, (0.96 * p.v_max - volume_l) / max(dt_h, 0.25) + perfusion_bleed_l_h)
        return float(max(0.0, min(feed_l_h, 18.0, max_headroom_l_h)))

    def simulate(
        self,
        duration_h: float = 120.0,
        step_h: float = 1.0,
        initial_volume_l: float = 800.0,
        initial_biomass_g_l: float = 0.65,
        initial_substrate_g_l: float = 4.5,
        perfusion_start_h: float = 72.0,
        perfusion_rate_l_h: float = 8.0,
        inject_do_drop: bool = False,
        do_drop_window_h: tuple[float, float] = (48.0, 56.0),
    ) -> BioreactorTrajectory:
        """Simulate a full fed-batch + perfusion bioreactor campaign via adaptive RKF45."""
        p = self.params
        # State: [V, X, S, L_inh, P, C_O2, T_c, pH, cum_S_consumed_g, cum_S_bled_g, cum_P_bled_g]
        y = np.array(
            [
                initial_volume_l,
                initial_biomass_g_l,
                initial_substrate_g_l,
                0.05,            # initial lactate g/L
                0.0,             # initial mAb product g/L
                6.2,             # initial DO mg/L
                p.t_opt_c,       # initial temp C
                7.18,            # initial pH
                0.0,             # cumulative substrate consumed (g)
                0.0,             # cumulative substrate lost in bleed (g)
                0.0,             # cumulative product harvested in bleed (g)
            ],
            dtype=np.float64,
        )

        states: List[BioreactorState] = []
        alarms: List[str] = []
        total_glucose_fed_g = initial_volume_l * initial_substrate_g_l

        n_steps = max(1, int(math.ceil(duration_h / step_h)))
        t_cur = 0.0
        substep_target_h = 0.1

        def eval_controls(t_val: float, state_arr: NDArray[np.float64], ctrl_dt: float) -> tuple[float, float, float, float, float, float]:
            vol_v, x_v, s_v, lac_v, _p_v, do_v, temp_v, ph_v = [float(v) for v in state_arr[:8]]
            mu_v = self.specific_growth_rate(s_v, do_v, lac_v, temp_v)
            bleed_v = perfusion_rate_l_h if t_val >= perfusion_start_h else 0.0
            feed_v = self.mpc_substrate_feed(vol_v, x_v, s_v, mu_v, max(ctrl_dt, 0.25), bleed_v)

            self.do_pid.call({"PV": do_v, "SP": self.do_setpoint_mg_l}, dt_s=ctrl_dt)
            agitation_v = self.do_pid.OUT
            aeration_v = 0.25 + 0.75 * ((agitation_v - 80.0) / 340.0)
            if inject_do_drop and do_drop_window_h[0] <= t_val <= do_drop_window_h[1]:
                aeration_v *= 0.10
                agitation_v = 90.0

            self.temp_pid.call({"PV": temp_v, "SP": self.temp_setpoint_c}, dt_s=ctrl_dt)
            jacket_v = self.temp_pid.OUT

            self.ph_pid.call({"PV": ph_v, "SP": self.ph_setpoint}, dt_s=ctrl_dt)
            base_v = self.ph_pid.OUT
            return feed_v, bleed_v, agitation_v, aeration_v, jacket_v, base_v

        for step_idx in range(n_steps + 1):
            vol, x, s, lac, prod, do_val, temp_c, ph_val = [float(v) for v in y[:8]]
            mu = self.specific_growth_rate(s, do_val, lac, temp_c)
            bleed = perfusion_rate_l_h if t_cur >= perfusion_start_h else 0.0
            feed = self.mpc_substrate_feed(vol, x, s, mu, 0.25, bleed)
            agitation_rpm = self.do_pid.OUT if step_idx > 0 else 180.0
            aeration_vvm = 0.25 + 0.75 * ((agitation_rpm - 80.0) / 340.0)
            if inject_do_drop and do_drop_window_h[0] <= t_cur <= do_drop_window_h[1]:
                aeration_vvm *= 0.10
                agitation_rpm = 90.0
            jacket_temp_c = self.temp_pid.OUT if step_idx > 0 else p.t_opt_c
            base_flow_l_h = self.ph_pid.OUT if step_idx > 0 else 0.0

            states.append(
                BioreactorState(
                    time_h=t_cur,
                    volume_l=vol,
                    biomass_g_l=x,
                    substrate_g_l=s,
                    lactate_g_l=lac,
                    product_g_l=prod,
                    dissolved_o2_mg_l=do_val,
                    temperature_c=temp_c,
                    ph=ph_val,
                    feed_rate_l_h=feed,
                    bleed_rate_l_h=bleed,
                    agitation_rpm=agitation_rpm,
                    aeration_vvm=aeration_vvm,
                    jacket_temp_c=jacket_temp_c,
                    base_flow_l_h=base_flow_l_h,
                )
            )

            if do_val < 1.5:
                alarms.append(f"t={t_cur:.1f}h: LOW_DO_ALARM ({do_val:.2f} mg/L < 1.50)")
            if lac > 3.5:
                alarms.append(f"t={t_cur:.1f}h: HIGH_LACTATE_ALARM ({lac:.2f} g/L > 3.50)")
            if temp_c > 38.5 or temp_c < 35.0:
                alarms.append(f"t={t_cur:.1f}h: TEMP_DEVIATION ({temp_c:.2f} C)")

            if step_idx == n_steps:
                break

            dt_outer = min(step_h, duration_h - t_cur)
            n_sub = max(1, int(math.ceil(dt_outer / substep_target_h)))
            dt_sub = dt_outer / n_sub
            t_sub = t_cur

            for _ in range(n_sub):
                f_in, f_bleed, agit, aer, t_jac, f_base = eval_controls(t_sub, y, dt_sub)
                kla = self.compute_kla(agit, aer)

                def rhs(_t: float, st: NDArray[np.float64]) -> NDArray[np.float64]:
                    v_i, x_i, s_i, lac_i, p_i, o2_i, t_i, _ph_i, _cs, _bs, _bp = st
                    v_safe = max(v_i, 1.0)
                    mu_i = self.specific_growth_rate(s_i, o2_i, lac_i, t_i)
                    k_d = p.k_d_base * (1.0 + 0.35 * max(0.0, lac_i - 2.0))
                    f_total_in = f_in + f_base
                    cell_retention = 0.85

                    dv = f_total_in - f_bleed
                    dx = (mu_i - k_d) * x_i - (f_total_in / v_safe) * x_i - (1.0 - cell_retention) * (f_bleed / v_safe) * x_i

                    hypoxia_shift = 1.0 + 3.5 * max(0.0, 1.0 - o2_i / 1.5)
                    q_s = mu_i / p.y_xs + p.m_s * hypoxia_shift
                    ds = (f_in * p.s_feed) / v_safe - (f_total_in / v_safe) * s_i - q_s * x_i

                    q_lac = p.y_lac_s * q_s * hypoxia_shift - 0.015 * lac_i
                    dlac = q_lac * x_i - (f_total_in / v_safe) * lac_i

                    q_p = p.q_p_max * (1.0 + p.alpha_growth_prod * (mu_i / p.mu_max)) * self.thermal_factor(t_i)
                    dp = q_p * x_i - (f_total_in / v_safe) * p_i

                    our_mg_l_h = p.q_o2_spec * 1000.0 * x_i
                    do2 = kla * (p.c_o2_star - o2_i) - our_mg_l_h - (f_total_in / v_safe) * o2_i

                    metabolic_w = (q_s * x_i * v_safe / 3600.0) * p.delta_h_met
                    jacket_w = p.ua_jacket * (t_jac - t_i)
                    thermal_mass_j_k = v_safe * p.rho_broth * p.cp_broth
                    dtemp = 3600.0 * (metabolic_w + jacket_w) / thermal_mass_j_k

                    acid_mol_l_h = (max(0.0, q_lac) * x_i) / 90.08
                    base_mol_l_h = (f_base * p.base_conc_mol_l) / v_safe
                    dph = (base_mol_l_h - acid_mol_l_h) / p.buffer_capacity

                    d_cons_s = q_s * x_i * v_safe
                    d_bled_s = f_bleed * s_i
                    d_bled_p = f_bleed * p_i

                    return np.array(
                        [dv, dx, ds, dlac, dp, do2, dtemp, dph, d_cons_s, d_bled_s, d_bled_p],
                        dtype=np.float64,
                    )

                res = integrate_rkf45(
                    rhs,
                    (t_sub, t_sub + dt_sub),
                    y,
                    rtol=1e-6,
                    atol=1e-8,
                    non_negative_indices=(0, 1, 2, 3, 4, 5, 8, 9, 10),
                )
                y = res.y[-1]
                y[7] = float(np.clip(y[7], 4.0, 9.5))
                total_glucose_fed_g += f_in * p.s_feed * dt_sub
                t_sub += dt_sub

            t_cur += dt_outer

        accounted_glucose_g = float(y[0] * y[2]) + float(y[8]) + float(y[9])
        closure_err_pct = 100.0 * abs(total_glucose_fed_g - accounted_glucose_g) / max(total_glucose_fed_g, 1.0)

        return BioreactorTrajectory(
            states=states,
            total_glucose_fed_g=total_glucose_fed_g,
            total_product_harvested_g=float(y[10]) + float(y[0] * y[4]),
            final_titer_g_l=float(y[4]),
            min_do_mg_l=min(st.dissolved_o2_mg_l for st in states),
            max_lactate_g_l=max(st.lactate_g_l for st in states),
            carbon_mass_closure_error_pct=closure_err_pct,
            alarm_events=alarms,
        )
