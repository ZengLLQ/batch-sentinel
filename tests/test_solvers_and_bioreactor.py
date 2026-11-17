"""Tests for adaptive RKF45 ODE solver accuracy and bioreactor twin conservation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from batch_sentinel.twins import BioreactorTwin, integrate_rkf45


def test_rkf45_exponential_and_harmonic_convergence() -> None:
    # Exponential decay dy/dt = -0.5 y, y(0) = 3.0 -> exact 3 * exp(-2.5)
    res_exp = integrate_rkf45(
        lambda _t, y: -0.5 * y,
        (0.0, 5.0),
        np.array([3.0]),
        rtol=1e-8,
        atol=1e-10,
    )
    exact = 3.0 * math.exp(-2.5)
    assert abs(float(res_exp.y[-1, 0]) - exact) < 1e-7

    # 2D harmonic oscillator: energy x^2 + v^2 must remain conserved
    res_osc = integrate_rkf45(
        lambda _t, y: np.array([y[1], -y[0]]),
        (0.0, 2.0 * math.pi),
        np.array([1.0, 0.0]),
        rtol=1e-8,
        atol=1e-10,
    )
    assert abs(float(res_osc.y[-1, 0]) - 1.0) < 1e-6
    assert abs(float(res_osc.y[-1, 1])) < 1e-6


def test_bioreactor_nominal_vs_hypoxic_do_drop_and_mass_closure() -> None:
    twin = BioreactorTwin()
    nominal = twin.simulate(duration_h=96.0, step_h=2.0, inject_do_drop=False)
    hypoxic = twin.simulate(
        duration_h=96.0,
        step_h=2.0,
        inject_do_drop=True,
        do_drop_window_h=(36.0, 52.0),
    )

    assert nominal.final_titer_g_l > 1.0
    assert nominal.carbon_mass_closure_error_pct < 1.5
    assert nominal.min_do_mg_l > 1.8

    # Hypoxic excursion must drop DO, spike lactate, and trigger alarms
    assert hypoxic.min_do_mg_l < 1.5
    assert hypoxic.max_lactate_g_l > nominal.max_lactate_g_l
    assert len(hypoxic.alarm_events) >= 1
    assert all(st.biomass_g_l >= 0.0 and st.substrate_g_l >= 0.0 for st in nominal.states)
