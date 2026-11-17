"""Tests for Heckel pill-press compaction and PBE cooling crystallizer twins."""

from __future__ import annotations

import pytest

from batch_sentinel.twins import CrystallizerTwin, PillPressTwin


def test_heckel_inversion_kawakita_and_ryshkewitch_physics() -> None:
    press = PillPressTwin()
    for pressure in (50.0, 120.0, 200.0):
        d = press.heckel_relative_density(pressure)
        p_inv = press.heckel_pressure_for_density(d)
        assert p_inv == pytest.approx(pressure, rel=1e-4)

    # Kawakita and tensile strength are monotonically increasing with pressure/density
    assert press.kawakita_volume_reduction(200.0) > press.kawakita_volume_reduction(50.0)
    assert press.tensile_strength_mpa(0.92) > press.tensile_strength_mpa(0.75)


def test_pill_press_closed_loop_control_and_overload_estop() -> None:
    press = PillPressTwin()
    nom = press.simulate_run(n_tablets=50, bulk_density_drift=0.05)
    assert not nom.e_stop_triggered
    assert nom.mean_mass_mg == pytest.approx(press.params.target_mass_mg, rel=0.03)
    assert nom.acceptance_rate_pct > 85.0

    fault = press.simulate_run(n_tablets=50, inject_overload_at=20)
    assert fault.e_stop_triggered
    assert fault.e_stop_tablet_index == 20
    assert fault.peak_force_kn > press.params.max_force_kn
    assert len(fault.tablets) == 21


def test_crystallizer_pbe_mass_conservation_and_cooling_profile_effect() -> None:
    cryst = CrystallizerTwin()
    cubic = cryst.simulate_cooling_batch(duration_min=180.0, cooling_profile="cubic")
    crash = cryst.simulate_cooling_batch(duration_min=180.0, cooling_profile="crash")

    # Solute + crystal mass must be conserved tightly (< 0.2%)
    assert cubic.solute_mass_closure_error_pct < 0.2
    assert crash.solute_mass_closure_error_pct < 0.2
    assert cubic.crystal_yield_kg > 1.0
    assert cubic.final_l43_um > 55.0
    # Controlled cubic cooling avoids crash-nucleation fines -> larger mean crystal size L43
    assert cubic.final_l43_um > crash.final_l43_um
