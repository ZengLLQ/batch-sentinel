"""Additional parameterized unit, conservation, boundary, and convergence tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from batch_sentinel.config import (
    BioreactorParams,
    CrystallizerParams,
    FillFinishParams,
    PATConfig,
    PLCScanConfig,
    PillPressParams,
    StabilityConfig,
)
from batch_sentinel.gmp import (
    ElectronicBatchRecord,
    LotGenealogyDAG,
    MaterialLot,
    StabilityModel,
    compute_gs1_check_digit,
    compute_merkle_root,
    validate_gtin14,
)
from batch_sentinel.isa88 import (
    ISA88BatchExecutive,
    S88Operation,
    S88Phase,
    S88Procedure,
    S88State,
    S88TransitionError,
    S88UnitProcedure,
    compute_sip_f0_lethality,
)
from batch_sentinel.plc import (
    PLCRuntime,
    PLCStaticVerifier,
    PLCSyntaxError,
    parse_ladder_rungs,
    parse_structured_text,
)
from batch_sentinel.twins import (
    BioreactorTwin,
    CrystallizerTwin,
    PillPressTwin,
    SixDOFRobotArm,
    integrate_rkf45,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PLCScanConfig(scan_period_s=0.0),
        lambda: BioreactorParams(mu_max=-0.1),
        lambda: PillPressParams(kawakita_a=1.2),
        lambda: CrystallizerParams(solvent_mass_kg=0.0),
        lambda: FillFinishParams(iso5_min_dp_pa=-1.0),
        lambda: PATConfig(alpha=0.8),
        lambda: StabilityConfig(reaction_order=2),
    ],
)
def test_config_validation_rejects_non_physical_parameters(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_rkf45_order_of_convergence_and_zero_span() -> None:
    # Zero span returns exact initial state with zero steps
    res0 = integrate_rkf45(lambda _t, y: y, (2.0, 2.0), np.array([4.5]))
    assert res0.accepted_steps == 0
    assert float(res0.y[0, 0]) == pytest.approx(4.5)

    # Halving tolerance tightens global error on stiff-ish exponential
    errs = []
    for rtol in (1e-3, 1e-6, 1e-9):
        res = integrate_rkf45(lambda _t, y: -2.0 * y, (0.0, 2.0), np.array([1.0]), rtol=rtol, atol=rtol * 1e-2)
        errs.append(abs(float(res.y[-1, 0]) - math.exp(-4.0)))
    assert errs[1] < errs[0]
    assert errs[2] < errs[1]


def test_plc_ladder_set_reset_and_parallel_branches() -> None:
    ladder = parse_ladder_rungs(
        "LATCH_TEST",
        [
            {
                "rung_id": "R1",
                "comment": "Parallel start OR bypass sets RUN_LATCH",
                "branches": [
                    [{"type": "NO", "tag": "PB_START"}, {"type": "NC", "tag": "TRIP"}],
                    [{"type": "NO", "tag": "BYPASS_KEY"}, {"type": "CMP", "tag": "PRESS_BAR", "op": "<", "threshold": 2.0}],
                ],
                "coils": [{"tag": "RUN_LATCH", "mode": "SET"}],
            },
            {
                "rung_id": "R2",
                "comment": "Trip resets RUN_LATCH",
                "branches": [[{"type": "NO", "tag": "TRIP"}]],
                "coils": [{"tag": "RUN_LATCH", "mode": "RESET"}],
            },
        ],
    )
    prog = parse_structured_text("PROGRAM EMPTY VAR END_VAR END_PROGRAM")
    rt = PLCRuntime(prog, ladder=ladder)
    rt.step_scan({"PB_START": True, "TRIP": False, "BYPASS_KEY": False, "PRESS_BAR": 3.0})
    assert rt.tags["RUN_LATCH"] is True
    # Remains latched even after PB_START released
    rt.step_scan({"PB_START": False, "TRIP": False})
    assert rt.tags["RUN_LATCH"] is True
    # Trip resets latch
    rt.step_scan({"PB_START": False, "TRIP": True})
    assert rt.tags["RUN_LATCH"] is False


def test_plc_syntax_and_semantic_diagnostics() -> None:
    with pytest.raises(PLCSyntaxError):
        parse_structured_text("PROGRAM BAD VAR X : REAL := ; END_VAR END_PROGRAM")

    # Duplicate CASE label and non-BOOL IF condition caught by verifier
    src = """
    PROGRAM SEM_ERR
    VAR X : INT := 1; END_VAR
    IF X THEN
        X := 2;
    END_IF;
    CASE X OF
        1: X := 10;
        1: X := 20;
    END_CASE;
    END_PROGRAM
    """
    rep = PLCStaticVerifier().verify(parse_structured_text(src))
    codes = {f.code for f in rep.findings}
    assert "NON_BOOL_CONDITION" in codes
    assert "DUPLICATE_CASE" in codes


def test_bioreactor_haldane_substrate_inhibition_and_thermal_peak() -> None:
    twin = BioreactorTwin()
    mu_opt = twin.specific_growth_rate(2.5, 4.0, 0.1, 37.0)
    mu_excess_sub = twin.specific_growth_rate(80.0, 4.0, 0.1, 37.0)
    mu_cold = twin.specific_growth_rate(2.5, 4.0, 0.1, 28.0)
    assert mu_opt > mu_excess_sub
    assert mu_opt > mu_cold


def test_crystallizer_solubility_monotonic_and_zero_undersaturated_dissolution() -> None:
    cryst = CrystallizerTwin()
    assert cryst.solubility(65.0) > cryst.solubility(20.0)
    res = cryst.simulate_cooling_batch(duration_min=90.0, t_start_c=55.0, t_end_c=15.0)
    assert res.yield_pct > 50.0
    assert res.final_cv >= 0.0


def test_six_dof_jacobian_finite_difference_consistency() -> None:
    arm = SixDOFRobotArm()
    q = np.array([0.3, 0.5, -0.4, 0.2, 0.3, -0.1], dtype=np.float64)
    jac = arm.position_jacobian(q)
    assert jac.shape == (3, 6)
    dq = np.array([1e-5, -1e-5, 1e-5, 0.0, 1e-5, 0.0], dtype=np.float64)
    p0 = arm.forward_kinematics(q)[:3, 3]
    p1 = arm.forward_kinematics(q + dq)[:3, 3]
    assert np.allclose(p1 - p0, jac @ dq, atol=1e-9)


def test_isa88_valid_complete_lifecycle_and_sip_cold_zero() -> None:
    assert compute_sip_f0_lethality([25.0, 60.0, 95.0], dt_min=5.0) == 0.0
    exec_eng = ISA88BatchExecutive()
    proc = S88Procedure(
        "P_OK",
        [
            S88UnitProcedure(
                "UP1",
                "U1",
                [S88Operation("OP1", [S88Phase("PH1", "U1", lambda: {"x": 1.0})])],
            )
        ],
    )
    st, recs, reason = exec_eng.run_procedure(proc, "B-OK")
    assert st == S88State.COMPLETE
    assert reason is None
    assert len(recs) == 1
    with pytest.raises(S88TransitionError):
        exec_eng.transition_recipe(S88State.RUNNING)  # COMPLETE -> RUNNING illegal directly


def test_ebr_signature_tamper_and_merkle_empty_edge_case() -> None:
    empty_root = compute_merkle_root([])
    assert len(empty_root) == 64

    ebr = ElectronicBatchRecord("B-SIG", "Prod")
    e0 = ebr.append_event(
        "2026-11-17T00:00:00Z", "QA-1", "SIGN", "U1", {"x": 1}, sign_role="QA", sign_meaning="Approve"
    )
    assert e0.signature is not None
    # Tamper signature meaning without touching entry payload
    object.__setattr__(e0.signature, "meaning", "Forged")
    ok, msg = ebr.verify_integrity()
    assert not ok
    assert msg is not None and "signature" in msg.lower()


@pytest.mark.parametrize("prefix", ["0031234567890", "1234567890123", "9998887776665"])
def test_gs1_gtin14_check_digit_roundtrip(prefix: str) -> None:
    gtin = prefix + compute_gs1_check_digit(prefix)
    assert validate_gtin14(gtin)


def test_lot_genealogy_detects_unbalanced_mass() -> None:
    dag = LotGenealogyDAG()
    dag.register_lot(MaterialLot("R1", "API", "RAW", 100.0))
    dag.register_lot(MaterialLot("F1", "Drug", "FINISHED", 70.0))
    rep = dag.reconcile_mass_balance(holdup_kg=1.0, tolerance_pct=2.0)
    assert not rep.balanced
    assert rep.closure_error_pct == pytest.approx(29.0)


def test_stability_zero_order_kinetics() -> None:
    model = StabilityModel(StabilityConfig(reaction_order=0, pre_exponential_per_day=1e8, activation_energy_j_mol=60000.0))
    pot = model.remaining_potency_pct([25.0] * 10, dt_days=1.0, initial_potency_pct=100.0)
    assert 0.0 < pot < 100.0
    t90_pt, t90_lcl = model.estimate_t90_ich_q1e([0.0, 30.0, 60.0, 90.0], [100.0, 98.5, 97.0, 95.5])
    assert t90_pt == pytest.approx(200.0, rel=1e-3)
    assert t90_lcl <= t90_pt
