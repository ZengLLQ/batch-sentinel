"""Tests for 6-DOF aseptic robotics IK, cleanroom DP interlock, and ISA-88 executive."""

from __future__ import annotations

import numpy as np
import pytest

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
from batch_sentinel.twins import AsepticFillLineTwin, SixDOFRobotArm


def test_six_dof_ik_converges_to_reachable_targets() -> None:
    arm = SixDOFRobotArm()
    for target in [(0.42, 0.0, 0.30), (0.48, 0.12, 0.28), (0.38, -0.10, 0.34)]:
        q, err = arm.solve_ik_position(target)
        assert err < 1e-3
        fk_pos = arm.forward_kinematics(q)[:3, 3]
        assert np.linalg.norm(fk_pos - np.asarray(target)) == pytest.approx(err, abs=1e-9)


def test_aseptic_line_nominal_and_door_breach_interlock() -> None:
    line = AsepticFillLineTwin()
    nom = line.simulate_batch(vial_count=20)
    assert not nom.airlock_breach_halted
    assert nom.min_dp_pa >= line.params.iso5_min_dp_pa
    assert nom.acceptance_rate_pct > 90.0

    breach = line.simulate_batch(vial_count=20, door_breach_at_vial=8)
    assert breach.airlock_breach_halted
    assert breach.min_dp_pa < line.params.iso5_min_dp_pa
    assert len(breach.vials) < 20


def test_isa88_state_machine_unit_locking_and_sip_f0() -> None:
    # 10 min hold at 121.1 C yields F0 ~ 10.0 min
    f0 = compute_sip_f0_lethality([121.1, 121.1, 121.1], dt_min=5.0)
    assert f0 == pytest.approx(10.0, rel=1e-3)

    exec_eng = ISA88BatchExecutive()
    exec_eng.acquire_unit("UNIT-A", "BATCH-1")
    with pytest.raises(S88TransitionError):
        exec_eng.acquire_unit("UNIT-A", "BATCH-2")
    exec_eng.release_unit("UNIT-A", "BATCH-1")

    phase = S88Phase("PH_TEST", "UNIT-A", lambda: {"ok": 1.0})
    with pytest.raises(S88TransitionError):
        phase.transition(S88State.COMPLETE)  # IDLE -> COMPLETE is illegal

    proc = S88Procedure(
        "PROC_ABORT_TEST",
        [S88UnitProcedure("UP1", "UNIT-A", [S88Operation("OP1", [phase])])],
    )
    state, records, reason = exec_eng.run_procedure(
        proc,
        "BATCH-ABORT",
        abort_on_predicate=lambda _name, _m: "Simulated trip",
    )
    assert state == S88State.ABORTED
    assert reason == "Simulated trip"
    assert records[-1].final_state == S88State.ABORTED
