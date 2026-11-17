"""Tests for IEC 61131-3 Function Blocks and deterministic OB1 scan-cycle runtime."""

from __future__ import annotations

import pytest

from batch_sentinel.plant import DEFAULT_PLANT_LADDER_SPECS, DEFAULT_PLANT_PLC_ST
from batch_sentinel.plc import (
    CTD,
    CTU,
    F_TRIG,
    PID_FB,
    PLCRuntime,
    PLCWatchdogError,
    R_TRIG,
    TOF,
    TON,
    TP,
    parse_ladder_rungs,
    parse_structured_text,
)


def test_timers_ton_tof_tp_semantics() -> None:
    ton = TON()
    for _ in range(4):
        ton.call({"IN": True, "PT": 0.25}, dt_s=0.05)
        assert not ton.Q
    ton.call({"IN": True, "PT": 0.25}, dt_s=0.05)
    assert ton.Q
    ton.call({"IN": False, "PT": 0.25}, dt_s=0.05)
    assert not ton.Q
    assert ton.ET == pytest.approx(0.0)

    tof = TOF()
    tof.call({"IN": True, "PT": 0.15}, dt_s=0.05)
    assert tof.Q
    tof.call({"IN": False, "PT": 0.15}, dt_s=0.05)
    assert tof.Q
    tof.call({"IN": False, "PT": 0.15}, dt_s=0.10)
    assert not tof.Q

    tp = TP()
    tp.call({"IN": True, "PT": 0.12}, dt_s=0.05)
    assert tp.Q
    tp.call({"IN": False, "PT": 0.12}, dt_s=0.05)
    assert tp.Q
    tp.call({"IN": False, "PT": 0.12}, dt_s=0.05)
    assert not tp.Q


def test_counters_and_edge_triggers() -> None:
    ctu = CTU()
    for pulse in (True, False, True, False, True):
        ctu.call({"CU": pulse, "R": False, "PV": 3}, dt_s=0.05)
    assert ctu.CV == 3
    assert ctu.Q
    ctu.call({"CU": False, "R": True, "PV": 3}, dt_s=0.05)
    assert ctu.CV == 0
    assert not ctu.Q

    ctd = CTD()
    ctd.call({"LD": True, "PV": 2, "CD": False}, dt_s=0.05)
    assert ctd.CV == 2
    for pulse in (True, False, True):
        ctd.call({"LD": False, "CD": pulse, "PV": 2}, dt_s=0.05)
    assert ctd.CV == 0
    assert ctd.Q

    rt = R_TRIG()
    ft = F_TRIG()
    rt.call({"CLK": True}, 0.05)
    assert rt.Q
    rt.call({"CLK": True}, 0.05)
    assert not rt.Q
    ft.call({"CLK": True}, 0.05)
    assert not ft.Q
    ft.call({"CLK": False}, 0.05)
    assert ft.Q


def test_pid_fb_anti_windup_and_bumpless_manual_transfer() -> None:
    pid = PID_FB(KP=2.0, KI=1.0, KD=0.2, OUT_MIN=0.0, OUT_MAX=100.0)
    # Drive into saturation with large positive error
    for _ in range(40):
        pid.call({"PV": 0.0, "SP": 100.0}, dt_s=0.5)
    assert pid.OUT == pytest.approx(100.0)
    sat_integral = pid._integral
    # Because anti-windup freezes integrator when saturated high, reversing error recovers quickly
    pid.call({"PV": 110.0, "SP": 100.0}, dt_s=0.5)
    assert pid._integral <= sat_integral

    # Manual bumpless mode
    pid.call({"PV": 50.0, "SP": 50.0, "MANUAL": True, "MANUAL_VAL": 42.0}, dt_s=0.5)
    assert pid.OUT == pytest.approx(42.0)
    pid.call({"PV": 50.0, "SP": 50.0, "MANUAL": False}, dt_s=0.5)
    assert pid.OUT == pytest.approx(42.0)


def test_plc_runtime_ob1_scan_and_stuck_fault_injection() -> None:
    prog = parse_structured_text(DEFAULT_PLANT_PLC_ST)
    ladder = parse_ladder_rungs(
        "L1",
        DEFAULT_PLANT_LADDER_SPECS,
        tag_types={"MASTER_RUN_PERMIT": "BOOL", "PRESSURE_BAR": "REAL"},
    )
    rt = PLCRuntime(prog, ladder=ladder, scan_period_s=0.05)
    for _ in range(4):
        tel = rt.step_scan({"START_CMD": True, "PRESSURE_BAR": 1.2, "DO_MG_L": 3.0})
        assert tel.watchdog_ok
    assert rt.tags["VALVE_INLET_CMD"] is True
    assert rt.tags["PUNCH_ENABLE"] is True

    # Inject E-stop trip
    rt.step_scan({"E_STOP_TRIP": True})
    assert rt.tags["VALVE_INLET_CMD"] is False
    assert rt.tags["ALARM_CODE"] == 101

    # Stuck tag fault injection
    rt.inject_stuck_tag("E_STOP_TRIP", True)
    rt.step_scan({"E_STOP_TRIP": False})
    assert rt.tags["E_STOP_TRIP"] is True
    rt.clear_faults()


def test_runtime_enforces_loop_bound_watchdog() -> None:
    src = """
    PROGRAM LOOP_GUARD
    VAR I : INT := 0; END_VAR
    FOR I := 1 TO 25 DO
        I := I;
    END_FOR;
    END_PROGRAM
    """
    prog = parse_structured_text(src)
    rt = PLCRuntime(prog, max_loop_iterations=15)
    with pytest.raises(PLCWatchdogError):
        rt.step_scan()
