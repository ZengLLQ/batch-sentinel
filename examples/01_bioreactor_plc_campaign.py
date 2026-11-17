"""Example 01: IEC 61131-3 PLC Interlock Verification & Fed-Batch Bioreactor Twin."""

from __future__ import annotations

from batch_sentinel.plant import (
    DEFAULT_INTERLOCK_RULES,
    DEFAULT_PLANT_LADDER_SPECS,
    DEFAULT_PLANT_PLC_ST,
)
from batch_sentinel.plc import (
    PLCRuntime,
    PLCStaticVerifier,
    parse_ladder_rungs,
    parse_structured_text,
)
from batch_sentinel.twins import BioreactorTwin
from batch_sentinel.visualizer import render_bioreactor_ascii, render_ladder_ascii


def main() -> None:
    program = parse_structured_text(DEFAULT_PLANT_PLC_ST)
    ladder = parse_ladder_rungs(
        "BIOREACTOR_SAFETY",
        DEFAULT_PLANT_LADDER_SPECS,
        tag_types={"MASTER_RUN_PERMIT": "BOOL", "PRESSURE_BAR": "REAL"},
    )
    verifier = PLCStaticVerifier()
    report = verifier.verify(program, ladder=ladder, interlock_rules=DEFAULT_INTERLOCK_RULES)
    print(f"PLC Verification Passed: {report.passed} ({report.checked_combinations} trip combos checked)")
    print(render_ladder_ascii(ladder))

    runtime = PLCRuntime(program, ladder=ladder, scan_period_s=0.05)
    for _ in range(4):
        runtime.step_scan({"START_CMD": True, "PRESSURE_BAR": 1.25, "DO_MG_L": 3.4})
    print(f"Nominal Valve Cmd: {runtime.tags['VALVE_INLET_CMD']} | Agitator %: {runtime.tags['AGITATOR_PCT']:.1f}")

    # Inject overpressure trip
    runtime.step_scan({"START_CMD": True, "PRESSURE_BAR": 4.6, "DO_MG_L": 3.4})
    print(f"Overpressure Valve Cmd: {runtime.tags['VALVE_INLET_CMD']} | AlarmCode: {runtime.tags['ALARM_CODE']}")

    twin = BioreactorTwin()
    nominal = twin.simulate(duration_h=100.0, step_h=2.0, inject_do_drop=False)
    print(
        f"\nNominal Bioreactor | Final mAb={nominal.final_titer_g_l:.3f} g/L "
        f"| Min DO={nominal.min_do_mg_l:.2f} mg/L | ClosureErr={nominal.carbon_mass_closure_error_pct:.4f}%"
    )
    print(render_bioreactor_ascii(nominal))


if __name__ == "__main__":
    main()
