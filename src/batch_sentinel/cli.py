"""Command-line interface for BatchSentinel pharmaceutical manufacturing & PLC suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import PlantScenarioConfig, StabilityConfig
from .gmp.batch_record import build_merkle_proof, verify_merkle_proof
from .gmp.stability import StabilityModel
from .pat.multivariate import MultivariatePATMonitor
from .plant import (
    DEFAULT_INTERLOCK_RULES,
    DEFAULT_PLANT_LADDER_SPECS,
    DEFAULT_PLANT_PLC_ST,
    BatchSentinelPlant,
)
from .plc.parser import parse_ladder_rungs, parse_structured_text
from .plc.runtime import PLCRuntime
from .plc.verifier import PLCStaticVerifier
from .visualizer import (
    render_ascii_series,
    render_bioreactor_ascii,
    render_ladder_ascii,
    render_pat_control_chart_ascii,
)


def _cmd_simulate(args: argparse.Namespace) -> int:
    cfg = PlantScenarioConfig(
        batch_id=args.batch_id,
        bioreactor_hours=args.hours,
        crystallizer_minutes=args.cryst_min,
        pill_press_tablets=args.tablets,
        vial_count=args.vials,
        inject_fault=args.fault,
    )
    plant = BatchSentinelPlant(cfg)
    res = plant.run_campaign()

    summary: Dict[str, Any] = {
        "batch_id": res.batch_id,
        "disposition": res.batch_disposition,
        "disposition_reasons": res.disposition_reasons,
        "s88_state": res.s88_final_state.value,
        "sip_f0_min": round(res.sip_f0_minutes, 3),
        "plc_verified": res.plc_verification.passed,
        "plc_checked_combos": res.plc_verification.checked_combinations,
        "bioreactor": {
            "final_titer_g_l": round(res.bioreactor.final_titer_g_l, 4),
            "min_do_mg_l": round(res.bioreactor.min_do_mg_l, 3),
            "carbon_closure_err_pct": round(res.bioreactor.carbon_mass_closure_error_pct, 4),
        },
        "crystallizer": {
            "final_l43_um": round(res.crystallizer.final_l43_um, 2),
            "crystal_yield_kg": round(res.crystallizer.crystal_yield_kg, 3),
            "solute_closure_err_pct": round(res.crystallizer.solute_mass_closure_error_pct, 4),
        },
        "pill_press": {
            "mean_mass_mg": round(res.pill_press.mean_mass_mg, 2),
            "mass_rsd_pct": round(res.pill_press.mass_rsd_pct, 3),
            "peak_force_kn": round(res.pill_press.peak_force_kn, 2),
            "e_stop_triggered": res.pill_press.e_stop_triggered,
        },
        "aseptic_line": {
            "vials_processed": len(res.aseptic_line.vials),
            "min_dp_pa": round(res.aseptic_line.min_dp_pa, 2),
            "max_ik_error_m": round(res.aseptic_line.max_ik_error_m, 6),
            "airlock_halted": res.aseptic_line.airlock_breach_halted,
        },
        "pat_audit": {
            "t2_exceedance_pct": round(res.pat_audit.t2_exceedance_pct, 2),
            "spe_exceedance_pct": round(res.pat_audit.spe_exceedance_pct, 2),
            "fault_detected": res.pat_audit.fault_detected,
            "pls_predicted_titer_g_l": round(res.pls_predicted_titer_g_l, 4),
        },
        "ebr_valid": res.ebr_valid,
        "merkle_root": res.merkle_root,
        "mass_balanced": res.genealogy_report.balanced,
        "stability_potency_pct": round(res.stability.remaining_potency_pct, 3),
        "stability_mkt_c": round(res.stability.mean_kinetic_temp_c, 2),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"BatchSentinel Campaign Summary | Batch={res.batch_id} | Disposition={res.batch_disposition}")
    print(f"  ISA-88 State: {res.s88_final_state.value} | SIP F0: {res.sip_f0_minutes:.2f} min | PLC Checked: {res.plc_verification.checked_combinations} combos")
    print(f"  Bioreactor: Titer={res.bioreactor.final_titer_g_l:.3f} g/L | Min DO={res.bioreactor.min_do_mg_l:.2f} mg/L | ClosureErr={res.bioreactor.carbon_mass_closure_error_pct:.4f}%")
    print(f"  Crystallizer: L43={res.crystallizer.final_l43_um:.1f} um | Yield={res.crystallizer.crystal_yield_kg:.2f} kg | ClosureErr={res.crystallizer.solute_mass_closure_error_pct:.4f}%")
    print(f"  Pill Press: MeanMass={res.pill_press.mean_mass_mg:.2f} mg (RSD={res.pill_press.mass_rsd_pct:.2f}%) | PeakF={res.pill_press.peak_force_kn:.2f} kN | EStop={res.pill_press.e_stop_triggered}")
    print(f"  Aseptic Line: Vials={len(res.aseptic_line.vials)} | MinDP={res.aseptic_line.min_dp_pa:.2f} Pa | MaxIK={res.aseptic_line.max_ik_error_m:.6f} m")
    print(f"  PAT: T2_exc={res.pat_audit.t2_exceedance_pct:.1f}% | SPE_exc={res.pat_audit.spe_exceedance_pct:.1f}% | PLS_Titer={res.pls_predicted_titer_g_l:.3f} g/L")
    print(f"  21 CFR Part 11 EBR Valid={res.ebr_valid} | MerkleRoot={res.merkle_root[:20]}... | Stability={res.stability.remaining_potency_pct:.2f}% (MKT={res.stability.mean_kinetic_temp_c:.2f} C)")
    if res.disposition_reasons:
        print("  Disposition Reasons: " + "; ".join(res.disposition_reasons))
    print()
    print(render_bioreactor_ascii(res.bioreactor))
    return 0


def _cmd_verify_plc(args: argparse.Namespace) -> int:
    source = Path(args.file).read_text(encoding="utf-8") if args.file else DEFAULT_PLANT_PLC_ST
    prog = parse_structured_text(source)
    ladder = parse_ladder_rungs(
        "PHARMA_LADDER_INTERLOCKS",
        DEFAULT_PLANT_LADDER_SPECS,
        tag_types={"MASTER_RUN_PERMIT": "BOOL", "PRESSURE_BAR": "REAL"},
    )
    verifier = PLCStaticVerifier()
    rules = DEFAULT_INTERLOCK_RULES if not args.file else ()
    report = verifier.verify(prog, ladder=ladder if not args.file else None, interlock_rules=rules)

    rt = PLCRuntime(prog, ladder=ladder if not args.file else None)
    for _ in range(args.scans):
        rt.step_scan({"START_CMD": True, "PRESSURE_BAR": 1.2, "TEMP_C": 37.0, "DO_MG_L": 3.2})

    payload = {
        "program": report.program_name,
        "passed": report.passed,
        "checked_combinations": report.checked_combinations,
        "rules_checked": report.interlock_rules_checked,
        "scans_executed": rt.scan_count,
        "findings": [
            {"severity": f.severity, "code": f.code, "message": f.message, "line": f.line}
            for f in report.findings
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"PLC Verification: {report.program_name} | Passed={report.passed} | Combos={report.checked_combinations}")
        print(f"  OB1 Scans Executed: {rt.scan_count} | Last ValveCmd={rt.tags.get('VALVE_INLET_CMD')}")
        if not args.file:
            print()
            print(render_ladder_ascii(ladder))
    return 0 if report.passed else 1


def _cmd_pat_audit(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    x_train = rng.normal(loc=[3.6, 37.0, 7.15, 0.8], scale=[0.1, 0.15, 0.04, 0.2], size=(80, 4))
    x_test = rng.normal(loc=[3.6, 37.0, 7.15, 0.8], scale=[0.1, 0.15, 0.04, 0.2], size=(args.samples, 4))
    if args.inject_drift:
        x_test[args.samples // 2:, 0] -= 1.4  # DO drop excursion
        x_test[args.samples // 2:, 3] += 2.2  # Lactate surge

    mon = MultivariatePATMonitor(n_components=2, alpha=0.01).fit(
        x_train, feature_names=["DO_mg_L", "Temp_C", "pH", "Lactate_g_L"]
    )
    audit = mon.score(x_test)
    if args.json:
        print(
            json.dumps(
                {
                    "t2_ucl": round(audit.t2_ucl, 4),
                    "spe_ucl": round(audit.spe_ucl, 4),
                    "t2_exceedance_pct": round(audit.t2_exceedance_pct, 2),
                    "spe_exceedance_pct": round(audit.spe_exceedance_pct, 2),
                    "fault_detected": audit.fault_detected,
                    "top_fault_drivers": [(k, round(v, 3)) for k, v in audit.top_fault_drivers],
                },
                indent=2,
            )
        )
    else:
        print(render_pat_control_chart_ascii(audit))
        print(f"Fault Detected: {audit.fault_detected} | Top Drivers: {audit.top_fault_drivers}")
    return 0


def _cmd_trace_batch(args: argparse.Namespace) -> int:
    plant = BatchSentinelPlant(PlantScenarioConfig(batch_id=args.batch_id))
    res = plant.run_campaign()
    leaf_hashes = [e.entry_hash for e in res.ebr.entries]
    proof = build_merkle_proof(leaf_hashes, 0)
    proof_valid = verify_merkle_proof(leaf_hashes[0], proof, res.merkle_root)

    out = {
        "batch_id": res.batch_id,
        "ebr_entries": len(res.ebr.entries),
        "ebr_integrity_valid": res.ebr_valid,
        "merkle_root": res.merkle_root,
        "genesis_entry_merkle_proof_valid": proof_valid,
        "mass_reconciliation": {
            "input_kg": res.genealogy_report.total_input_kg,
            "finished_kg": res.genealogy_report.total_output_product_kg,
            "waste_sample_kg": res.genealogy_report.total_waste_and_sample_kg,
            "closure_error_pct": round(res.genealogy_report.closure_error_pct, 4),
            "balanced": res.genealogy_report.balanced,
        },
        "dscsa_units": len(res.serialization.unit_sgtins),
        "dscsa_cases": len(res.serialization.cases),
        "dscsa_pallets": len(res.serialization.pallets),
        "dscsa_complete": res.serialization.verify_completeness(),
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"21 CFR Part 11 Trace | Batch={res.batch_id} | Entries={len(res.ebr.entries)} | Valid={res.ebr_valid}")
        print(f"  Merkle Root: {res.merkle_root} | Leaf[0] Proof Valid={proof_valid}")
        print(f"  Mass Closure Error: {res.genealogy_report.closure_error_pct:.4f}% (Balanced={res.genealogy_report.balanced})")
        print(f"  DSCSA Units={len(res.serialization.unit_sgtins)} -> Cases={len(res.serialization.cases)} -> Pallets={len(res.serialization.pallets)}")
    return 0


def _cmd_stability(args: argparse.Namespace) -> int:
    cfg = StabilityConfig(storage_target_c=args.target_c, release_spec_pct=args.spec_pct)
    model = StabilityModel(cfg)
    temp_log = [args.target_c] * 48 + ([args.excursion_c] * args.excursion_hours if args.excursion_hours > 0 else []) + [args.target_c] * 48
    ass = model.evaluate_excursion(temp_log, dt_days=1.0 / 24.0, planned_shelf_days=args.shelf_days)

    out = {
        "mean_kinetic_temp_c": round(ass.mean_kinetic_temp_c, 3),
        "remaining_potency_pct": round(ass.remaining_potency_pct, 4),
        "predicted_t90_days": round(ass.predicted_t90_days, 2),
        "t90_lower_95_days": round(ass.t90_lower_95_days, 2),
        "within_release_spec": ass.within_release_spec,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"ICH Q1A/Q1E Stability | MKT={ass.mean_kinetic_temp_c:.2f} C | Potency={ass.remaining_potency_pct:.3f}%")
        print(f"  t90={ass.predicted_t90_days:.1f} days (95% LCL={ass.t90_lower_95_days:.1f} days) | WithinSpec={ass.within_release_spec}")
        print(render_ascii_series("Cold-Chain Temperature Profile (degC)", [("Temp_C", temp_log, "*")], width=54, height=8, y_label="C"))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    del args
    print("=== BatchSentinel Integrated GMP & PLC Demo ===")
    ns_nom = argparse.Namespace(
        batch_id="BS-DEMO-NOMINAL",
        hours=96.0,
        cryst_min=150.0,
        tablets=40,
        vials=16,
        fault="none",
        json=False,
    )
    _cmd_simulate(ns_nom)
    print("\n--- Fault Scenario: Rotary Press Overload + PLC Interlock ---")
    ns_fault = argparse.Namespace(
        batch_id="BS-DEMO-OVERLOAD",
        hours=48.0,
        cryst_min=90.0,
        tablets=30,
        vials=12,
        fault="press_overload",
        json=False,
    )
    _cmd_simulate(ns_fault)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch-sentinel",
        description="Automated pharmaceutical manufacturing, IEC 61131-3 PLC verification, PAT MSPC, and GMP batch traceability simulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="Run full integrated GMP plant simulation campaign.")
    p_sim.add_argument("--batch-id", default="BS-MAB-2026-001")
    p_sim.add_argument("--hours", type=float, default=96.0)
    p_sim.add_argument("--cryst-min", type=float, default=150.0)
    p_sim.add_argument("--tablets", type=int, default=48)
    p_sim.add_argument("--vials", type=int, default=18)
    p_sim.add_argument(
        "--fault",
        choices=["none", "do_drop", "press_overload", "door_breach", "cold_excursion"],
        default="none",
    )
    p_sim.add_argument("--json", action="store_true")
    p_sim.set_defaults(func=_cmd_simulate)

    p_plc = sub.add_parser("verify-plc", help="Parse and formally verify IEC 61131-3 PLC interlocks.")
    p_plc.add_argument("--file", default=None, help="Optional Structured Text (.st) file path.")
    p_plc.add_argument("--scans", type=int, default=10)
    p_plc.add_argument("--json", action="store_true")
    p_plc.set_defaults(func=_cmd_verify_plc)

    p_pat = sub.add_parser("pat-audit", help="Run PAT PCA Hotelling T^2 & SPE multivariate audit.")
    p_pat.add_argument("--samples", type=int, default=40)
    p_pat.add_argument("--seed", type=int, default=11)
    p_pat.add_argument("--inject-drift", action="store_true")
    p_pat.add_argument("--json", action="store_true")
    p_pat.set_defaults(func=_cmd_pat_audit)

    p_tr = sub.add_parser("trace-batch", help="Verify 21 CFR Part 11 EBR Merkle proof and DSCSA genealogy.")
    p_tr.add_argument("--batch-id", default="BS-TRACE-2026-01")
    p_tr.add_argument("--json", action="store_true")
    p_tr.set_defaults(func=_cmd_trace_batch)

    p_st = sub.add_parser("stability", help="Evaluate ICH Q1A/Q1E Arrhenius stability and MKT excursion.")
    p_st.add_argument("--target-c", type=float, default=5.0)
    p_st.add_argument("--excursion-c", type=float, default=28.0)
    p_st.add_argument("--excursion-hours", type=int, default=12)
    p_st.add_argument("--shelf-days", type=float, default=365.0)
    p_st.add_argument("--spec-pct", type=float, default=90.0)
    p_st.add_argument("--json", action="store_true")
    p_st.set_defaults(func=_cmd_stability)

    p_demo = sub.add_parser("demo", help="Run nominal and fault-injection GMP campaign showcase.")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
