"""Integrated GMP pharmaceutical plant campaign orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PlantScenarioConfig
from .gmp.batch_record import (
    ElectronicBatchRecord,
    LotGenealogyDAG,
    MassReconciliationReport,
    MaterialLot,
    SerializationHierarchy,
)
from .gmp.stability import StabilityAssessment, StabilityModel
from .isa88.recipe import (
    ISA88BatchExecutive,
    PhaseExecutionRecord,
    S88Operation,
    S88Phase,
    S88Procedure,
    S88State,
    S88UnitProcedure,
    compute_sip_f0_lethality,
)
from .pat.multivariate import MultivariatePATMonitor, PATBatchAudit, PLSQualityPredictor
from .plc.parser import parse_ladder_rungs, parse_structured_text
from .plc.runtime import PLCRuntime
from .plc.verifier import PLCStaticVerifier, SafetyInterlockRule, VerificationReport
from .twins.aseptic_line import AsepticFillLineTwin, AsepticLineResult
from .twins.bioreactor import BioreactorTrajectory, BioreactorTwin
from .twins.crystallizer import CrystallizerResult, CrystallizerTwin
from .twins.pill_press import PillPressRunResult, PillPressTwin


DEFAULT_PLANT_PLC_ST = """
PROGRAM PHARMA_SKID_SAFETY
VAR_INPUT
    E_STOP_TRIP : BOOL := FALSE;
    PRESSURE_HIHI : BOOL := FALSE;
    DOOR_OPEN : BOOL := FALSE;
    OVERLOAD_TRIP : BOOL := FALSE;
    START_CMD : BOOL := FALSE;
    TEMP_C : REAL := 37.0;
    PRESSURE_BAR : REAL := 1.2;
    DO_MG_L : REAL := 3.6;
    DO_SP_MG_L : REAL := 3.6;
END_VAR
VAR_OUTPUT
    INTERLOCK_SAFE : BOOL := FALSE;
    VALVE_INLET_CMD : BOOL := FALSE;
    PUNCH_ENABLE : BOOL := FALSE;
    ISOLATOR_RUN : BOOL := FALSE;
    AGITATOR_PCT : REAL := 0.0;
    ALARM_CODE : INT := 0;
END_VAR
VAR
    TON_PURGE : TON;
    DO_PID : PID_FB;
    PURGE_READY : BOOL := FALSE;
END_VAR

(* Evaluate primary hardwired-equivalent safety interlock *)
IF E_STOP_TRIP OR PRESSURE_HIHI OR DOOR_OPEN OR OVERLOAD_TRIP OR (PRESSURE_BAR > 4.0) OR (TEMP_C > 135.0) THEN
    INTERLOCK_SAFE := FALSE;
    VALVE_INLET_CMD := FALSE;
    PUNCH_ENABLE := FALSE;
    ISOLATOR_RUN := FALSE;
    AGITATOR_PCT := 0.0;
    IF E_STOP_TRIP THEN
        ALARM_CODE := 101;
    ELSIF OVERLOAD_TRIP THEN
        ALARM_CODE := 102;
    ELSIF DOOR_OPEN THEN
        ALARM_CODE := 103;
    ELSE
        ALARM_CODE := 104;
    END_IF;
ELSE
    INTERLOCK_SAFE := TRUE;
    ALARM_CODE := 0;
    TON_PURGE(IN := START_CMD, PT := T#100ms);
    PURGE_READY := TON_PURGE.Q;
    IF START_CMD AND PURGE_READY THEN
        VALVE_INLET_CMD := TRUE;
        PUNCH_ENABLE := TRUE;
        ISOLATOR_RUN := TRUE;
        DO_PID(PV := DO_MG_L, SP := DO_SP_MG_L, KP := 12.0, KI := 2.5, KD := 0.4, OUT_MIN := 15.0, OUT_MAX := 100.0);
        AGITATOR_PCT := DO_PID.OUT;
    ELSE
        VALVE_INLET_CMD := FALSE;
        PUNCH_ENABLE := FALSE;
        ISOLATOR_RUN := FALSE;
        AGITATOR_PCT := 0.0;
    END_IF;
END_IF;
END_PROGRAM
"""


DEFAULT_PLANT_LADDER_SPECS: List[Dict[str, object]] = [
    {
        "rung_id": "R001",
        "comment": "Master safety relay latch drops on E-stop, overload, door open, or overpressure",
        "branches": [
            [
                {"type": "NO", "tag": "START_CMD"},
                {"type": "NC", "tag": "E_STOP_TRIP"},
                {"type": "NC", "tag": "PRESSURE_HIHI"},
                {"type": "NC", "tag": "DOOR_OPEN"},
                {"type": "NC", "tag": "OVERLOAD_TRIP"},
                {"type": "CMP", "tag": "PRESSURE_BAR", "op": "<=", "threshold": 4.0},
            ]
        ],
        "coils": [{"tag": "MASTER_RUN_PERMIT", "mode": "COIL"}],
    },
    {
        "rung_id": "R002",
        "comment": "De-energize inlet valve when master safety relay drops",
        "branches": [
            [
                {"type": "NO", "tag": "MASTER_RUN_PERMIT"},
            ]
        ],
        "coils": [{"tag": "VALVE_INLET_CMD", "mode": "COIL"}],
    },
]


DEFAULT_INTERLOCK_RULES: Tuple[SafetyInterlockRule, ...] = (
    SafetyInterlockRule(
        name="INLET_VALVE_TRIP_SAFETY",
        trip_tags=("E_STOP_TRIP", "PRESSURE_HIHI", "DOOR_OPEN", "OVERLOAD_TRIP"),
        actuator_tag="VALVE_INLET_CMD",
        forbidden_actuator_value=True,
        analog_trips=(("PRESSURE_BAR", ">", 4.0),),
    ),
    SafetyInterlockRule(
        name="PUNCH_PRESS_OVERLOAD_SAFETY",
        trip_tags=("E_STOP_TRIP", "OVERLOAD_TRIP", "DOOR_OPEN"),
        actuator_tag="PUNCH_ENABLE",
        forbidden_actuator_value=True,
    ),
    SafetyInterlockRule(
        name="ISOLATOR_DOOR_BREACH_SAFETY",
        trip_tags=("E_STOP_TRIP", "DOOR_OPEN"),
        actuator_tag="ISOLATOR_RUN",
        forbidden_actuator_value=True,
    ),
)


@dataclass
class PlantCampaignResult:
    """Comprehensive output of an end-to-end GMP pharmaceutical manufacturing campaign."""

    batch_id: str
    product_name: str
    plc_verification: VerificationReport
    plc_scans_executed: int
    s88_final_state: S88State
    s88_phase_records: List[PhaseExecutionRecord]
    sip_f0_minutes: float
    bioreactor: BioreactorTrajectory
    crystallizer: CrystallizerResult
    pill_press: PillPressRunResult
    aseptic_line: AsepticLineResult
    pat_audit: PATBatchAudit
    pls_predicted_titer_g_l: float
    ebr: ElectronicBatchRecord
    ebr_valid: bool
    merkle_root: str
    genealogy_report: MassReconciliationReport
    serialization: SerializationHierarchy
    stability: StabilityAssessment
    batch_disposition: str  # "RELEASED" | "QUARANTINE_HOLD" | "REJECTED"
    disposition_reasons: List[str]


class BatchSentinelPlant:
    """End-to-end digital twin orchestrator for automated GMP pharma manufacturing."""

    def __init__(self, config: Optional[PlantScenarioConfig] = None) -> None:
        self.config = config or PlantScenarioConfig()
        self.plc_program = parse_structured_text(DEFAULT_PLANT_PLC_ST)
        self.ladder_program = parse_ladder_rungs(
            "PHARMA_LADDER_INTERLOCKS",
            DEFAULT_PLANT_LADDER_SPECS,
            tag_types={"MASTER_RUN_PERMIT": "BOOL", "PRESSURE_BAR": "REAL"},
        )
        self.verifier = PLCStaticVerifier(max_loop_iterations=self.config.plc.max_loop_iterations)

    def _build_golden_pat_models(
        self,
        bio_ref: BioreactorTrajectory,
        cryst_ref: CrystallizerResult,
        press_ref: PillPressRunResult,
        line_ref: AsepticLineResult,
    ) -> Tuple[MultivariatePATMonitor, PLSQualityPredictor]:
        """Calibrate PCA MSPC & PLS CQA soft-sensor from golden-batch nominal envelope."""
        rng = np.random.default_rng(2026)
        n_obs = max(10, min(len(bio_ref.states), len(cryst_ref.states), len(press_ref.tablets), len(line_ref.vials)))
        base_rows = []
        for k in range(n_obs):
            b_st = bio_ref.states[min(k, len(bio_ref.states) - 1)]
            c_st = cryst_ref.states[min(k, len(cryst_ref.states) - 1)]
            t_rec = press_ref.tablets[min(k, len(press_ref.tablets) - 1)]
            v_rec = line_ref.vials[min(k, len(line_ref.vials) - 1)]
            base_rows.append(
                [
                    b_st.dissolved_o2_mg_l,
                    b_st.temperature_c,
                    b_st.ph,
                    b_st.lactate_g_l,
                    c_st.supersaturation_ratio,
                    t_rec.peak_force_kn,
                    v_rec.cleanroom_dp_pa,
                ]
            )
        base_mat = np.asarray(base_rows, dtype=np.float64)
        # Synthesize 8 golden-batch replicates with realistic sensor & process variation
        replicates = []
        y_titer_parts = []
        noise_scale = np.array([0.22, 0.15, 0.04, 0.18, 0.015, 0.85, 0.90], dtype=np.float64)
        for _ in range(8):
            rep = base_mat + rng.normal(0.0, noise_scale, size=base_mat.shape)
            replicates.append(rep)
            y_titer_parts.append(
                2.8 + 0.25 * (rep[:, 0] - 3.6) - 0.30 * rep[:, 3] + rng.normal(0.0, 0.04, size=n_obs)
            )

        x_train = np.vstack(replicates)
        y_titer = np.concatenate(y_titer_parts)

        pat_mon = MultivariatePATMonitor(
            n_components=self.config.pat.n_components,
            alpha=self.config.pat.alpha,
        )
        pat_mon.fit(
            x_train,
            feature_names=["DO_mg_L", "Temp_C", "pH", "Lactate_g_L", "Supersat", "Force_kN", "Cleanroom_DP_Pa"],
        )
        pls = PLSQualityPredictor(n_components=3)
        pls.fit(x_train, y_titer)
        return pat_mon, pls

    def run_campaign(self) -> PlantCampaignResult:
        """Execute full verified PLC + ISA-88 + 4-twin + PAT + 21 CFR Part 11 GMP campaign."""
        cfg = self.config

        # 1. Static PLC Verification Gate
        plc_report = self.verifier.verify(
            self.plc_program,
            ladder=self.ladder_program,
            interlock_rules=DEFAULT_INTERLOCK_RULES,
        )
        if not plc_report.passed:
            raise RuntimeError(f"PLC static safety verification failed: {plc_report.findings}")

        plc_runtime = PLCRuntime(
            program=self.plc_program,
            ladder=self.ladder_program,
            scan_period_s=cfg.plc.scan_period_s,
            watchdog_timeout_s=cfg.plc.watchdog_timeout_s,
            jitter_std_s=cfg.plc.jitter_std_s,
            seed=cfg.plc.seed,
        )

        # Warm-up PLC purge timer (3 scans >= 100ms)
        for _ in range(4):
            plc_runtime.step_scan({"START_CMD": True, "PRESSURE_BAR": 1.2, "TEMP_C": 37.0})

        # 2. Initialize 21 CFR Part 11 EBR & Twins
        ebr = ElectronicBatchRecord(batch_id=cfg.batch_id, product_name=cfg.product_name)
        ebr.append_event(
            "2026-11-17T08:00:00Z",
            cfg.operator_ids[0],
            "PLC_INTERLOCK_VERIFIED",
            "PLC-MAIN-01",
            {
                "checked_combinations": plc_report.checked_combinations,
                "rules": plc_report.interlock_rules_checked,
            },
            sign_role="Automation_QA",
            sign_meaning="Verification_Approval",
        )

        # SIP Sterilization thermal profile (30 min ramp + hold at 122.5 C + cool)
        sip_temps = [85.0, 108.0, 118.5, 122.4, 122.8, 122.6, 122.5, 122.3, 116.0, 96.0]
        sip_f0 = compute_sip_f0_lethality(sip_temps, dt_min=3.0)

        bio_twin = BioreactorTwin(params=cfg.bioreactor)
        cryst_twin = CrystallizerTwin(params=cfg.crystallizer)
        press_twin = PillPressTwin(params=cfg.pill_press)
        line_twin = AsepticFillLineTwin(params=cfg.fill_finish)

        # Holders populated by ISA-88 phases
        holder: Dict[str, Any] = {}

        def phase_sip() -> Dict[str, float]:
            ebr.append_event(
                "2026-11-17T08:30:00Z",
                cfg.operator_ids[1],
                "SIP_COMPLETE",
                "BIOREACTOR-2000L",
                {"f0_min": round(sip_f0, 3)},
            )
            return {"f0_minutes": sip_f0}

        def phase_bioreact() -> Dict[str, float]:
            traj = bio_twin.simulate(
                duration_h=cfg.bioreactor_hours,
                step_h=2.0,
                inject_do_drop=(cfg.inject_fault == "do_drop"),
            )
            holder["bioreactor"] = traj
            # Mirror last states into PLC OB1 scan
            for st in traj.states[::5]:
                plc_runtime.step_scan(
                    {
                        "START_CMD": True,
                        "DO_MG_L": st.dissolved_o2_mg_l,
                        "TEMP_C": st.temperature_c,
                        "PRESSURE_BAR": 1.35,
                    }
                )
            ebr.append_event(
                "2026-11-17T12:00:00Z",
                cfg.operator_ids[1],
                "BIOREACTOR_HARVEST",
                "BIOREACTOR-2000L",
                {
                    "final_titer_g_l": round(traj.final_titer_g_l, 4),
                    "min_do_mg_l": round(traj.min_do_mg_l, 3),
                    "closure_err_pct": round(traj.carbon_mass_closure_error_pct, 4),
                },
            )
            return {
                "final_titer_g_l": traj.final_titer_g_l,
                "min_do_mg_l": traj.min_do_mg_l,
            }

        def phase_crystallize() -> Dict[str, float]:
            res = cryst_twin.simulate_cooling_batch(
                duration_min=cfg.crystallizer_minutes,
                step_min=3.0,
            )
            holder["crystallizer"] = res
            ebr.append_event(
                "2026-11-17T15:00:00Z",
                cfg.operator_ids[1],
                "CRYSTALLIZATION_COMPLETE",
                "CRYSTALLIZER-250KG",
                {
                    "l43_um": round(res.final_l43_um, 2),
                    "yield_kg": round(res.crystal_yield_kg, 3),
                    "closure_err_pct": round(res.solute_mass_closure_error_pct, 4),
                },
            )
            return {"final_l43_um": res.final_l43_um, "crystal_yield_kg": res.crystal_yield_kg}

        def phase_tableting() -> Dict[str, float]:
            overload_idx = cfg.pill_press_tablets // 2 if cfg.inject_fault == "press_overload" else None
            res = press_twin.simulate_run(
                n_tablets=cfg.pill_press_tablets,
                inject_overload_at=overload_idx,
            )
            holder["pill_press"] = res
            plc_runtime.step_scan(
                {
                    "START_CMD": True,
                    "OVERLOAD_TRIP": res.e_stop_triggered,
                    "PRESSURE_BAR": 1.2,
                }
            )
            ebr.append_event(
                "2026-11-17T16:00:00Z",
                cfg.operator_ids[1],
                "TABLETING_COMPLETE",
                "ROTARY-PRESS-36",
                {
                    "mean_mass_mg": round(res.mean_mass_mg, 2),
                    "peak_force_kn": round(res.peak_force_kn, 2),
                    "e_stop_triggered": res.e_stop_triggered,
                },
            )
            return {"acceptance_rate_pct": res.acceptance_rate_pct, "peak_force_kn": res.peak_force_kn}

        def phase_aseptic_fill() -> Dict[str, float]:
            breach_idx = cfg.vial_count // 2 if cfg.inject_fault == "door_breach" else None
            res = line_twin.simulate_batch(
                vial_count=cfg.vial_count,
                door_breach_at_vial=breach_idx,
            )
            holder["aseptic_line"] = res
            plc_runtime.step_scan(
                {
                    "START_CMD": True,
                    "DOOR_OPEN": res.airlock_breach_halted,
                    "PRESSURE_BAR": 1.1,
                }
            )
            ebr.append_event(
                "2026-11-17T17:00:00Z",
                cfg.operator_ids[1],
                "ASEPTIC_FILL_COMPLETE",
                "ISOLATOR-LINE-A",
                {
                    "vials_processed": len(res.vials),
                    "min_dp_pa": round(res.min_dp_pa, 2),
                    "airlock_halted": res.airlock_breach_halted,
                },
            )
            return {"min_dp_pa": res.min_dp_pa, "acceptance_rate_pct": res.acceptance_rate_pct}

        master_proc = S88Procedure(
            name="GMP_MAB_MANUFACTURE",
            unit_procedures=[
                S88UnitProcedure(
                    name="UP_UPSTREAM_BIOREACTION",
                    unit_id="BIOREACTOR-2000L",
                    operations=[
                        S88Operation("OP_SIP", [S88Phase("PH_STEAM_STERILIZE", "BIOREACTOR-2000L", phase_sip)]),
                        S88Operation("OP_FERMENT", [S88Phase("PH_FED_BATCH_PERFUSION", "BIOREACTOR-2000L", phase_bioreact)]),
                    ],
                ),
                S88UnitProcedure(
                    name="UP_DOWNSTREAM_CRYSTALLIZATION",
                    unit_id="CRYSTALLIZER-250KG",
                    operations=[
                        S88Operation("OP_COOL_CRYST", [S88Phase("PH_CONTROLLED_COOLING", "CRYSTALLIZER-250KG", phase_crystallize)]),
                    ],
                ),
                S88UnitProcedure(
                    name="UP_ORAL_SOLID_TABLETING",
                    unit_id="ROTARY-PRESS-36",
                    operations=[
                        S88Operation("OP_COMPACTION", [S88Phase("PH_ROTARY_COMPRESSION", "ROTARY-PRESS-36", phase_tableting)]),
                    ],
                ),
                S88UnitProcedure(
                    name="UP_ASEPTIC_FILL_FINISH",
                    unit_id="ISOLATOR-LINE-A",
                    operations=[
                        S88Operation("OP_FILLING", [S88Phase("PH_ROBOTIC_VIAL_FILL", "ISOLATOR-LINE-A", phase_aseptic_fill)]),
                    ],
                ),
            ],
        )

        s88_exec = ISA88BatchExecutive()
        s88_state, s88_records, _ = s88_exec.run_procedure(master_proc, cfg.batch_id)

        bio_res: BioreactorTrajectory = holder["bioreactor"]
        cryst_res: CrystallizerResult = holder["crystallizer"]
        press_res: PillPressRunResult = holder["pill_press"]
        line_res: AsepticLineResult = holder["aseptic_line"]

        # 3. PAT Multivariate Statistical Process Control & PLS CQA Prediction
        # Calibrate golden reference on nominal twins when fault is injected so fault is isolated cleanly
        if cfg.inject_fault == "none":
            pat_mon, pls_model = self._build_golden_pat_models(bio_res, cryst_res, press_res, line_res)
        else:
            ref_bio = bio_twin.simulate(duration_h=cfg.bioreactor_hours, step_h=2.0, inject_do_drop=False)
            ref_cryst = cryst_res
            ref_press = press_twin.simulate_run(n_tablets=cfg.pill_press_tablets, inject_overload_at=None)
            ref_line = line_twin.simulate_batch(vial_count=cfg.vial_count, door_breach_at_vial=None)
            pat_mon, pls_model = self._build_golden_pat_models(ref_bio, ref_cryst, ref_press, ref_line)
        n_obs = min(len(bio_res.states), len(cryst_res.states), len(press_res.tablets), len(line_res.vials))
        n_obs = max(n_obs, 10)
        pat_rows = []
        for k in range(n_obs):
            b_st = bio_res.states[min(k, len(bio_res.states) - 1)]
            c_st = cryst_res.states[min(k, len(cryst_res.states) - 1)]
            t_rec = press_res.tablets[min(k, len(press_res.tablets) - 1)]
            v_rec = line_res.vials[min(k, len(line_res.vials) - 1)]
            pat_rows.append(
                [
                    b_st.dissolved_o2_mg_l,
                    b_st.temperature_c,
                    b_st.ph,
                    b_st.lactate_g_l,
                    c_st.supersaturation_ratio,
                    t_rec.peak_force_kn,
                    v_rec.cleanroom_dp_pa,
                ]
            )
        x_batch = np.asarray(pat_rows, dtype=np.float64)
        pat_audit = pat_mon.score(x_batch)
        pls_titer = float(np.mean(pls_model.predict(x_batch)))

        # 4. Lot Genealogy DAG & Mass-Balance Reconciliation
        dag = LotGenealogyDAG()
        raw_api_kg = 120.0
        raw_excip_kg = 80.0
        finished_kg = 186.5
        waste_kg = 9.2
        sample_kg = 2.8
        holdup_kg = 1.5
        dag.register_lot(MaterialLot("LOT-RAW-API", "Active Biologic Bulk", "RAW", raw_api_kg))
        dag.register_lot(MaterialLot("LOT-RAW-EXC", "GMP Mannitol Excipient", "RAW", raw_excip_kg))
        dag.register_lot(MaterialLot("LOT-INT-CRYST", "Purified Crystal Cake", "INTERMEDIATE", 190.0))
        dag.register_lot(MaterialLot("LOT-FIN-BATCH", cfg.product_name, "FINISHED", finished_kg))
        dag.register_lot(MaterialLot("LOT-WASTE-01", "Mother Liquor Waste", "WASTE", waste_kg))
        dag.register_lot(MaterialLot("LOT-QC-SAMP", "Retain & PAT Samples", "SAMPLE", sample_kg))
        dag.link("LOT-RAW-API", "LOT-INT-CRYST", "Crystallization", 115.0)
        dag.link("LOT-RAW-EXC", "LOT-INT-CRYST", "Compounding", 75.0)
        dag.link("LOT-INT-CRYST", "LOT-FIN-BATCH", "FillFinish", finished_kg)
        dag.link("LOT-RAW-API", "LOT-WASTE-01", "Purge", 5.0)
        dag.link("LOT-RAW-EXC", "LOT-WASTE-01", "Rejects", 4.2)
        dag.link("LOT-INT-CRYST", "LOT-QC-SAMP", "QC_Sampling", sample_kg)
        genealogy_rep = dag.reconcile_mass_balance(holdup_kg=holdup_kg, tolerance_pct=1.0)

        # 5. DSCSA GS1 Serialization
        serialization = SerializationHierarchy.generate(
            gtin14=cfg.gtin14,
            batch_id=cfg.batch_id,
            unit_count=max(1, len(line_res.vials)),
            units_per_case=12,
            cases_per_pallet=4,
        )

        # 6. Cold-Chain Arrhenius Stability Evaluation
        if cfg.inject_fault == "cold_excursion":
            # 72 hours at 44 C severe thermal excursion during lane transit
            temp_log = [5.0] * 24 + [44.0] * 72 + [5.0] * 24
        else:
            temp_log = [5.0 + 0.15 * ((-1) ** i) for i in range(120)]
        stability = StabilityModel(cfg.stability).evaluate_excursion(temp_log, dt_days=1.0 / 24.0)

        # 7. QA / QP Batch Disposition & Final 21 CFR Part 11 Signature
        reasons: List[str] = []
        if sip_f0 < 15.0:
            reasons.append(f"SIP F0 lethality insufficient ({sip_f0:.2f} < 15.0 min)")
        if press_res.e_stop_triggered:
            reasons.append("Rotary press punch overload E-stop tripped")
        if line_res.airlock_breach_halted:
            reasons.append("ISO-5 cleanroom differential pressure breach halted filling")
        if pat_audit.fault_detected:
            reasons.append(
                f"PAT MSPC excursion (T2_exc={pat_audit.t2_exceedance_pct:.1f}%, SPE_exc={pat_audit.spe_exceedance_pct:.1f}%)"
            )
        if not genealogy_rep.balanced:
            reasons.append("Lot mass balance outside 1.0% tolerance")
        if not stability.within_release_spec:
            reasons.append(f"Stability shelf-life/potency out of spec (potency={stability.remaining_potency_pct:.2f}%)")

        if press_res.e_stop_triggered or line_res.airlock_breach_halted or not stability.within_release_spec:
            disposition = "REJECTED"
        elif reasons:
            disposition = "QUARANTINE_HOLD"
        else:
            disposition = "RELEASED"

        ebr.append_event(
            "2026-11-17T17:30:00Z",
            cfg.operator_ids[0],
            "QP_BATCH_DISPOSITION",
            "QA-RELEASE-DESK",
            {"disposition": disposition, "reasons": reasons, "merkle_preview": ebr.merkle_root()},
            reason_for_change="Final batch disposition review",
            sign_role="Qualified_Person",
            sign_meaning=disposition,
        )
        ebr_valid, _ = ebr.verify_integrity()

        return PlantCampaignResult(
            batch_id=cfg.batch_id,
            product_name=cfg.product_name,
            plc_verification=plc_report,
            plc_scans_executed=plc_runtime.scan_count,
            s88_final_state=s88_state,
            s88_phase_records=s88_records,
            sip_f0_minutes=sip_f0,
            bioreactor=bio_res,
            crystallizer=cryst_res,
            pill_press=press_res,
            aseptic_line=line_res,
            pat_audit=pat_audit,
            pls_predicted_titer_g_l=pls_titer,
            ebr=ebr,
            ebr_valid=ebr_valid,
            merkle_root=ebr.merkle_root(),
            genealogy_report=genealogy_rep,
            serialization=serialization,
            stability=stability,
            batch_disposition=disposition,
            disposition_reasons=reasons,
        )
