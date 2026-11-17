"""Example 03: 6-DOF Aseptic Fill-Finish Robotics, 21 CFR Part 11 EBR & DSCSA."""

from __future__ import annotations

from batch_sentinel.gmp import (
    ElectronicBatchRecord,
    LotGenealogyDAG,
    MaterialLot,
    SerializationHierarchy,
    build_merkle_proof,
    verify_merkle_proof,
)
from batch_sentinel.twins import AsepticFillLineTwin


def main() -> None:
    line = AsepticFillLineTwin()
    res = line.simulate_batch(vial_count=24)
    print(
        f"Aseptic Fill Line: Vials={len(res.vials)} | MeanFill={res.mean_fill_ml:.3f} mL "
        f"| MinDP={res.min_dp_pa:.2f} Pa | MaxIK={res.max_ik_error_m:.6f} m"
    )

    breach = line.simulate_batch(vial_count=24, door_breach_at_vial=10)
    print(
        f"Door Breach Test: Halted={breach.airlock_breach_halted} after {len(breach.vials)} vials "
        f"(MinDP={breach.min_dp_pa:.2f} Pa < 15.0 Pa)"
    )

    ebr = ElectronicBatchRecord(batch_id="BS-ASEPTIC-003", product_name="Sterile Vial 10 mL")
    ebr.append_event("2026-11-17T10:00:00Z", "OP-101", "LINE_CLEARANCE", "ISOLATOR-A", {"passed": True})
    ebr.append_event("2026-11-17T10:15:00Z", "OP-101", "FILTER_INTEGRITY", "ISOLATOR-A", {"bubble_pt_psi": 54.2})
    ebr.append_event(
        "2026-11-17T11:00:00Z",
        "QA-204",
        "BATCH_APPROVAL",
        "QA-DESK",
        {"vials": len(res.vials)},
        sign_role="QA_Reviewer",
        sign_meaning="Release_Approval",
    )
    ok, err = ebr.verify_integrity()
    root = ebr.merkle_root()
    proof = build_merkle_proof([e.entry_hash for e in ebr.entries], 1)
    proof_ok = verify_merkle_proof(ebr.entries[1].entry_hash, proof, root)
    print(f"21 CFR Part 11 EBR Valid={ok} ({err}) | MerkleRoot={root[:24]}... | ProofValid={proof_ok}")

    dag = LotGenealogyDAG()
    dag.register_lot(MaterialLot("RAW-BULK", "Sterile Bulk Solution", "RAW", 50.0))
    dag.register_lot(MaterialLot("FIN-VIALS", "Filled Vials", "FINISHED", 47.2))
    dag.register_lot(MaterialLot("WST-PURGE", "Line Purge", "WASTE", 2.4))
    dag.link("RAW-BULK", "FIN-VIALS", "AsepticFilling", 47.2)
    dag.link("RAW-BULK", "WST-PURGE", "LineFlush", 2.4)
    recon = dag.reconcile_mass_balance(holdup_kg=0.4)
    print(f"Mass Reconciliation: Balanced={recon.balanced} | ClosureErr={recon.closure_error_pct:.4f}%")

    dscsa = SerializationHierarchy.generate("00312345678906", "BS-ASEPTIC-003", unit_count=24)
    print(f"DSCSA Serialization Complete={dscsa.verify_completeness()} | Cases={len(dscsa.cases)} | Pallets={len(dscsa.pallets)}")


if __name__ == "__main__":
    main()
