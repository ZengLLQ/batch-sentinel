"""Tests for 21 CFR Part 11 EBR, Merkle proofs, lot mass balance, DSCSA, and ICH stability."""

from __future__ import annotations

import pytest

from batch_sentinel.config import StabilityConfig
from batch_sentinel.gmp import (
    ElectronicBatchRecord,
    LotGenealogyDAG,
    MaterialLot,
    SerializationHierarchy,
    StabilityModel,
    build_merkle_proof,
    compute_gs1_check_digit,
    validate_gtin14,
    verify_merkle_proof,
)


def test_21_cfr_part_11_ebr_hash_chain_tamper_detection_and_merkle_proofs() -> None:
    ebr = ElectronicBatchRecord("B-001", "mAb-101")
    for idx in range(5):
        ebr.append_event(
            f"2026-11-17T0{idx}:00:00Z",
            "OP-10",
            f"STEP_{idx}",
            "UNIT-1",
            {"val": idx * 10},
            sign_role="QA" if idx == 4 else None,
            sign_meaning="Release" if idx == 4 else None,
        )
    ok, msg = ebr.verify_integrity()
    assert ok and msg is None

    root = ebr.merkle_root()
    leaves = [e.entry_hash for e in ebr.entries]
    for i in range(len(leaves)):
        proof = build_merkle_proof(leaves, i)
        assert verify_merkle_proof(leaves[i], proof, root)

    # Tamper payload in entry #2 -> integrity check MUST fail
    ebr.entries[2].payload["val"] = 9999
    tampered_ok, tampered_msg = ebr.verify_integrity()
    assert not tampered_ok
    assert tampered_msg is not None and "Tampered" in tampered_msg


def test_lot_genealogy_mass_reconciliation_and_dscsa_check_digits() -> None:
    dag = LotGenealogyDAG()
    dag.register_lot(MaterialLot("RAW-1", "API", "RAW", 100.0))
    dag.register_lot(MaterialLot("FIN-1", "Drug Product", "FINISHED", 94.0))
    dag.register_lot(MaterialLot("WST-1", "Purge", "WASTE", 5.2))
    dag.link("RAW-1", "FIN-1", "Fill", 94.0)
    dag.link("RAW-1", "WST-1", "Flush", 5.2)
    assert "RAW-1" in dag.ancestors("FIN-1")
    rep = dag.reconcile_mass_balance(holdup_kg=0.8)
    assert rep.balanced
    assert rep.closure_error_pct == pytest.approx(0.0, abs=1e-6)

    gtin13 = "0031234567890"
    gtin14 = gtin13 + compute_gs1_check_digit(gtin13)
    assert validate_gtin14(gtin14)
    assert not validate_gtin14(gtin13 + str((int(gtin14[-1]) + 1) % 10))

    ser = SerializationHierarchy.generate(gtin14, "B-001", unit_count=25, units_per_case=10, cases_per_pallet=2)
    assert ser.verify_completeness()
    assert len(ser.cases) == 3
    assert len(ser.pallets) == 2


def test_ich_arrhenius_stability_mkt_and_t90_lcl() -> None:
    model = StabilityModel(StabilityConfig(storage_target_c=5.0, release_spec_pct=90.0))
    # Constant temperature -> MKT equals exact temperature
    assert model.mean_kinetic_temperature_c([5.0, 5.0, 5.0]) == pytest.approx(5.0, abs=1e-6)

    # Excursion spikes MKT nonlinearly above arithmetic mean
    logs = [5.0] * 48 + [32.0] * 24 + [5.0] * 48
    mkt = model.mean_kinetic_temperature_c(logs)
    assert mkt > sum(logs) / len(logs)

    t90_pt, t90_lcl = model.estimate_t90_ich_q1e(
        [0.0, 90.0, 180.0, 270.0, 360.0],
        [100.0, 98.1, 96.4, 94.7, 92.8],
    )
    assert t90_lcl <= t90_pt
    assert t90_pt > 360.0
