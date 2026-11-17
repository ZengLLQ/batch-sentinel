"""FDA 21 CFR Part 11 Electronic Batch Record, Merkle genealogy, and DSCSA serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


def _canonical_json(payload: Dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ElectronicSignature:
    """FDA 21 CFR Part 11 electronic signature binding signer, role, and approval meaning."""

    signer_id: str
    role: str
    meaning: str  # e.g. "Author", "QA_Review", "QP_Release"
    timestamp_utc: str
    signature_digest: str


@dataclass
class AuditEntry:
    """Single immutable ALCOA+ hash-chained audit trail record."""

    sequence: int
    timestamp_utc: str
    operator_id: str
    action: str
    unit_id: str
    payload: Dict[str, object]
    reason_for_change: str
    prev_hash: str
    entry_hash: str
    signature: Optional[ElectronicSignature] = None


@dataclass(frozen=True)
class MerkleProofStep:
    """Sibling hash and side ('L' or 'R') in a binary SHA-256 Merkle inclusion proof."""

    sibling_hash: str
    sibling_side: str  # "L" or "R"


def compute_merkle_root(leaf_hashes: Sequence[str]) -> str:
    """Compute binary SHA-256 Merkle root over hex leaf hashes (RFC 6962-style domain tags)."""
    if not leaf_hashes:
        return _sha256_hex(b"EMPTY_MERKLE_TREE")
    level = [_sha256_hex(b"\x00" + bytes.fromhex(h)) for h in leaf_hashes]
    while len(level) > 1:
        next_level: List[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            combined = b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)
            next_level.append(_sha256_hex(combined))
        level = next_level
    return level[0]


def build_merkle_proof(leaf_hashes: Sequence[str], leaf_index: int) -> List[MerkleProofStep]:
    """Generate a Merkle inclusion proof for ``leaf_hashes[leaf_index]``."""
    if not (0 <= leaf_index < len(leaf_hashes)):
        raise IndexError(f"leaf_index {leaf_index} out of range [0, {len(leaf_hashes)})")
    level = [_sha256_hex(b"\x00" + bytes.fromhex(h)) for h in leaf_hashes]
    idx = leaf_index
    proof: List[MerkleProofStep] = []
    while len(level) > 1:
        is_right = (idx % 2) == 1
        pair_idx = idx - 1 if is_right else (idx + 1 if idx + 1 < len(level) else idx)
        proof.append(
            MerkleProofStep(
                sibling_hash=level[pair_idx],
                sibling_side="L" if is_right else "R",
            )
        )
        next_level: List[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            next_level.append(_sha256_hex(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)))
        level = next_level
        idx //= 2
    return proof


def verify_merkle_proof(leaf_hash: str, proof: Sequence[MerkleProofStep], expected_root: str) -> bool:
    """Verify that ``leaf_hash`` commits to ``expected_root`` under ``proof``."""
    cur = _sha256_hex(b"\x00" + bytes.fromhex(leaf_hash))
    for step in proof:
        if step.sibling_side == "L":
            cur = _sha256_hex(b"\x01" + bytes.fromhex(step.sibling_hash) + bytes.fromhex(cur))
        elif step.sibling_side == "R":
            cur = _sha256_hex(b"\x01" + bytes.fromhex(cur) + bytes.fromhex(step.sibling_hash))
        else:
            return False
    return cur == expected_root


class ElectronicBatchRecord:
    """Tamper-evident 21 CFR Part 11 Electronic Batch Record with Merkle commitment."""

    GENESIS_HASH = "0" * 64

    def __init__(self, batch_id: str, product_name: str) -> None:
        self.batch_id = batch_id
        self.product_name = product_name
        self.entries: List[AuditEntry] = []

    @staticmethod
    def _compute_entry_hash(
        sequence: int,
        timestamp_utc: str,
        operator_id: str,
        action: str,
        unit_id: str,
        payload: Dict[str, object],
        reason_for_change: str,
        prev_hash: str,
    ) -> str:
        body = {
            "sequence": sequence,
            "timestamp_utc": timestamp_utc,
            "operator_id": operator_id,
            "action": action,
            "unit_id": unit_id,
            "payload": payload,
            "reason_for_change": reason_for_change,
            "prev_hash": prev_hash,
        }
        return _sha256_hex(_canonical_json(body))

    def append_event(
        self,
        timestamp_utc: str,
        operator_id: str,
        action: str,
        unit_id: str,
        payload: Dict[str, object],
        reason_for_change: str = "Standard batch execution",
        sign_role: Optional[str] = None,
        sign_meaning: Optional[str] = None,
    ) -> AuditEntry:
        """Append a new hash-chained ALCOA+ event, optionally attaching an e-signature."""
        seq = len(self.entries)
        prev_hash = self.entries[-1].entry_hash if self.entries else self.GENESIS_HASH
        entry_hash = self._compute_entry_hash(
            seq, timestamp_utc, operator_id, action, unit_id, payload, reason_for_change, prev_hash
        )
        sig: Optional[ElectronicSignature] = None
        if sign_role and sign_meaning:
            sig_payload = {
                "entry_hash": entry_hash,
                "signer_id": operator_id,
                "role": sign_role,
                "meaning": sign_meaning,
                "timestamp_utc": timestamp_utc,
            }
            sig = ElectronicSignature(
                signer_id=operator_id,
                role=sign_role,
                meaning=sign_meaning,
                timestamp_utc=timestamp_utc,
                signature_digest=_sha256_hex(_canonical_json(sig_payload)),
            )
        entry = AuditEntry(
            sequence=seq,
            timestamp_utc=timestamp_utc,
            operator_id=operator_id,
            action=action,
            unit_id=unit_id,
            payload=dict(payload),
            reason_for_change=reason_for_change,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            signature=sig,
        )
        self.entries.append(entry)
        return entry

    def merkle_root(self) -> str:
        """Return the SHA-256 Merkle root across all EBR audit entries."""
        return compute_merkle_root([e.entry_hash for e in self.entries])

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify sequential hash chaining and all attached e-signatures."""
        expected_prev = self.GENESIS_HASH
        for idx, entry in enumerate(self.entries):
            if entry.sequence != idx:
                return False, f"Sequence mismatch at index {idx}: got {entry.sequence}"
            if entry.prev_hash != expected_prev:
                return False, f"Broken prev_hash link at entry {idx}"
            recomputed = self._compute_entry_hash(
                entry.sequence,
                entry.timestamp_utc,
                entry.operator_id,
                entry.action,
                entry.unit_id,
                entry.payload,
                entry.reason_for_change,
                entry.prev_hash,
            )
            if recomputed != entry.entry_hash:
                return False, f"Tampered payload or metadata detected at entry {idx}"
            if entry.signature is not None:
                sig_payload = {
                    "entry_hash": entry.entry_hash,
                    "signer_id": entry.signature.signer_id,
                    "role": entry.signature.role,
                    "meaning": entry.signature.meaning,
                    "timestamp_utc": entry.signature.timestamp_utc,
                }
                if _sha256_hex(_canonical_json(sig_payload)) != entry.signature.signature_digest:
                    return False, f"Invalid electronic signature at entry {idx}"
            expected_prev = entry.entry_hash
        return True, None


# --- Lot Genealogy & Mass-Balance Reconciliation ---


@dataclass(frozen=True)
class MaterialLot:
    """Raw material, intermediate, or finished pharmaceutical lot."""

    lot_id: str
    material_name: str
    role: str  # "RAW" | "INTERMEDIATE" | "FINISHED" | "WASTE" | "SAMPLE"
    mass_kg: float


@dataclass(frozen=True)
class GenealogyEdge:
    """Directed transformation edge linking input lot to output lot in a unit step."""

    source_lot_id: str
    target_lot_id: str
    step_name: str
    mass_transferred_kg: float


@dataclass(frozen=True)
class MassReconciliationReport:
    """Mass conservation audit across the manufacturing genealogy DAG."""

    total_input_kg: float
    total_output_product_kg: float
    total_waste_and_sample_kg: float
    holdup_loss_kg: float
    closure_error_pct: float
    balanced: bool


class LotGenealogyDAG:
    """Directed acyclic lot genealogy tracker with strict mass-balance verification."""

    def __init__(self) -> None:
        self.lots: Dict[str, MaterialLot] = {}
        self.edges: List[GenealogyEdge] = []

    def register_lot(self, lot: MaterialLot) -> None:
        if lot.mass_kg < 0.0:
            raise ValueError(f"Lot {lot.lot_id!r} cannot have negative mass")
        self.lots[lot.lot_id] = lot

    def link(self, source_lot_id: str, target_lot_id: str, step_name: str, mass_kg: float) -> None:
        if source_lot_id not in self.lots or target_lot_id not in self.lots:
            raise KeyError("Both source and target lots must be registered before linking")
        if mass_kg < 0.0:
            raise ValueError("Transferred mass must be non-negative")
        self.edges.append(GenealogyEdge(source_lot_id, target_lot_id, step_name, mass_kg))

    def ancestors(self, lot_id: str) -> List[str]:
        """Return upstream parent lot IDs via BFS."""
        visited: List[str] = []
        queue = [lot_id]
        seen = {lot_id}
        while queue:
            cur = queue.pop(0)
            for edge in self.edges:
                if edge.target_lot_id == cur and edge.source_lot_id not in seen:
                    seen.add(edge.source_lot_id)
                    visited.append(edge.source_lot_id)
                    queue.append(edge.source_lot_id)
        return visited

    def reconcile_mass_balance(
        self,
        holdup_kg: float = 0.0,
        tolerance_pct: float = 1.5,
    ) -> MassReconciliationReport:
        """Check ``sum(RAW) == sum(FINISHED + WASTE + SAMPLE) + holdup`` within tolerance."""
        total_in = sum(l.mass_kg for l in self.lots.values() if l.role == "RAW")
        total_fin = sum(l.mass_kg for l in self.lots.values() if l.role == "FINISHED")
        total_ws = sum(l.mass_kg for l in self.lots.values() if l.role in {"WASTE", "SAMPLE"})
        accounted = total_fin + total_ws + holdup_kg
        err_pct = 100.0 * abs(total_in - accounted) / max(total_in, 1e-9)
        return MassReconciliationReport(
            total_input_kg=total_in,
            total_output_product_kg=total_fin,
            total_waste_and_sample_kg=total_ws,
            holdup_loss_kg=holdup_kg,
            closure_error_pct=err_pct,
            balanced=err_pct <= tolerance_pct,
        )


# --- DSCSA GS1 GTIN-14 / SGTIN Serialization ---


def compute_gs1_check_digit(digits_13: str) -> str:
    """Compute the GS1 Mod-10 check digit for a 13-digit GTIN-14 prefix."""
    if len(digits_13) != 13 or not digits_13.isdigit():
        raise ValueError("GTIN-14 prefix must be exactly 13 numeric digits")
    total = 0
    for idx, ch in enumerate(reversed(digits_13)):
        weight = 3 if (idx % 2 == 0) else 1
        total += int(ch) * weight
    return str((10 - (total % 10)) % 10)


def validate_gtin14(gtin14: str) -> bool:
    """Validate a 14-digit GS1 GTIN-14 identifier against its Mod-10 check digit."""
    if len(gtin14) != 14 or not gtin14.isdigit():
        return False
    return compute_gs1_check_digit(gtin14[:13]) == gtin14[13]


@dataclass
class SerializationHierarchy:
    """DSCSA unit -> bundle -> case -> pallet packaging aggregation manifest."""

    gtin14: str
    batch_id: str
    unit_sgtins: List[str] = field(default_factory=list)
    cases: Dict[str, List[str]] = field(default_factory=dict)
    pallets: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def generate(
        cls,
        gtin14: str,
        batch_id: str,
        unit_count: int,
        units_per_case: int = 12,
        cases_per_pallet: int = 4,
    ) -> "SerializationHierarchy":
        if not validate_gtin14(gtin14):
            raise ValueError(f"Invalid GTIN-14 check digit: {gtin14!r}")
        hierarchy = cls(gtin14=gtin14, batch_id=batch_id)
        for u in range(unit_count):
            serial = f"{batch_id}-U{u + 1:05d}"
            sgtin = f"(01){gtin14}(21){serial}"
            hierarchy.unit_sgtins.append(sgtin)

        case_idx = 0
        for start in range(0, unit_count, units_per_case):
            case_idx += 1
            case_sscc = f"(00)0{gtin14[:12]}{case_idx:05d}"
            hierarchy.cases[case_sscc] = hierarchy.unit_sgtins[start:start + units_per_case]

        case_ids = list(hierarchy.cases.keys())
        pallet_idx = 0
        for start in range(0, len(case_ids), cases_per_pallet):
            pallet_idx += 1
            pallet_sscc = f"(00)1{gtin14[:12]}{pallet_idx:05d}"
            hierarchy.pallets[pallet_sscc] = case_ids[start:start + cases_per_pallet]

        return hierarchy

    def verify_completeness(self) -> bool:
        """Verify every serialized unit maps uniquely to one case and one pallet."""
        packed_units = [u for units in self.cases.values() for u in units]
        packed_cases = [c for cases in self.pallets.values() for c in cases]
        return (
            len(packed_units) == len(self.unit_sgtins)
            and set(packed_units) == set(self.unit_sgtins)
            and set(packed_cases) == set(self.cases.keys())
        )
