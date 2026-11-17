"""ISA-88 hierarchical batch procedural control and SIP F0 thermal lethality engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple


class S88State(str, Enum):
    """Canonical ANSI/ISA-88 procedural element states."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    HOLDING = "HOLDING"
    HELD = "HELD"
    RESTARTING = "RESTARTING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"


class S88TransitionError(RuntimeError):
    """Raised on an illegal ISA-88 state machine command or equipment conflict."""


ALLOWED_TRANSITIONS: Dict[S88State, frozenset[S88State]] = {
    S88State.IDLE: frozenset({S88State.RUNNING, S88State.ABORTING}),
    S88State.RUNNING: frozenset(
        {S88State.COMPLETE, S88State.PAUSING, S88State.HOLDING, S88State.STOPPING, S88State.ABORTING}
    ),
    S88State.PAUSING: frozenset({S88State.PAUSED, S88State.HOLDING, S88State.STOPPING, S88State.ABORTING}),
    S88State.PAUSED: frozenset({S88State.RUNNING, S88State.HOLDING, S88State.STOPPING, S88State.ABORTING}),
    S88State.HOLDING: frozenset({S88State.HELD, S88State.STOPPING, S88State.ABORTING}),
    S88State.HELD: frozenset({S88State.RESTARTING, S88State.STOPPING, S88State.ABORTING}),
    S88State.RESTARTING: frozenset({S88State.RUNNING, S88State.HOLDING, S88State.STOPPING, S88State.ABORTING}),
    S88State.STOPPING: frozenset({S88State.STOPPED, S88State.ABORTING}),
    S88State.STOPPED: frozenset({S88State.IDLE}),
    S88State.ABORTING: frozenset({S88State.ABORTED}),
    S88State.ABORTED: frozenset({S88State.IDLE}),
    S88State.COMPLETE: frozenset({S88State.IDLE}),
}


def compute_sip_f0_lethality(
    temps_c: Sequence[float],
    dt_min: float,
    t_ref_c: float = 121.1,
    z_c: float = 10.0,
    min_active_temp_c: float = 100.0,
) -> float:
    """Integrate Steam-in-Place (SIP) thermal lethality ``F0`` (equivalent minutes at 121.1 C).

    Uses trapezoidal integration of ``10^((T(t) - T_ref) / z)`` over intervals where
    ``T >= min_active_temp_c``.
    """
    if dt_min <= 0.0:
        raise ValueError("dt_min must be positive")
    if len(temps_c) < 2:
        return 0.0

    def rate(t_c: float) -> float:
        if t_c < min_active_temp_c:
            return 0.0
        return math.pow(10.0, (t_c - t_ref_c) / z_c)

    f0 = 0.0
    for i in range(len(temps_c) - 1):
        f0 += 0.5 * (rate(temps_c[i]) + rate(temps_c[i + 1])) * dt_min
    return f0


@dataclass
class PhaseExecutionRecord:
    """Audit record of an executed ISA-88 equipment phase."""

    unit_procedure: str
    operation: str
    phase: str
    unit_id: str
    final_state: S88State
    metrics: Dict[str, float]


@dataclass
class S88Phase:
    """Lowest procedural element executing directly on an allocated equipment unit."""

    name: str
    unit_id: str
    action: Callable[[], Dict[str, float]]
    state: S88State = S88State.IDLE

    def transition(self, target: S88State) -> None:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise S88TransitionError(
                f"Illegal S88 transition {self.state.value} -> {target.value} on phase {self.name!r}"
            )
        self.state = target


@dataclass
class S88Operation:
    """Ordered sequence of ISA-88 phases within a unit operation."""

    name: str
    phases: List[S88Phase] = field(default_factory=list)


@dataclass
class S88UnitProcedure:
    """Equipment-assigned unit procedure composed of sequential operations."""

    name: str
    unit_id: str
    operations: List[S88Operation] = field(default_factory=list)


@dataclass
class S88Procedure:
    """Master batch recipe procedure spanning multiple equipment unit procedures."""

    name: str
    unit_procedures: List[S88UnitProcedure] = field(default_factory=list)


class ISA88BatchExecutive:
    """Executes ISA-88 master recipes with unit arbitration and deterministic state tracking."""

    def __init__(self) -> None:
        self.allocated_units: Dict[str, str] = {}  # unit_id -> batch_id
        self.state: S88State = S88State.IDLE
        self.records: List[PhaseExecutionRecord] = []

    def acquire_unit(self, unit_id: str, batch_id: str) -> None:
        owner = self.allocated_units.get(unit_id)
        if owner is not None and owner != batch_id:
            raise S88TransitionError(
                f"Equipment unit {unit_id!r} already locked by batch {owner!r}"
            )
        self.allocated_units[unit_id] = batch_id

    def release_unit(self, unit_id: str, batch_id: str) -> None:
        if self.allocated_units.get(unit_id) == batch_id:
            del self.allocated_units[unit_id]

    def transition_recipe(self, target: S88State) -> None:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise S88TransitionError(
                f"Illegal recipe S88 transition {self.state.value} -> {target.value}"
            )
        self.state = target

    def run_procedure(
        self,
        procedure: S88Procedure,
        batch_id: str,
        abort_on_predicate: Optional[Callable[[str, Dict[str, float]], Optional[str]]] = None,
    ) -> Tuple[S88State, List[PhaseExecutionRecord], Optional[str]]:
        """Execute an entire ISA-88 procedure, acquiring and releasing each unit cleanly."""
        if self.state != S88State.IDLE:
            self.state = S88State.IDLE
        self.transition_recipe(S88State.RUNNING)
        records: List[PhaseExecutionRecord] = []
        abort_reason: Optional[str] = None

        for up in procedure.unit_procedures:
            self.acquire_unit(up.unit_id, batch_id)
            try:
                for op in up.operations:
                    for phase in op.phases:
                        phase.transition(S88State.RUNNING)
                        metrics = phase.action()
                        if abort_on_predicate is not None:
                            reason = abort_on_predicate(phase.name, metrics)
                            if reason:
                                phase.transition(S88State.ABORTING)
                                phase.transition(S88State.ABORTED)
                                self.transition_recipe(S88State.ABORTING)
                                self.transition_recipe(S88State.ABORTED)
                                records.append(
                                    PhaseExecutionRecord(
                                        up.name, op.name, phase.name, phase.unit_id, phase.state, metrics
                                    )
                                )
                                abort_reason = reason
                                self.records.extend(records)
                                return self.state, records, abort_reason
                        phase.transition(S88State.COMPLETE)
                        records.append(
                            PhaseExecutionRecord(
                                up.name, op.name, phase.name, phase.unit_id, phase.state, metrics
                            )
                        )
            finally:
                self.release_unit(up.unit_id, batch_id)

        self.transition_recipe(S88State.COMPLETE)
        self.records.extend(records)
        return self.state, records, abort_reason
