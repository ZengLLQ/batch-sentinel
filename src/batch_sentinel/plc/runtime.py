"""Deterministic OB1 scan-cycle runtime for IEC 61131-3 ST and Ladder Logic."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .ast_nodes import (
    AssignStmt,
    BinaryExpr,
    CaseStmt,
    ComparatorBlock,
    ContactNC,
    ContactNO,
    Expr,
    FBInvocation,
    ForStmt,
    IfStmt,
    LadderElement,
    LadderProgram,
    LiteralExpr,
    ProgramNode,
    Stmt,
    UnaryExpr,
    VarRef,
    WhileStmt,
)
from .blocks import FUNCTION_BLOCK_REGISTRY, FunctionBlock


class PLCWatchdogError(RuntimeError):
    """Raised if an OB1 scan cycle exceeds its deterministic watchdog deadline."""


@dataclass(frozen=True)
class ScanTelemetry:
    """Per-scan execution record for cycle-time, watchdog, and actuator audit."""

    scan_index: int
    sim_time_s: float
    exec_duration_s: float
    watchdog_ok: bool
    outputs: Dict[str, Any]


class PLCRuntime:
    """Deterministic virtual PLC executing Ladder Logic rungs and Structured Text ASTs.

    Each call to :meth:`step_scan` follows the IEC 61131-3 cyclic execution model:
    1. Freeze process input image.
    2. Evaluate Ladder Logic rungs (if attached).
    3. Execute Structured Text statements and stateful Function Blocks.
    4. Commit output image and record watchdog telemetry.
    """

    DEFAULT_INIT: Dict[str, Any] = {
        "BOOL": False,
        "INT": 0,
        "REAL": 0.0,
        "TIME": 0.0,
    }

    def __init__(
        self,
        program: ProgramNode,
        ladder: Optional[LadderProgram] = None,
        scan_period_s: float = 0.05,
        watchdog_timeout_s: float = 0.045,
        max_loop_iterations: int = 128,
        jitter_std_s: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.program = program
        self.ladder = ladder
        self.scan_period_s = scan_period_s
        self.watchdog_timeout_s = watchdog_timeout_s
        self.max_loop_iterations = max_loop_iterations
        self.jitter_std_s = jitter_std_s
        self._rng = np.random.default_rng(seed)

        self.tags: Dict[str, Any] = {}
        self.tag_types: Dict[str, str] = {}
        self.fb_instances: Dict[str, FunctionBlock] = {}
        self.forced_faults: Dict[str, Any] = {}

        for decl in program.declarations:
            if decl.data_type in FUNCTION_BLOCK_REGISTRY:
                self.fb_instances[decl.name] = FUNCTION_BLOCK_REGISTRY[decl.data_type]()
                self.tag_types[decl.name] = decl.data_type
            else:
                default_val = (
                    decl.initial_value
                    if decl.initial_value is not None
                    else self.DEFAULT_INIT[decl.data_type]
                )
                self.tags[decl.name] = default_val
                self.tag_types[decl.name] = decl.data_type

        if ladder is not None:
            for tag, dtype in ladder.tag_types.items():
                if tag not in self.tags and tag not in self.fb_instances:
                    self.tags[tag] = self.DEFAULT_INIT[dtype]
                    self.tag_types[tag] = dtype

        self.scan_count: int = 0
        self.sim_time_s: float = 0.0
        self.history: List[ScanTelemetry] = []

    def set_inputs(self, values: Dict[str, Any]) -> None:
        """Update input tags before the next OB1 scan (honoring active stuck faults)."""
        for key, val in values.items():
            if key in self.forced_faults:
                self.tags[key] = self.forced_faults[key]
            else:
                self.tags[key] = val

    def inject_stuck_tag(self, tag: str, stuck_value: Any) -> None:
        """Simulate a hardware sensor or relay stuck at a fixed value."""
        self.forced_faults[tag] = stuck_value
        self.tags[tag] = stuck_value

    def clear_faults(self) -> None:
        """Remove all injected tag faults."""
        self.forced_faults.clear()

    def step_scan(self, inputs: Optional[Dict[str, Any]] = None) -> ScanTelemetry:
        """Execute exactly one deterministic OB1 scan cycle."""
        t0 = time.perf_counter()
        if inputs:
            self.set_inputs(inputs)

        if self.ladder is not None:
            self._eval_ladder(self.ladder)

        self._exec_stmts(self.program.statements)

        # Re-apply stuck faults on outputs if an actuator is jammed
        for tag, stuck_val in self.forced_faults.items():
            self.tags[tag] = stuck_val

        elapsed_wall = time.perf_counter() - t0
        jitter = abs(float(self._rng.normal(0.0, self.jitter_std_s))) if self.jitter_std_s > 0.0 else 0.0
        exec_duration = elapsed_wall + jitter
        watchdog_ok = exec_duration <= self.watchdog_timeout_s

        self.scan_count += 1
        self.sim_time_s += self.scan_period_s

        out_snapshot = {
            d.name: self.tags[d.name]
            for d in self.program.declarations
            if d.direction in {"VAR_OUTPUT", "VAR_IN_OUT"} and d.name in self.tags
        }
        # Also include ladder coil tags
        if self.ladder is not None:
            for rung in self.ladder.rungs:
                for coil in rung.coils:
                    out_snapshot[coil.tag] = self.tags.get(coil.tag)

        telemetry = ScanTelemetry(
            scan_index=self.scan_count,
            sim_time_s=self.sim_time_s,
            exec_duration_s=exec_duration,
            watchdog_ok=watchdog_ok,
            outputs=out_snapshot,
        )
        self.history.append(telemetry)
        return telemetry

    def _eval_ladder(self, ladder: LadderProgram) -> None:
        for rung in ladder.rungs:
            energized = False
            for branch in rung.branches:
                if all(self._eval_ladder_elem(el) for el in branch):
                    energized = True
                    break
            for coil in rung.coils:
                if coil.mode == "COIL":
                    self.tags[coil.tag] = energized
                elif coil.mode == "SET" and energized:
                    self.tags[coil.tag] = True
                elif coil.mode == "RESET" and energized:
                    self.tags[coil.tag] = False

    def _eval_ladder_elem(self, elem: LadderElement) -> bool:
        if isinstance(elem, ContactNO):
            return bool(self.tags.get(elem.tag, False))
        if isinstance(elem, ContactNC):
            return not bool(self.tags.get(elem.tag, False))
        if isinstance(elem, ComparatorBlock):
            val = float(self.tags.get(elem.tag, 0.0))
            return _compare_values(elem.op, val, elem.threshold)
        return False

    def _exec_stmts(self, stmts: Sequence[Stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, AssignStmt):
                val = self._eval_expr(stmt.expr)
                self._write_var(stmt.target, val)
            elif isinstance(stmt, IfStmt):
                executed = False
                for br in stmt.branches:
                    if bool(self._eval_expr(br.condition)):
                        self._exec_stmts(br.body)
                        executed = True
                        break
                if not executed:
                    self._exec_stmts(stmt.else_body)
            elif isinstance(stmt, CaseStmt):
                sel_val = int(self._eval_expr(stmt.selector))
                matched = False
                for arm in stmt.arms:
                    if sel_val in arm.selectors:
                        self._exec_stmts(arm.body)
                        matched = True
                        break
                if not matched:
                    self._exec_stmts(stmt.else_body)
            elif isinstance(stmt, ForStmt):
                start = int(self._eval_expr(stmt.start))
                end = int(self._eval_expr(stmt.end))
                step = int(self._eval_expr(stmt.step)) if stmt.step is not None else 1
                if step == 0:
                    raise RuntimeError("FOR loop BY step cannot be 0")
                idx = start
                iters = 0
                while (idx <= end if step > 0 else idx >= end):
                    if iters >= self.max_loop_iterations:
                        raise PLCWatchdogError(
                            f"FOR loop exceeded max_loop_iterations={self.max_loop_iterations}"
                        )
                    self.tags[stmt.iterator] = idx
                    self._exec_stmts(stmt.body)
                    idx += step
                    iters += 1
            elif isinstance(stmt, WhileStmt):
                iters = 0
                while bool(self._eval_expr(stmt.condition)):
                    if iters >= self.max_loop_iterations:
                        raise PLCWatchdogError(
                            f"WHILE loop exceeded max_loop_iterations={self.max_loop_iterations}"
                        )
                    self._exec_stmts(stmt.body)
                    iters += 1
            elif isinstance(stmt, FBInvocation):
                fb = self.fb_instances[stmt.instance_name]
                evaluated_params = {k: self._eval_expr(v) for k, v in stmt.params}
                fb.call(evaluated_params, self.scan_period_s)

    def _write_var(self, ref: VarRef, value: Any) -> None:
        if ref.field_name is not None:
            fb = self.fb_instances[ref.name]
            setattr(fb, ref.field_name, value)
            return
        dtype = self.tag_types.get(ref.name)
        if dtype == "BOOL":
            self.tags[ref.name] = bool(value)
        elif dtype == "INT":
            self.tags[ref.name] = int(value)
        elif dtype in {"REAL", "TIME"}:
            self.tags[ref.name] = float(value)
        else:
            self.tags[ref.name] = value

    def _eval_expr(self, expr: Expr) -> Any:
        if isinstance(expr, LiteralExpr):
            return expr.value
        if isinstance(expr, VarRef):
            if expr.field_name is not None:
                return self.fb_instances[expr.name].read_field(expr.field_name)
            return self.tags[expr.name]
        if isinstance(expr, UnaryExpr):
            val = self._eval_expr(expr.operand)
            if expr.op == "NOT":
                return not bool(val)
            if expr.op == "-":
                return -val
        if isinstance(expr, BinaryExpr):
            if expr.op == "AND":
                return bool(self._eval_expr(expr.left)) and bool(self._eval_expr(expr.right))
            if expr.op == "OR":
                return bool(self._eval_expr(expr.left)) or bool(self._eval_expr(expr.right))
            if expr.op == "XOR":
                return bool(self._eval_expr(expr.left)) ^ bool(self._eval_expr(expr.right))
            left = self._eval_expr(expr.left)
            right = self._eval_expr(expr.right)
            if expr.op in {"=", "<>", "<", "<=", ">", ">="}:
                return _compare_values(expr.op, left, right)
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                if right == 0:
                    raise ZeroDivisionError("PLC Structured Text division by zero")
                return left / right
            if expr.op == "MOD":
                if right == 0:
                    raise ZeroDivisionError("PLC Structured Text modulo by zero")
                return int(left) % int(right)
            if expr.op == "**":
                return math.pow(float(left), float(right))
        raise RuntimeError(f"Unsupported expression node: {expr!r}")


def _compare_values(op: str, left: Any, right: Any) -> bool:
    if op == "=":
        return left == right
    if op == "<>":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise ValueError(f"Unknown comparator {op!r}")
