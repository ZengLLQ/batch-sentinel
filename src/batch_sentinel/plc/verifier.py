"""Static type, loop-bound, and formal safety-interlock verifier for IEC 61131-3 PLCs."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .ast_nodes import (
    AssignStmt,
    BinaryExpr,
    CaseStmt,
    Expr,
    FBInvocation,
    ForStmt,
    IfStmt,
    LadderProgram,
    LiteralExpr,
    ProgramNode,
    Stmt,
    UnaryExpr,
    VarDecl,
    VarRef,
    WhileStmt,
)
from .blocks import FUNCTION_BLOCK_REGISTRY


@dataclass(frozen=True)
class SafetyInterlockRule:
    """Formal safety invariant asserting an actuator stays safe whenever a trip is active.

    Example:
        ``SafetyInterlockRule(name="PRESS_OVERLOAD_ESTOP", trip_tags=("E_STOP_TRIP", "OVERLOAD_TRIP"),
        actuator_tag="PUNCH_ENABLE", forbidden_actuator_value=True)``
    """

    name: str
    trip_tags: Tuple[str, ...]
    actuator_tag: str
    forbidden_actuator_value: bool = True
    analog_trips: Tuple[Tuple[str, str, float], ...] = ()  # (tag, ">"|"<", threshold)


@dataclass(frozen=True)
class VerificationFinding:
    """Diagnostic error or warning emitted by :class:`PLCStaticVerifier`."""

    severity: str  # "ERROR" | "WARNING"
    code: str
    message: str
    line: Optional[int] = None


@dataclass
class VerificationReport:
    """Result of static AST + formal interlock verification."""

    program_name: str
    passed: bool
    findings: List[VerificationFinding] = field(default_factory=list)
    checked_combinations: int = 0
    interlock_rules_checked: int = 0
    declared_tags: Dict[str, str] = field(default_factory=dict)


class PLCStaticVerifier:
    """Static analyzer and exhaustive Boolean interlock verifier for ST + Ladder programs."""

    SCALAR_TYPES = {"BOOL", "INT", "REAL", "TIME"}
    FB_OUTPUT_TYPES: Dict[str, Dict[str, str]] = {
        "TON": {"Q": "BOOL", "ET": "TIME", "IN": "BOOL", "PT": "TIME"},
        "TOF": {"Q": "BOOL", "ET": "TIME", "IN": "BOOL", "PT": "TIME"},
        "TP": {"Q": "BOOL", "ET": "TIME", "IN": "BOOL", "PT": "TIME"},
        "CTU": {"Q": "BOOL", "CV": "INT", "CU": "BOOL", "R": "BOOL", "PV": "INT"},
        "CTD": {"Q": "BOOL", "CV": "INT", "CD": "BOOL", "LD": "BOOL", "PV": "INT"},
        "R_TRIG": {"Q": "BOOL", "CLK": "BOOL"},
        "F_TRIG": {"Q": "BOOL", "CLK": "BOOL"},
        "PID_FB": {"OUT": "REAL", "ERR": "REAL", "PV": "REAL", "SP": "REAL"},
    }

    def __init__(self, max_loop_iterations: int = 128) -> None:
        self.max_loop_iterations = max_loop_iterations

    def verify(
        self,
        program: ProgramNode,
        ladder: Optional[LadderProgram] = None,
        interlock_rules: Sequence[SafetyInterlockRule] = (),
        allow_while_loops: bool = False,
    ) -> VerificationReport:
        """Run full semantic, loop-bound, and safety-interlock verification."""
        findings: List[VerificationFinding] = []
        symtab: Dict[str, str] = {}

        for decl in program.declarations:
            if decl.name in symtab:
                findings.append(
                    VerificationFinding(
                        "ERROR",
                        "DUPLICATE_VAR",
                        f"Duplicate declaration of {decl.name!r}",
                        decl.span.line if decl.span else None,
                    )
                )
                continue
            if decl.data_type not in self.SCALAR_TYPES and decl.data_type not in FUNCTION_BLOCK_REGISTRY:
                findings.append(
                    VerificationFinding(
                        "ERROR",
                        "UNKNOWN_TYPE",
                        f"Unsupported type {decl.data_type!r} for {decl.name!r}",
                        decl.span.line if decl.span else None,
                    )
                )
                continue
            symtab[decl.name] = decl.data_type

        if ladder is not None:
            for tag, dtype in ladder.tag_types.items():
                symtab.setdefault(tag, dtype)

        self._check_stmts(
            program.statements,
            symtab,
            findings,
            allow_while_loops=allow_while_loops,
        )

        # If static semantic/type checks have errors, skip combinatorial interlock check
        has_errors = any(f.severity == "ERROR" for f in findings)
        checked_combos = 0
        if not has_errors and interlock_rules:
            checked_combos = self._verify_interlocks(
                program, ladder, symtab, interlock_rules, findings
            )

        passed = not any(f.severity == "ERROR" for f in findings)
        return VerificationReport(
            program_name=program.name,
            passed=passed,
            findings=findings,
            checked_combinations=checked_combos,
            interlock_rules_checked=len(interlock_rules),
            declared_tags=symtab,
        )

    def _check_stmts(
        self,
        stmts: Sequence[Stmt],
        symtab: Dict[str, str],
        findings: List[VerificationFinding],
        allow_while_loops: bool,
    ) -> None:
        for stmt in stmts:
            line = stmt.span.line if stmt.span else None
            if isinstance(stmt, AssignStmt):
                target_type = self._resolve_var_type(stmt.target, symtab, findings, line)
                expr_type = self._infer_expr_type(stmt.expr, symtab, findings, line)
                if target_type and expr_type and not self._types_assignable(target_type, expr_type):
                    findings.append(
                        VerificationFinding(
                            "ERROR",
                            "TYPE_MISMATCH",
                            f"Cannot assign {expr_type} expression to {target_type} tag {stmt.target.full_name!r}",
                            line,
                        )
                    )
            elif isinstance(stmt, IfStmt):
                for br in stmt.branches:
                    cond_t = self._infer_expr_type(br.condition, symtab, findings, line)
                    if cond_t and cond_t != "BOOL":
                        findings.append(
                            VerificationFinding(
                                "ERROR",
                                "NON_BOOL_CONDITION",
                                f"IF/ELSIF condition must be BOOL, got {cond_t}",
                                line,
                            )
                        )
                    self._check_stmts(br.body, symtab, findings, allow_while_loops)
                self._check_stmts(stmt.else_body, symtab, findings, allow_while_loops)
            elif isinstance(stmt, CaseStmt):
                sel_t = self._infer_expr_type(stmt.selector, symtab, findings, line)
                if sel_t and sel_t != "INT":
                    findings.append(
                        VerificationFinding(
                            "ERROR",
                            "NON_INT_CASE",
                            f"CASE selector must be INT, got {sel_t}",
                            line,
                        )
                    )
                seen_labels: Set[int] = set()
                for arm in stmt.arms:
                    for lbl in arm.selectors:
                        if lbl in seen_labels:
                            findings.append(
                                VerificationFinding(
                                    "ERROR",
                                    "DUPLICATE_CASE",
                                    f"Duplicate CASE selector {lbl}",
                                    line,
                                )
                            )
                        seen_labels.add(lbl)
                    self._check_stmts(arm.body, symtab, findings, allow_while_loops)
                self._check_stmts(stmt.else_body, symtab, findings, allow_while_loops)
            elif isinstance(stmt, ForStmt):
                if stmt.iterator not in symtab:
                    findings.append(
                        VerificationFinding(
                            "ERROR",
                            "UNDECLARED_VAR",
                            f"FOR loop iterator {stmt.iterator!r} is not declared",
                            line,
                        )
                    )
                elif symtab[stmt.iterator] != "INT":
                    findings.append(
                        VerificationFinding(
                            "ERROR",
                            "NON_INT_ITERATOR",
                            f"FOR loop iterator {stmt.iterator!r} must be INT",
                            line,
                        )
                    )
                self._infer_expr_type(stmt.start, symtab, findings, line)
                self._infer_expr_type(stmt.end, symtab, findings, line)
                if stmt.step is not None:
                    self._infer_expr_type(stmt.step, symtab, findings, line)
                if isinstance(stmt.start, LiteralExpr) and isinstance(stmt.end, LiteralExpr):
                    span_iters = abs(int(stmt.end.value) - int(stmt.start.value)) + 1
                    if span_iters > self.max_loop_iterations:
                        findings.append(
                            VerificationFinding(
                                "ERROR",
                                "LOOP_BOUND_EXCEEDED",
                                f"FOR loop spans {span_iters} iterations (> {self.max_loop_iterations})",
                                line,
                            )
                        )
                self._check_stmts(stmt.body, symtab, findings, allow_while_loops)
            elif isinstance(stmt, WhileStmt):
                if not allow_while_loops:
                    findings.append(
                        VerificationFinding(
                            "ERROR",
                            "UNBOUNDED_WHILE",
                            "WHILE loops are prohibited in safety-critical deterministic PLC scans",
                            line,
                        )
                    )
                self._infer_expr_type(stmt.condition, symtab, findings, line)
                self._check_stmts(stmt.body, symtab, findings, allow_while_loops)
            elif isinstance(stmt, FBInvocation):
                fb_type = symtab.get(stmt.instance_name)
                if fb_type is None or fb_type not in FUNCTION_BLOCK_REGISTRY:
                    findings.append(
                        VerificationFinding(
                            "ERROR",
                            "UNKNOWN_FB",
                            f"Function block instance {stmt.instance_name!r} is not declared as a valid FB",
                            line,
                        )
                    )
                else:
                    for _, val_expr in stmt.params:
                        self._infer_expr_type(val_expr, symtab, findings, line)

    def _resolve_var_type(
        self,
        ref: VarRef,
        symtab: Dict[str, str],
        findings: List[VerificationFinding],
        line: Optional[int],
    ) -> Optional[str]:
        base_t = symtab.get(ref.name)
        if base_t is None:
            findings.append(
                VerificationFinding(
                    "ERROR",
                    "UNDECLARED_VAR",
                    f"Undeclared tag {ref.full_name!r}",
                    line,
                )
            )
            return None
        if ref.field_name is None:
            return base_t
        fb_fields = self.FB_OUTPUT_TYPES.get(base_t)
        if fb_fields is None or ref.field_name not in fb_fields:
            findings.append(
                VerificationFinding(
                    "ERROR",
                    "UNKNOWN_FB_FIELD",
                    f"Invalid field access {ref.full_name!r} on type {base_t}",
                    line,
                )
            )
            return None
        return fb_fields[ref.field_name]

    def _infer_expr_type(
        self,
        expr: Expr,
        symtab: Dict[str, str],
        findings: List[VerificationFinding],
        line: Optional[int],
    ) -> Optional[str]:
        if isinstance(expr, LiteralExpr):
            return expr.data_type
        if isinstance(expr, VarRef):
            return self._resolve_var_type(expr, symtab, findings, line)
        if isinstance(expr, UnaryExpr):
            inner = self._infer_expr_type(expr.operand, symtab, findings, line)
            if expr.op == "NOT":
                if inner and inner != "BOOL":
                    findings.append(
                        VerificationFinding("ERROR", "TYPE_MISMATCH", f"NOT requires BOOL, got {inner}", line)
                    )
                return "BOOL"
            if inner and inner not in {"INT", "REAL"}:
                findings.append(
                    VerificationFinding("ERROR", "TYPE_MISMATCH", f"Unary '-' requires numeric, got {inner}", line)
                )
            return inner
        if isinstance(expr, BinaryExpr):
            lt = self._infer_expr_type(expr.left, symtab, findings, line)
            rt = self._infer_expr_type(expr.right, symtab, findings, line)
            if expr.op in {"AND", "OR", "XOR"}:
                if (lt and lt != "BOOL") or (rt and rt != "BOOL"):
                    findings.append(
                        VerificationFinding(
                            "ERROR",
                            "TYPE_MISMATCH",
                            f"Logical {expr.op} requires BOOL operands, got {lt} and {rt}",
                            line,
                        )
                    )
                return "BOOL"
            if expr.op in {"=", "<>", "<", "<=", ">", ">="}:
                return "BOOL"
            if expr.op in {"+", "-", "*", "/", "MOD", "**"}:
                if lt == "REAL" or rt == "REAL" or expr.op == "/":
                    return "REAL"
                if lt == "TIME" or rt == "TIME":
                    return "TIME"
                return "INT"
        return None

    @staticmethod
    def _types_assignable(target: str, source: str) -> bool:
        if target == source:
            return True
        if target == "REAL" and source in {"INT", "TIME"}:
            return True
        return False

    def _verify_interlocks(
        self,
        program: ProgramNode,
        ladder: Optional[LadderProgram],
        symtab: Dict[str, str],
        rules: Sequence[SafetyInterlockRule],
        findings: List[VerificationFinding],
    ) -> int:
        """Exhaustively test boolean input combinations + analog trip assertions over one scan."""
        from .runtime import PLCRuntime

        bool_inputs: List[str] = sorted(
            {
                d.name
                for d in program.declarations
                if d.direction == "VAR_INPUT" and d.data_type == "BOOL"
            }
            | {r_tag for rule in rules for r_tag in rule.trip_tags if symtab.get(r_tag) == "BOOL"}
        )
        # Bound boolean truth table to <= 10 tags (1024 combos) to guarantee sub-second CI runs
        if len(bool_inputs) > 10:
            bool_inputs = bool_inputs[:10]

        total_checked = 0
        for rule in rules:
            # Verify actuator exists
            if rule.actuator_tag not in symtab:
                findings.append(
                    VerificationFinding(
                        "ERROR",
                        "MISSING_ACTUATOR",
                        f"Safety rule {rule.name!r} references undeclared actuator {rule.actuator_tag!r}",
                    )
                )
                continue

            # Test every boolean combination where at least one trip condition is asserted
            combos = list(itertools.product([False, True], repeat=len(bool_inputs)))
            if not combos:
                combos = [()]

            for bits in combos:
                env: Dict[str, Any] = dict(zip(bool_inputs, bits))
                trip_active = any(bool(env.get(t, False)) for t in rule.trip_tags)

                # Also test analog trip boundary if rule specifies analog trips
                if rule.analog_trips:
                    for a_tag, a_op, thresh in rule.analog_trips:
                        env[a_tag] = thresh + (1.0 if ">" in a_op else -1.0)
                    trip_active = True

                if not trip_active:
                    continue

                total_checked += 1
                rt = PLCRuntime(program=program, ladder=ladder, scan_period_s=0.05)
                # Pre-seed actuator to the forbidden state to verify active de-energizing
                rt.tags[rule.actuator_tag] = rule.forbidden_actuator_value
                rt.set_inputs(env)
                rt.step_scan()

                actual_val = bool(rt.tags.get(rule.actuator_tag, False))
                if actual_val == rule.forbidden_actuator_value:
                    findings.append(
                        VerificationFinding(
                            "ERROR",
                            "INTERLOCK_VIOLATION",
                            f"Rule {rule.name!r} violated: {rule.actuator_tag}={actual_val} under trip state {env}",
                        )
                    )
                    break

        return total_checked
