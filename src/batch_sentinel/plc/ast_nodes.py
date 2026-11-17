"""Typed AST nodes for IEC 61131-3 Structured Text and Ladder Logic programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union


PLCDataType = Literal["BOOL", "INT", "REAL", "TIME"]
VarDirection = Literal["VAR", "VAR_INPUT", "VAR_OUTPUT", "VAR_IN_OUT"]


@dataclass(frozen=True)
class SourceSpan:
    """Line and column location inside a Structured Text source string."""

    line: int
    column: int


@dataclass(frozen=True)
class VarDecl:
    """Variable or function-block instance declaration inside a PLC program."""

    name: str
    data_type: str  # BOOL, INT, REAL, TIME, TON, TOF, TP, CTU, CTD, R_TRIG, F_TRIG, PID_FB
    direction: VarDirection = "VAR"
    initial_value: Optional[Union[bool, int, float]] = None
    span: Optional[SourceSpan] = None


# --- Expressions ---


@dataclass(frozen=True)
class LiteralExpr:
    """Scalar literal expression (BOOL, INT, REAL, or TIME in seconds)."""

    value: Union[bool, int, float]
    data_type: PLCDataType
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class VarRef:
    """Variable reference or dotted FB field access (e.g., ``TIMER1.Q``)."""

    name: str
    field_name: Optional[str] = None
    span: Optional[SourceSpan] = None

    @property
    def full_name(self) -> str:
        return f"{self.name}.{self.field_name}" if self.field_name else self.name


@dataclass(frozen=True)
class UnaryExpr:
    """Unary operator expression (``NOT`` or unary ``-``)."""

    op: str
    operand: Expr
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class BinaryExpr:
    """Binary arithmetic, relational, or logical expression."""

    op: str
    left: Expr
    right: Expr
    span: Optional[SourceSpan] = None


Expr = Union[LiteralExpr, VarRef, UnaryExpr, BinaryExpr]


# --- Statements ---


@dataclass(frozen=True)
class AssignStmt:
    """Assignment statement ``target := expr;``."""

    target: VarRef
    expr: Expr
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class IfBranch:
    """Single ``IF`` or ``ELSIF`` condition/body pair."""

    condition: Expr
    body: Tuple[Stmt, ...]


@dataclass(frozen=True)
class IfStmt:
    """Conditional branching statement ``IF ... ELSIF ... ELSE ... END_IF``."""

    branches: Tuple[IfBranch, ...]
    else_body: Tuple[Stmt, ...] = ()
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class CaseArm:
    """Single selector arm inside a ``CASE`` statement."""

    selectors: Tuple[int, ...]
    body: Tuple[Stmt, ...]


@dataclass(frozen=True)
class CaseStmt:
    """Integer selector ``CASE expr OF ... END_CASE`` statement."""

    selector: Expr
    arms: Tuple[CaseArm, ...]
    else_body: Tuple[Stmt, ...] = ()
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class ForStmt:
    """Bounded ``FOR idx := start TO end [BY step] DO ... END_FOR`` loop."""

    iterator: str
    start: Expr
    end: Expr
    step: Optional[Expr]
    body: Tuple[Stmt, ...]
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class WhileStmt:
    """Bounded ``WHILE cond DO ... END_WHILE`` loop (guarded by static max iter)."""

    condition: Expr
    body: Tuple[Stmt, ...]
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class FBInvocation:
    """Function block call such as ``TON_PURGE(IN := START, PT := T#5s);``."""

    instance_name: str
    params: Tuple[Tuple[str, Expr], ...]
    span: Optional[SourceSpan] = None


Stmt = Union[AssignStmt, IfStmt, CaseStmt, ForStmt, WhileStmt, FBInvocation]


@dataclass(frozen=True)
class ProgramNode:
    """Top-level compiled IEC 61131-3 Structured Text program AST."""

    name: str
    declarations: Tuple[VarDecl, ...]
    statements: Tuple[Stmt, ...]


# --- Ladder Logic (LD) AST ---


@dataclass(frozen=True)
class ContactNO:
    """Normally-Open (--| |--) ladder contact reading a BOOL tag."""

    tag: str


@dataclass(frozen=True)
class ContactNC:
    """Normally-Closed (--|/|-- ) ladder contact reading inverted BOOL tag."""

    tag: str


@dataclass(frozen=True)
class ComparatorBlock:
    """Inline analog comparison contact (e.g., ``TEMP_C > 121.0``)."""

    tag: str
    op: Literal["<", "<=", ">", ">=", "=", "<>"]
    threshold: float


LadderElement = Union[ContactNO, ContactNC, ComparatorBlock]


@dataclass(frozen=True)
class LadderCoil:
    """Ladder output coil: standard ``COIL``, latching ``SET``, or unlatching ``RESET``."""

    tag: str
    mode: Literal["COIL", "SET", "RESET"] = "COIL"


@dataclass(frozen=True)
class LadderRung:
    """Single Ladder Logic rung with parallel series-branches feeding output coils.

    Power flows if ANY branch in ``branches`` evaluates to True (all elements in
    that branch evaluate to True).
    """

    rung_id: str
    comment: str
    branches: Tuple[Tuple[LadderElement, ...], ...]
    coils: Tuple[LadderCoil, ...]


@dataclass
class LadderProgram:
    """Collection of Ladder Logic rungs evaluated top-to-bottom per OB1 scan."""

    name: str
    rungs: List[LadderRung] = field(default_factory=list)
    tag_types: Dict[str, PLCDataType] = field(default_factory=dict)
