"""Recursive-descent parser for IEC 61131-3 Structured Text and Ladder Logic."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .ast_nodes import (
    AssignStmt,
    BinaryExpr,
    CaseArm,
    CaseStmt,
    ComparatorBlock,
    ContactNC,
    ContactNO,
    Expr,
    FBInvocation,
    ForStmt,
    IfBranch,
    IfStmt,
    LadderCoil,
    LadderElement,
    LadderProgram,
    LadderRung,
    LiteralExpr,
    PLCDataType,
    ProgramNode,
    SourceSpan,
    Stmt,
    UnaryExpr,
    VarDecl,
    VarDirection,
    VarRef,
    WhileStmt,
)
from .lexer import PLCSyntaxError, Token, parse_time_literal, tokenize_structured_text


class _STParser:
    """Recursive-descent parser over tokenized Structured Text."""

    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    @property
    def cur(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def span(self) -> SourceSpan:
        return SourceSpan(self.cur.line, self.cur.column)

    def match(self, kind: str, value: Optional[str] = None) -> Optional[Token]:
        tok = self.cur
        if tok.kind == kind and (value is None or tok.value == value):
            self.pos += 1
            return tok
        return None

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        tok = self.cur
        if tok.kind != kind or (value is not None and tok.value != value):
            expected = f"{kind}({value})" if value else kind
            raise PLCSyntaxError(
                f"Expected {expected}, got {tok.kind}({tok.value!r})",
                tok.line,
                tok.column,
            )
        self.pos += 1
        return tok

    def parse_program(self) -> ProgramNode:
        self.expect("KEYWORD", "PROGRAM")
        name_tok = self.expect("IDENT")
        decls: List[VarDecl] = []
        while self.cur.kind == "KEYWORD" and self.cur.value in {
            "VAR",
            "VAR_INPUT",
            "VAR_OUTPUT",
            "VAR_IN_OUT",
        }:
            decls.extend(self.parse_var_block())

        stmts = self.parse_statement_list(stop_keywords={"END_PROGRAM"})
        self.expect("KEYWORD", "END_PROGRAM")
        return ProgramNode(name=name_tok.value, declarations=tuple(decls), statements=tuple(stmts))

    def parse_var_block(self) -> List[VarDecl]:
        dir_tok = self.expect("KEYWORD")
        direction: VarDirection = dir_tok.value  # type: ignore[assignment]
        decls: List[VarDecl] = []
        while not (self.cur.kind == "KEYWORD" and self.cur.value == "END_VAR"):
            span = self.span()
            var_name = self.expect("IDENT").value
            self.expect("PUNCT", ":")
            type_tok = self.expect("IDENT")
            data_type = type_tok.value.upper()
            init_val = None
            if self.match("OP", ":="):
                lit = self.parse_primary()
                if not isinstance(lit, LiteralExpr):
                    raise PLCSyntaxError(
                        "Variable initial value must be a scalar literal",
                        span.line,
                        span.column,
                    )
                init_val = lit.value
            self.expect("PUNCT", ";")
            decls.append(
                VarDecl(
                    name=var_name,
                    data_type=data_type,
                    direction=direction,
                    initial_value=init_val,
                    span=span,
                )
            )
        self.expect("KEYWORD", "END_VAR")
        return decls

    def parse_statement_list(self, stop_keywords: set[str]) -> List[Stmt]:
        stmts: List[Stmt] = []
        while self.cur.kind != "EOF":
            if self.cur.kind == "KEYWORD" and self.cur.value in stop_keywords:
                break
            stmts.append(self.parse_statement())
        return stmts

    def parse_statement(self) -> Stmt:
        tok = self.cur
        if tok.kind == "KEYWORD":
            if tok.value == "IF":
                return self.parse_if()
            if tok.value == "CASE":
                return self.parse_case()
            if tok.value == "FOR":
                return self.parse_for()
            if tok.value == "WHILE":
                return self.parse_while()

        if tok.kind == "IDENT":
            # Either FBInvocation: Name(param := expr, ...) or AssignStmt: Name[.field] := expr;
            if self.peek().kind == "PUNCT" and self.peek().value == "(":
                return self.parse_fb_invocation()
            span = self.span()
            target = self.parse_var_ref()
            self.expect("OP", ":=")
            expr = self.parse_expr()
            self.expect("PUNCT", ";")
            return AssignStmt(target=target, expr=expr, span=span)

        raise PLCSyntaxError(f"Unexpected token {tok.value!r} in statement", tok.line, tok.column)

    def parse_var_ref(self) -> VarRef:
        span = self.span()
        ident = self.expect("IDENT").value
        field_name: Optional[str] = None
        if self.match("PUNCT", "."):
            field_name = self.expect("IDENT").value
        return VarRef(name=ident, field_name=field_name, span=span)

    def parse_fb_invocation(self) -> FBInvocation:
        span = self.span()
        instance_name = self.expect("IDENT").value
        self.expect("PUNCT", "(")
        params: List[Tuple[str, Expr]] = []
        if not (self.cur.kind == "PUNCT" and self.cur.value == ")"):
            while True:
                param_name = self.expect("IDENT").value
                self.expect("OP", ":=")
                val_expr = self.parse_expr()
                params.append((param_name.upper(), val_expr))
                if not self.match("PUNCT", ","):
                    break
        self.expect("PUNCT", ")")
        self.expect("PUNCT", ";")
        return FBInvocation(instance_name=instance_name, params=tuple(params), span=span)

    def parse_if(self) -> IfStmt:
        span = self.span()
        self.expect("KEYWORD", "IF")
        cond = self.parse_expr()
        self.expect("KEYWORD", "THEN")
        body = self.parse_statement_list({"ELSIF", "ELSE", "END_IF"})
        branches: List[IfBranch] = [IfBranch(condition=cond, body=tuple(body))]
        while self.match("KEYWORD", "ELSIF"):
            e_cond = self.parse_expr()
            self.expect("KEYWORD", "THEN")
            e_body = self.parse_statement_list({"ELSIF", "ELSE", "END_IF"})
            branches.append(IfBranch(condition=e_cond, body=tuple(e_body)))
        else_body: List[Stmt] = []
        if self.match("KEYWORD", "ELSE"):
            else_body = self.parse_statement_list({"END_IF"})
        self.expect("KEYWORD", "END_IF")
        self.match("PUNCT", ";")
        return IfStmt(branches=tuple(branches), else_body=tuple(else_body), span=span)

    def parse_case(self) -> CaseStmt:
        span = self.span()
        self.expect("KEYWORD", "CASE")
        selector = self.parse_expr()
        self.expect("KEYWORD", "OF")
        arms: List[CaseArm] = []
        else_body: List[Stmt] = []
        while not (self.cur.kind == "KEYWORD" and self.cur.value in {"ELSE", "END_CASE"}):
            sels: List[int] = []
            while True:
                sign = -1 if self.match("OP", "-") else 1
                num_tok = self.expect("INT")
                sels.append(sign * int(num_tok.value))
                if not self.match("PUNCT", ","):
                    break
            self.expect("PUNCT", ":")
            arm_body = self.parse_case_arm_body()
            arms.append(CaseArm(selectors=tuple(sels), body=tuple(arm_body)))
        if self.match("KEYWORD", "ELSE"):
            else_body = self.parse_statement_list({"END_CASE"})
        self.expect("KEYWORD", "END_CASE")
        self.match("PUNCT", ";")
        return CaseStmt(selector=selector, arms=tuple(arms), else_body=tuple(else_body), span=span)

    def parse_case_arm_body(self) -> List[Stmt]:
        stmts: List[Stmt] = []
        while self.cur.kind != "EOF":
            if self.cur.kind == "KEYWORD" and self.cur.value in {"ELSE", "END_CASE"}:
                break
            if self._looks_like_case_label():
                break
            stmts.append(self.parse_statement())
        return stmts

    def _looks_like_case_label(self) -> bool:
        idx = self.pos
        if self.tokens[idx].kind == "OP" and self.tokens[idx].value == "-":
            idx += 1
        if idx >= len(self.tokens) or self.tokens[idx].kind != "INT":
            return False
        idx += 1
        while idx < len(self.tokens) and self.tokens[idx].kind == "PUNCT" and self.tokens[idx].value == ",":
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx].kind == "OP" and self.tokens[idx].value == "-":
                idx += 1
            if idx >= len(self.tokens) or self.tokens[idx].kind != "INT":
                return False
            idx += 1
        return idx < len(self.tokens) and self.tokens[idx].kind == "PUNCT" and self.tokens[idx].value == ":"

    def parse_for(self) -> ForStmt:
        span = self.span()
        self.expect("KEYWORD", "FOR")
        iterator = self.expect("IDENT").value
        self.expect("OP", ":=")
        start = self.parse_expr()
        self.expect("KEYWORD", "TO")
        end = self.parse_expr()
        step: Optional[Expr] = None
        if self.match("KEYWORD", "BY"):
            step = self.parse_expr()
        self.expect("KEYWORD", "DO")
        body = self.parse_statement_list({"END_FOR"})
        self.expect("KEYWORD", "END_FOR")
        self.match("PUNCT", ";")
        return ForStmt(iterator=iterator, start=start, end=end, step=step, body=tuple(body), span=span)

    def parse_while(self) -> WhileStmt:
        span = self.span()
        self.expect("KEYWORD", "WHILE")
        cond = self.parse_expr()
        self.expect("KEYWORD", "DO")
        body = self.parse_statement_list({"END_WHILE"})
        self.expect("KEYWORD", "END_WHILE")
        self.match("PUNCT", ";")
        return WhileStmt(condition=cond, body=tuple(body), span=span)

    # --- Expression precedence ---

    def parse_expr(self) -> Expr:
        return self.parse_or()

    def parse_or(self) -> Expr:
        expr = self.parse_xor()
        while self.match("KEYWORD", "OR"):
            span = self.span()
            expr = BinaryExpr("OR", expr, self.parse_xor(), span)
        return expr

    def parse_xor(self) -> Expr:
        expr = self.parse_and()
        while self.match("KEYWORD", "XOR"):
            span = self.span()
            expr = BinaryExpr("XOR", expr, self.parse_and(), span)
        return expr

    def parse_and(self) -> Expr:
        expr = self.parse_cmp()
        while self.match("KEYWORD", "AND"):
            span = self.span()
            expr = BinaryExpr("AND", expr, self.parse_cmp(), span)
        return expr

    def parse_cmp(self) -> Expr:
        expr = self.parse_add()
        while self.cur.kind == "OP" and self.cur.value in {"=", "<>", "<", "<=", ">", ">="}:
            op = self.cur.value
            span = self.span()
            self.pos += 1
            expr = BinaryExpr(op, expr, self.parse_add(), span)
        return expr

    def parse_add(self) -> Expr:
        expr = self.parse_mul()
        while self.cur.kind == "OP" and self.cur.value in {"+", "-"}:
            op = self.cur.value
            span = self.span()
            self.pos += 1
            expr = BinaryExpr(op, expr, self.parse_mul(), span)
        return expr

    def parse_mul(self) -> Expr:
        expr = self.parse_power()
        while (self.cur.kind == "OP" and self.cur.value in {"*", "/"}) or (
            self.cur.kind == "KEYWORD" and self.cur.value == "MOD"
        ):
            op = self.cur.value
            span = self.span()
            self.pos += 1
            expr = BinaryExpr(op, expr, self.parse_power(), span)
        return expr

    def parse_power(self) -> Expr:
        expr = self.parse_unary()
        if self.match("OP", "**"):
            span = self.span()
            expr = BinaryExpr("**", expr, self.parse_power(), span)
        return expr

    def parse_unary(self) -> Expr:
        if self.match("KEYWORD", "NOT"):
            return UnaryExpr("NOT", self.parse_unary(), self.span())
        if self.match("OP", "-"):
            return UnaryExpr("-", self.parse_unary(), self.span())
        if self.match("OP", "+"):
            return self.parse_unary()
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self.cur
        span = self.span()
        if tok.kind == "KEYWORD" and tok.value in {"TRUE", "FALSE"}:
            self.pos += 1
            return LiteralExpr(tok.value == "TRUE", "BOOL", span)
        if tok.kind == "INT":
            self.pos += 1
            return LiteralExpr(int(tok.value), "INT", span)
        if tok.kind == "REAL":
            self.pos += 1
            return LiteralExpr(float(tok.value), "REAL", span)
        if tok.kind == "TIME":
            self.pos += 1
            return LiteralExpr(parse_time_literal(tok.value, tok.line, tok.column), "TIME", span)
        if tok.kind == "IDENT":
            return self.parse_var_ref()
        if self.match("PUNCT", "("):
            inner = self.parse_expr()
            self.expect("PUNCT", ")")
            return inner
        raise PLCSyntaxError(f"Unexpected token {tok.value!r} in expression", tok.line, tok.column)


def parse_structured_text(source: str) -> ProgramNode:
    """Lex and parse an IEC 61131-3 Structured Text program into a :class:`ProgramNode`."""
    tokens = tokenize_structured_text(source)
    parser = _STParser(tokens)
    node = parser.parse_program()
    if parser.cur.kind != "EOF":
        raise PLCSyntaxError(
            f"Unexpected trailing token {parser.cur.value!r}",
            parser.cur.line,
            parser.cur.column,
        )
    return node


def parse_ladder_rungs(
    name: str,
    rung_specs: Sequence[Dict[str, object]],
    tag_types: Optional[Dict[str, PLCDataType]] = None,
) -> LadderProgram:
    """Compile structured Ladder Logic rung specifications into a :class:`LadderProgram`.

    Each rung dictionary accepts:
    - ``rung_id``: identifier string (e.g., ``"R001"``)
    - ``comment``: engineering description
    - ``branches``: list of parallel series branches; each branch is a list of element dicts:
      - ``{"type": "NO", "tag": "START_PB"}``
      - ``{"type": "NC", "tag": "E_STOP_TRIP"}``
      - ``{"type": "CMP", "tag": "PRESSURE_BAR", "op": "<=", "threshold": 3.5}``
    - ``coils``: list of output coils:
      - ``{"tag": "VALVE_INLET_CMD", "mode": "COIL" | "SET" | "RESET"}``
    """
    rungs: List[LadderRung] = []
    inferred_types: Dict[str, PLCDataType] = dict(tag_types or {})

    for idx, spec in enumerate(rung_specs):
        rung_id = str(spec.get("rung_id", f"R{idx + 1:03d}"))
        comment = str(spec.get("comment", ""))
        raw_branches = spec.get("branches", [])
        raw_coils = spec.get("coils", [])
        if not isinstance(raw_branches, list) or not raw_branches:
            raise PLCSyntaxError(f"Ladder rung {rung_id} must define non-empty branches")
        if not isinstance(raw_coils, list) or not raw_coils:
            raise PLCSyntaxError(f"Ladder rung {rung_id} must define at least one output coil")

        parsed_branches: List[Tuple[LadderElement, ...]] = []
        for branch in raw_branches:
            elems: List[LadderElement] = []
            for elem in branch:
                etype = str(elem["type"]).upper()
                tag = str(elem["tag"])
                if etype == "NO":
                    elems.append(ContactNO(tag=tag))
                    inferred_types.setdefault(tag, "BOOL")
                elif etype == "NC":
                    elems.append(ContactNC(tag=tag))
                    inferred_types.setdefault(tag, "BOOL")
                elif etype == "CMP":
                    op = str(elem["op"])
                    if op not in {"<", "<=", ">", ">=", "=", "<>"}:
                        raise PLCSyntaxError(f"Unsupported comparator operator {op!r} in {rung_id}")
                    threshold = float(elem["threshold"])
                    elems.append(ComparatorBlock(tag=tag, op=op, threshold=threshold))  # type: ignore[arg-type]
                    inferred_types.setdefault(tag, "REAL")
                else:
                    raise PLCSyntaxError(f"Unknown ladder element type {etype!r} in {rung_id}")
            parsed_branches.append(tuple(elems))

        parsed_coils: List[LadderCoil] = []
        for coil in raw_coils:
            tag = str(coil["tag"])
            mode = str(coil.get("mode", "COIL")).upper()
            if mode not in {"COIL", "SET", "RESET"}:
                raise PLCSyntaxError(f"Unknown coil mode {mode!r} in {rung_id}")
            parsed_coils.append(LadderCoil(tag=tag, mode=mode))  # type: ignore[arg-type]
            inferred_types.setdefault(tag, "BOOL")

        rungs.append(
            LadderRung(
                rung_id=rung_id,
                comment=comment,
                branches=tuple(parsed_branches),
                coils=tuple(parsed_coils),
            )
        )

    return LadderProgram(name=name, rungs=rungs, tag_types=inferred_types)
