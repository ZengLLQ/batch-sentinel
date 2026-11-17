"""IEC 61131-3 Structured Text lexer with time-literal and comment support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


KEYWORDS = frozenset({
    "PROGRAM",
    "END_PROGRAM",
    "VAR",
    "VAR_INPUT",
    "VAR_OUTPUT",
    "VAR_IN_OUT",
    "END_VAR",
    "IF",
    "THEN",
    "ELSIF",
    "ELSE",
    "END_IF",
    "CASE",
    "OF",
    "END_CASE",
    "FOR",
    "TO",
    "BY",
    "DO",
    "END_FOR",
    "WHILE",
    "END_WHILE",
    "AND",
    "OR",
    "XOR",
    "NOT",
    "MOD",
    "TRUE",
    "FALSE",
})


class PLCSyntaxError(SyntaxError):
    """Raised when IEC 61131-3 Structured Text or Ladder syntax is invalid."""

    def __init__(self, message: str, line: int = 1, column: int = 1) -> None:
        super().__init__(f"[line {line}, col {column}] {message}")
        self.line = line
        self.column = column


@dataclass(frozen=True)
class Token:
    """Lexical token produced by :func:`tokenize_structured_text`."""

    kind: str  # KEYWORD, IDENT, INT, REAL, TIME, OP, PUNCT, EOF
    value: str
    line: int
    column: int


def parse_time_literal(text: str, line: int = 1, column: int = 1) -> float:
    """Convert an IEC 61131-3 time literal (e.g., ``T#5s``, ``T#1m250ms``) to seconds."""
    upper = text.upper()
    if upper.startswith("TIME#"):
        body = upper[5:]
    elif upper.startswith("T#"):
        body = upper[2:]
    else:
        raise PLCSyntaxError(f"Invalid TIME literal: {text!r}", line, column)
    if not body:
        raise PLCSyntaxError(f"Empty TIME literal: {text!r}", line, column)

    total_seconds = 0.0
    idx = 0
    unit_multipliers = {
        "MS": 0.001,
        "S": 1.0,
        "M": 60.0,
        "H": 3600.0,
        "D": 86400.0,
    }
    while idx < len(body):
        num_start = idx
        seen_dot = False
        while idx < len(body) and (body[idx].isdigit() or (body[idx] == "." and not seen_dot)):
            if body[idx] == ".":
                seen_dot = True
            idx += 1
        if idx == num_start:
            raise PLCSyntaxError(f"Expected number in TIME literal {text!r}", line, column)
        magnitude = float(body[num_start:idx])

        unit_start = idx
        while idx < len(body) and body[idx].isalpha():
            idx += 1
        unit = body[unit_start:idx]
        if unit not in unit_multipliers:
            raise PLCSyntaxError(f"Unknown TIME unit {unit!r} in {text!r}", line, column)
        total_seconds += magnitude * unit_multipliers[unit]
        if idx < len(body) and body[idx] == "_":
            idx += 1
    return total_seconds


def tokenize_structured_text(source: str) -> List[Token]:
    """Tokenize an IEC 61131-3 Structured Text source string into :class:`Token` objects."""
    tokens: List[Token] = []
    n = len(source)
    i = 0
    line = 1
    col = 1

    def advance(steps: int = 1) -> None:
        nonlocal i, line, col
        for _ in range(steps):
            if i < n and source[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        ch = source[i]
        if ch.isspace():
            advance()
            continue

        # Block comments (* ... *)
        if source.startswith("(*", i):
            start_line, start_col = line, col
            advance(2)
            while i < n and not source.startswith("*)", i):
                advance()
            if i >= n:
                raise PLCSyntaxError("Unterminated (* ... *) comment", start_line, start_col)
            advance(2)
            continue

        # Line comments // ...
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                advance()
            continue

        start_line, start_col = line, col

        # IEC TIME literals: T#... or TIME#...
        if (source[i:i + 2].upper() == "T#" or source[i:i + 5].upper() == "TIME#"):
            buf = []
            while i < n and (source[i].isalnum() or source[i] in "#._"):
                buf.append(source[i])
                advance()
            literal = "".join(buf)
            parse_time_literal(literal, start_line, start_col)
            tokens.append(Token("TIME", literal, start_line, start_col))
            continue

        # Multi-character operators
        two_char = source[i:i + 2]
        if two_char in {":=", "<=", ">=", "<>", "**"}:
            tokens.append(Token("OP", two_char, start_line, start_col))
            advance(2)
            continue

        # Single-character operators and punctuation
        if ch in {"+", "-", "*", "/", "=", "<", ">"}:
            tokens.append(Token("OP", ch, start_line, start_col))
            advance()
            continue
        if ch in {"(", ")", ",", ";", ":", "."}:
            tokens.append(Token("PUNCT", ch, start_line, start_col))
            advance()
            continue

        # Numeric literals (INT or REAL)
        if ch.isdigit():
            buf = []
            is_real = False
            while i < n and (source[i].isdigit() or source[i] == "_"):
                if source[i] != "_":
                    buf.append(source[i])
                advance()
            if i < n and source[i] == "." and (i + 1 < n and source[i + 1].isdigit()):
                is_real = True
                buf.append(".")
                advance()
                while i < n and (source[i].isdigit() or source[i] == "_"):
                    if source[i] != "_":
                        buf.append(source[i])
                    advance()
            if i < n and source[i] in ("e", "E"):
                is_real = True
                buf.append(source[i])
                advance()
                if i < n and source[i] in ("+", "-"):
                    buf.append(source[i])
                    advance()
                if i >= n or not source[i].isdigit():
                    raise PLCSyntaxError("Malformed scientific notation literal", start_line, start_col)
                while i < n and source[i].isdigit():
                    buf.append(source[i])
                    advance()
            tokens.append(Token("REAL" if is_real else "INT", "".join(buf), start_line, start_col))
            continue

        # Identifiers and keywords
        if ch.isalpha() or ch == "_":
            buf = []
            while i < n and (source[i].isalnum() or source[i] == "_"):
                buf.append(source[i])
                advance()
            word = "".join(buf)
            upper = word.upper()
            if upper in KEYWORDS:
                tokens.append(Token("KEYWORD", upper, start_line, start_col))
            else:
                tokens.append(Token("IDENT", word, start_line, start_col))
            continue

        raise PLCSyntaxError(f"Unexpected character {ch!r}", start_line, start_col)

    tokens.append(Token("EOF", "", line, col))
    return tokens
