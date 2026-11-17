"""Tests for IEC 61131-3 ST/Ladder lexer, parser, and static safety-interlock verifier."""

from __future__ import annotations

import pytest

from batch_sentinel.plant import (
    DEFAULT_INTERLOCK_RULES,
    DEFAULT_PLANT_LADDER_SPECS,
    DEFAULT_PLANT_PLC_ST,
)
from batch_sentinel.plc import (
    PLCStaticVerifier,
    PLCSyntaxError,
    SafetyInterlockRule,
    parse_ladder_rungs,
    parse_structured_text,
    parse_time_literal,
    tokenize_structured_text,
)


def test_time_literal_parsing() -> None:
    assert parse_time_literal("T#5s") == pytest.approx(5.0)
    assert parse_time_literal("T#250ms") == pytest.approx(0.25)
    assert parse_time_literal("TIME#1m30s") == pytest.approx(90.0)
    assert parse_time_literal("T#1h30m") == pytest.approx(5400.0)
    with pytest.raises(PLCSyntaxError):
        parse_time_literal("T#")
    with pytest.raises(PLCSyntaxError):
        parse_time_literal("5s")


def test_lexer_comments_and_tokens() -> None:
    src = """
    (* multi-line
       block comment *)
    PROGRAM TEST // line comment
    VAR X : REAL := 1.5e2; END_VAR
    X := X + 1.0;
    END_PROGRAM
    """
    tokens = tokenize_structured_text(src)
    kinds = [t.kind for t in tokens]
    assert "REAL" in kinds
    assert tokens[-1].kind == "EOF"

    with pytest.raises(PLCSyntaxError):
        tokenize_structured_text("PROGRAM BAD (* unterminated")


def test_parser_supports_if_case_for_and_fb() -> None:
    src = """
    PROGRAM CTRL
    VAR_INPUT STEP : INT := 1; GO : BOOL := TRUE; END_VAR
    VAR_OUTPUT OUT_VAL : INT := 0; END_VAR
    VAR I : INT := 0; T1 : TON; END_VAR
    T1(IN := GO, PT := T#100ms);
    CASE STEP OF
        1, 2:
            FOR I := 1 TO 4 BY 1 DO
                OUT_VAL := OUT_VAL + I;
            END_FOR;
        3:
            OUT_VAL := 99;
        ELSE
            OUT_VAL := -1;
    END_CASE;
    END_PROGRAM
    """
    prog = parse_structured_text(src)
    assert prog.name == "CTRL"
    assert len(prog.declarations) == 5
    report = PLCStaticVerifier().verify(prog)
    assert report.passed, report.findings


def test_verifier_rejects_type_mismatch_undeclared_and_unbounded_while() -> None:
    bad_src = """
    PROGRAM UNSAFE
    VAR_INPUT TRIP : BOOL := FALSE; END_VAR
    VAR_OUTPUT FLAG : BOOL := FALSE; END_VAR
    FLAG := 42;
    MISSING_TAG := TRUE;
    WHILE TRIP DO
        FLAG := FALSE;
    END_WHILE;
    END_PROGRAM
    """
    prog = parse_structured_text(bad_src)
    report = PLCStaticVerifier().verify(prog)
    assert not report.passed
    codes = {f.code for f in report.findings}
    assert "TYPE_MISMATCH" in codes
    assert "UNDECLARED_VAR" in codes
    assert "UNBOUNDED_WHILE" in codes


def test_verifier_proves_default_plant_interlocks_and_catches_buggy_program() -> None:
    prog = parse_structured_text(DEFAULT_PLANT_PLC_ST)
    ladder = parse_ladder_rungs(
        "L1",
        DEFAULT_PLANT_LADDER_SPECS,
        tag_types={"MASTER_RUN_PERMIT": "BOOL", "PRESSURE_BAR": "REAL"},
    )
    report = PLCStaticVerifier().verify(prog, ladder=ladder, interlock_rules=DEFAULT_INTERLOCK_RULES)
    assert report.passed
    assert report.checked_combinations > 10

    buggy_src = """
    PROGRAM BUGGY_SKID
    VAR_INPUT E_STOP_TRIP : BOOL := FALSE; START_CMD : BOOL := FALSE; END_VAR
    VAR_OUTPUT PUNCH_ENABLE : BOOL := FALSE; END_VAR
    (* Bug: forgets to check E_STOP_TRIP! *)
    IF START_CMD THEN
        PUNCH_ENABLE := TRUE;
    END_IF;
    END_PROGRAM
    """
    buggy_prog = parse_structured_text(buggy_src)
    rule = SafetyInterlockRule("ESTOP_PUNCH", ("E_STOP_TRIP",), "PUNCH_ENABLE", True)
    bad_report = PLCStaticVerifier().verify(buggy_prog, interlock_rules=(rule,))
    assert not bad_report.passed
    assert any(f.code == "INTERLOCK_VIOLATION" for f in bad_report.findings)
