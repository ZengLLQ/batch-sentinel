"""End-to-end integration tests for BatchSentinelPlant, CLI subcommands, and ASCII output."""

from __future__ import annotations

import json

import pytest

from batch_sentinel.cli import main as cli_main
from batch_sentinel.config import PlantScenarioConfig
from batch_sentinel.plant import BatchSentinelPlant
from batch_sentinel.visualizer import (
    render_ascii_series,
    render_bioreactor_ascii,
    render_ladder_ascii,
)


def test_full_plant_nominal_and_fault_campaigns() -> None:
    nom_res = BatchSentinelPlant(PlantScenarioConfig(batch_id="B-NOM", inject_fault="none")).run_campaign()
    assert nom_res.plc_verification.passed
    assert nom_res.ebr_valid
    assert nom_res.genealogy_report.balanced
    assert nom_res.serialization.verify_completeness()
    assert nom_res.batch_disposition == "RELEASED"

    press_res = BatchSentinelPlant(
        PlantScenarioConfig(batch_id="B-OVL", inject_fault="press_overload")
    ).run_campaign()
    assert press_res.pill_press.e_stop_triggered
    assert press_res.batch_disposition == "REJECTED"

    door_res = BatchSentinelPlant(
        PlantScenarioConfig(batch_id="B-DOOR", inject_fault="door_breach")
    ).run_campaign()
    assert door_res.aseptic_line.airlock_breach_halted
    assert door_res.batch_disposition == "REJECTED"

    cold_res = BatchSentinelPlant(
        PlantScenarioConfig(batch_id="B-COLD", inject_fault="cold_excursion")
    ).run_campaign()
    assert not cold_res.stability.within_release_spec
    assert cold_res.batch_disposition == "REJECTED"


def test_visualizer_emits_strict_7bit_ascii() -> None:
    plant = BatchSentinelPlant()
    res = plant.run_campaign()
    txt1 = render_bioreactor_ascii(res.bioreactor)
    txt2 = render_ladder_ascii(plant.ladder_program)
    txt3 = render_ascii_series("Constant", [("Const", [5.0, 5.0, 5.0], "*")])
    for txt in (txt1, txt2, txt3):
        txt.encode("ascii")


@pytest.mark.parametrize(
    "argv",
    [
        ["simulate", "--hours", "24", "--cryst-min", "60", "--tablets", "20", "--vials", "8", "--json"],
        ["verify-plc", "--scans", "6", "--json"],
        ["pat-audit", "--samples", "24", "--inject-drift", "--json"],
        ["trace-batch", "--json"],
        ["stability", "--excursion-hours", "8", "--json"],
        ["demo"],
    ],
)
def test_cli_subcommands_run_cleanly(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(argv)
    assert rc == 0
    captured = capsys.readouterr()
    # Ensure strict 7-bit ASCII stdout for Windows cp1252 safety
    captured.out.encode("ascii")
    if "--json" in argv:
        parsed = json.loads(captured.out)
        assert isinstance(parsed, dict)
