"""Pure 7-bit ASCII terminal visualizer for trajectories, PLC ladders, and PAT charts.

All rendered glyphs are strictly constrained to printable 7-bit ASCII codes
(32-126 plus newline) so Windows cp1252 and headless CI terminals never fail
with ``UnicodeEncodeError``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from .pat.multivariate import PATBatchAudit
from .plc.ast_nodes import ComparatorBlock, ContactNC, ContactNO, LadderProgram
from .twins.bioreactor import BioreactorTrajectory


def _assert_ascii(text: str) -> str:
    text.encode("ascii")
    return text


def render_ascii_series(
    title: str,
    series_list: Sequence[Tuple[str, Sequence[float], str]],
    width: int = 64,
    height: int = 12,
    y_label: str = "Val",
) -> str:
    """Render one or more numeric sequences on a normalized 7-bit ASCII canvas."""
    if not series_list or all(len(vals) == 0 for _, vals, _ in series_list):
        return f"{title}\n  [no data]\n"

    all_vals = [float(v) for _, vals, _ in series_list for v in vals]
    y_min = min(all_vals)
    y_max = max(all_vals)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    canvas = [[" " for _ in range(width)] for _ in range(height)]

    for _, vals, marker in series_list:
        ch = marker[0] if marker else "*"
        n = len(vals)
        for col in range(width):
            idx = min(n - 1, int(round(col * (n - 1) / max(1, width - 1))))
            norm = (float(vals[idx]) - y_min) / (y_max - y_min)
            row = height - 1 - int(round(norm * (height - 1)))
            row = max(0, min(height - 1, row))
            canvas[row][col] = ch

    lines = [f"=== {title} ==="]
    for r_idx, row_chars in enumerate(canvas):
        val_at_row = y_max - (r_idx / max(1, height - 1)) * (y_max - y_min)
        prefix = f"{val_at_row:8.2f} |"
        lines.append(prefix + "".join(row_chars))
    lines.append(" " * 9 + "+" + "-" * width)
    legend = ", ".join(f"[{m[0]}]={name}" for name, _, m in series_list)
    lines.append(f"  {y_label} range [{y_min:.2f}, {y_max:.2f}] | {legend}")
    return _assert_ascii("\n".join(lines))


def render_bioreactor_ascii(traj: BioreactorTrajectory) -> str:
    """Render bioreactor Biomass, Product mAb, and Dissolved Oxygen curves in ASCII."""
    x_vals = [s.biomass_g_l for s in traj.states]
    p_vals = [s.product_g_l for s in traj.states]
    do_vals = [s.dissolved_o2_mg_l for s in traj.states]
    return render_ascii_series(
        "Bioreactor Fed-Batch/Perfusion Profile (g/L & mg/L)",
        [
            ("Biomass X (g/L)", x_vals, "*"),
            ("Product mAb (g/L)", p_vals, "#"),
            ("DO (mg/L)", do_vals, "."),
        ],
        width=60,
        height=12,
        y_label="Conc",
    )


def render_ladder_ascii(ladder: LadderProgram) -> str:
    """Render an IEC 61131-3 Ladder Logic program as a 7-bit ASCII schematic."""
    lines = [f"=== Ladder Program: {ladder.name} ==="]
    for rung in ladder.rungs:
        lines.append(f"[{rung.rung_id}] {rung.comment}")
        coil_str = " ".join(
            f"({c.mode[0] if c.mode != 'COIL' else ' '}:{c.tag})" for c in rung.coils
        )
        for b_idx, branch in enumerate(rung.branches):
            parts = []
            for elem in branch:
                if isinstance(elem, ContactNO):
                    parts.append(f"--[ {elem.tag} ]")
                elif isinstance(elem, ContactNC):
                    parts.append(f"--[/{elem.tag}]")
                elif isinstance(elem, ComparatorBlock):
                    parts.append(f"--[{elem.tag} {elem.op} {elem.threshold}]")
            left = "|+" if b_idx > 0 else "|-"
            right = f"--{coil_str}--|" if b_idx == 0 else "--+"
            lines.append(f"  {left}{''.join(parts)}{right}")
    return _assert_ascii("\n".join(lines))


def render_pat_control_chart_ascii(audit: PATBatchAudit) -> str:
    """Render Hotelling T^2 & SPE control chart summary with 99% UCL markers."""
    t2_vals = [s.t2 for s in audit.samples]
    spe_vals = [s.spe for s in audit.samples]
    t2_plot = render_ascii_series(
        f"PAT Hotelling T^2 (UCL={audit.t2_ucl:.2f}, Exc={audit.t2_exceedance_pct:.1f}%)",
        [("T2", t2_vals, "*"), ("UCL", [audit.t2_ucl] * len(t2_vals), "-")],
        width=58,
        height=10,
        y_label="T2",
    )
    spe_plot = render_ascii_series(
        f"PAT Q-Residual SPE (UCL={audit.spe_ucl:.2f}, Exc={audit.spe_exceedance_pct:.1f}%)",
        [("SPE", spe_vals, "#"), ("UCL", [audit.spe_ucl] * len(spe_vals), "-")],
        width=58,
        height=10,
        y_label="SPE",
    )
    return _assert_ascii(t2_plot + "\n\n" + spe_plot)
