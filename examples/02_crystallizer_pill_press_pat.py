"""Example 02: Cooling PBE Crystallizer, Rotary Tablet Press & PAT MSPC Control."""

from __future__ import annotations

import numpy as np

from batch_sentinel.pat import MultivariatePATMonitor, PLSQualityPredictor
from batch_sentinel.twins import CrystallizerTwin, PillPressTwin
from batch_sentinel.visualizer import render_pat_control_chart_ascii


def main() -> None:
    cryst = CrystallizerTwin()
    cubic_res = cryst.simulate_cooling_batch(duration_min=180.0, cooling_profile="cubic")
    linear_res = cryst.simulate_cooling_batch(duration_min=180.0, cooling_profile="linear")
    print(
        f"Cubic Cooling:  L43={cubic_res.final_l43_um:.1f} um | Yield={cubic_res.crystal_yield_kg:.2f} kg "
        f"| ClosureErr={cubic_res.solute_mass_closure_error_pct:.4f}%"
    )
    print(
        f"Linear Cooling: L43={linear_res.final_l43_um:.1f} um | Yield={linear_res.crystal_yield_kg:.2f} kg "
        f"| ClosureErr={linear_res.solute_mass_closure_error_pct:.4f}%"
    )

    press = PillPressTwin()
    nom_press = press.simulate_run(n_tablets=50)
    fault_press = press.simulate_run(n_tablets=50, inject_overload_at=25)
    print(
        f"\nNominal Press: MeanMass={nom_press.mean_mass_mg:.2f} mg | RSD={nom_press.mass_rsd_pct:.2f}% "
        f"| PeakF={nom_press.peak_force_kn:.2f} kN | EStop={nom_press.e_stop_triggered}"
    )
    print(
        f"Overload Press: Processed={len(fault_press.tablets)} | PeakF={fault_press.peak_force_kn:.2f} kN "
        f"| EStop={fault_press.e_stop_triggered} at tablet #{fault_press.e_stop_tablet_index}"
    )

    # Calibrate PAT PCA & PLS models on synthetic golden batches
    rng = np.random.default_rng(7)
    x_gold = rng.normal([1.05, 14.5, 4.68], [0.03, 0.55, 0.04], size=(80, 3))
    y_tensile = 3.2 + 0.18 * (x_gold[:, 1] - 14.5) + rng.normal(0.0, 0.05, size=80)

    mon = MultivariatePATMonitor(n_components=2, alpha=0.01).fit(
        x_gold, feature_names=["Supersat", "Force_kN", "Thickness_mm"]
    )
    pls = PLSQualityPredictor(n_components=2).fit(x_gold, y_tensile)

    x_batch = rng.normal([1.05, 14.5, 4.68], [0.03, 0.55, 0.04], size=(30, 3))
    x_batch[20:, 1] += 8.5  # Compaction force spike
    audit = mon.score(x_batch)
    pred_tensile = pls.predict(x_batch[:5])
    print(f"\nPLS Predicted Tensile (first 5 tablets, MPa): {[round(float(v), 2) for v in pred_tensile.ravel()]}")
    print(render_pat_control_chart_ascii(audit))


if __name__ == "__main__":
    main()
