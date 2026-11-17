"""Example 04: End-to-End Integrated Plant Campaign & ICH Stability Release."""

from __future__ import annotations

from batch_sentinel.config import PlantScenarioConfig
from batch_sentinel.plant import BatchSentinelPlant


def main() -> None:
    nominal_cfg = PlantScenarioConfig(batch_id="BS-2026-NOMINAL", inject_fault="none")
    nominal_res = BatchSentinelPlant(nominal_cfg).run_campaign()
    print(
        f"[NOMINAL] Disposition={nominal_res.batch_disposition} | S88={nominal_res.s88_final_state.value} "
        f"| F0={nominal_res.sip_f0_minutes:.2f} min | Titer={nominal_res.bioreactor.final_titer_g_l:.3f} g/L "
        f"| Potency={nominal_res.stability.remaining_potency_pct:.3f}%"
    )

    excursion_cfg = PlantScenarioConfig(batch_id="BS-2026-COLD-EXC", inject_fault="cold_excursion")
    excursion_res = BatchSentinelPlant(excursion_cfg).run_campaign()
    print(
        f"[COLD EXCURSION] Disposition={excursion_res.batch_disposition} "
        f"| MKT={excursion_res.stability.mean_kinetic_temp_c:.2f} C "
        f"| t90_LCL={excursion_res.stability.t90_lower_95_days:.1f} days "
        f"| Reasons={excursion_res.disposition_reasons}"
    )


if __name__ == "__main__":
    main()
