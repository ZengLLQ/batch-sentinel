"""Rotary tablet pill-press Heckel compaction and closed-loop die-fill digital twin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import PillPressParams


@dataclass(frozen=True)
class TabletRecord:
    """Per-tablet compaction mechanics and USP <905> quality outcome."""

    tablet_index: int
    station_index: int
    fill_depth_mm: float
    mass_mg: float
    peak_pressure_mpa: float
    peak_force_kn: float
    relative_density: float
    porosity: float
    tensile_strength_mpa: float
    thickness_mm: float
    accepted: bool
    rejection_reason: str


@dataclass
class PillPressRunResult:
    """Summary of a rotary tablet press simulation run."""

    tablets: List[TabletRecord]
    mean_mass_mg: float
    mass_rsd_pct: float
    mean_tensile_mpa: float
    peak_force_kn: float
    acceptance_rate_pct: float
    e_stop_triggered: bool
    e_stop_tablet_index: Optional[int]


class PillPressTwin:
    """Rotary tablet press simulator coupling Heckel/Kawakita powder compaction and die-fill control.

    Physics implemented:
    - **Heckel equation**: ``ln(1 / (1 - D)) = K * P + A`` -> ``D(P) = 1 - exp(-(K * P + A))``
    - **Kawakita compressibility**: ``C = (V0 - V) / V0 = (a * b * P) / (1 + b * P)``
    - **Ryshkewitch-Duckworth tensile strength**: ``sigma_t = sigma_0 * exp(-b_r * (1 - D))``
    - **Elastic frame deflection**: ``dz_frame = F / k_frame`` altering true in-die minimum thickness.
    """

    def __init__(
        self,
        params: Optional[PillPressParams] = None,
        bulk_density_g_cm3: float = 0.62,
        target_thickness_mm: float = 4.65,
        seed: int = 17,
    ) -> None:
        self.params = params or PillPressParams()
        self.bulk_density_g_cm3 = bulk_density_g_cm3
        self.target_thickness_mm = target_thickness_mm
        self._rng = np.random.default_rng(seed)

    def heckel_relative_density(self, pressure_mpa: float) -> float:
        """Compute solid fraction ``D in (0, 1)`` from compaction pressure ``P`` (MPa)."""
        p = self.params
        exponent = -(p.k_heckel * max(0.0, pressure_mpa) + p.a_heckel)
        d = 1.0 - math.exp(exponent)
        return float(min(0.995, max(0.35, d)))

    def heckel_pressure_for_density(self, relative_density: float) -> float:
        """Invert Heckel equation to solve compaction pressure ``P`` (MPa) for target ``D``."""
        p = self.params
        d_clamped = min(0.992, max(0.45, relative_density))
        lhs = math.log(1.0 / (1.0 - d_clamped))
        return max(0.0, (lhs - p.a_heckel) / p.k_heckel)

    def kawakita_volume_reduction(self, pressure_mpa: float) -> float:
        """Compute Kawakita engineering strain ``(V0 - V) / V0`` at pressure ``P``."""
        p = self.params
        pr = max(0.0, pressure_mpa)
        return (p.kawakita_a * p.kawakita_b * pr) / (1.0 + p.kawakita_b * pr)

    def tensile_strength_mpa(self, relative_density: float) -> float:
        """Compute Ryshkewitch-Duckworth diametral tensile strength (MPa)."""
        p = self.params
        porosity = max(0.0, 1.0 - relative_density)
        return p.sigma0_mpa * math.exp(-p.b_ryshkewitch * porosity)

    def simulate_run(
        self,
        n_tablets: int = 60,
        inject_overload_at: Optional[int] = None,
        bulk_density_drift: float = 0.0,
    ) -> PillPressRunResult:
        """Simulate rotary compression of ``n_tablets`` with closed-loop weight control."""
        p = self.params
        # Initial die-fill depth: mass (g) = rho_bulk (g/cm^3) * area (cm^2) * depth (cm)
        area_cm2 = p.tablet_area_mm2 / 100.0
        nominal_fill_mm = 10.0 * (p.target_mass_mg / 1000.0) / (self.bulk_density_g_cm3 * area_cm2)
        fill_depth_mm = nominal_fill_mm
        integral_err_mm = 0.0

        tablets: List[TabletRecord] = []
        e_stop_triggered = False
        e_stop_idx: Optional[int] = None

        for idx in range(n_tablets):
            station = (idx % p.n_stations) + 1
            # Powder hopper bulk density variation + optional systematic drift
            frac = idx / max(1, n_tablets - 1)
            local_bulk_rho = self.bulk_density_g_cm3 * (1.0 + bulk_density_drift * frac) + float(
                self._rng.normal(0.0, 0.009)
            )
            local_bulk_rho = max(0.35, local_bulk_rho)

            if inject_overload_at is not None and idx >= inject_overload_at:
                # Double-fill / slugging jam doubles powder charge inside die
                mass_mg = 1000.0 * local_bulk_rho * area_cm2 * (fill_depth_mm / 10.0) * 1.95
            else:
                mass_mg = 1000.0 * local_bulk_rho * area_cm2 * (fill_depth_mm / 10.0)

            # Solve coupled Heckel density + elastic turret frame deflection iteratively
            # True solid volume (mm^3) = (mass_mg / 1000) / rho_true * 1000
            v_solid_mm3 = mass_mg / p.rho_true_g_cm3
            nominal_in_die_vol_mm3 = p.tablet_area_mm2 * self.target_thickness_mm
            d_est = min(0.99, max(0.45, v_solid_mm3 / nominal_in_die_vol_mm3))
            pressure_mpa = self.heckel_pressure_for_density(d_est)
            for _ in range(6):
                force_kn = pressure_mpa * p.tablet_area_mm2 / 1000.0
                deflection_mm = force_kn / p.frame_stiffness_kn_mm
                eff_thick_mm = max(1.2, self.target_thickness_mm + deflection_mm)
                d_raw = v_solid_mm3 / (p.tablet_area_mm2 * eff_thick_mm)
                if d_raw > 0.99:
                    # Overfill solid-wall compression: excess powder compresses machine frame directly
                    excess_thick_mm = (v_solid_mm3 / (0.99 * p.tablet_area_mm2)) - self.target_thickness_mm
                    wall_force_kn = max(0.0, excess_thick_mm * p.frame_stiffness_kn_mm)
                    pressure_mpa = max(
                        self.heckel_pressure_for_density(0.99),
                        1000.0 * wall_force_kn / p.tablet_area_mm2,
                    )
                else:
                    d_est = max(0.45, d_raw)
                    pressure_mpa = 0.5 * pressure_mpa + 0.5 * self.heckel_pressure_for_density(d_est)

            force_kn = pressure_mpa * p.tablet_area_mm2 / 1000.0
            deflection_mm = force_kn / p.frame_stiffness_kn_mm
            thickness_mm = self.target_thickness_mm + deflection_mm
            rel_density = self.heckel_relative_density(pressure_mpa)
            porosity = 1.0 - rel_density
            tensile = self.tensile_strength_mpa(rel_density)

            overload = force_kn > p.max_force_kn
            mass_err_pct = 100.0 * abs(mass_mg - p.target_mass_mg) / p.target_mass_mg
            accepted = (not overload) and (mass_err_pct <= p.mass_tolerance_pct) and (tensile >= 1.5)

            reason = "OK"
            if overload:
                reason = f"PUNCH_OVERLOAD ({force_kn:.1f} kN > {p.max_force_kn:.1f} kN)"
                e_stop_triggered = True
                e_stop_idx = idx
            elif mass_err_pct > p.mass_tolerance_pct:
                reason = f"MASS_OOS ({mass_mg:.1f} mg, err={mass_err_pct:.2f}%)"
            elif tensile < 1.5:
                reason = f"LOW_TENSILE_STRENGTH ({tensile:.2f} MPa < 1.50 MPa)"

            tablets.append(
                TabletRecord(
                    tablet_index=idx,
                    station_index=station,
                    fill_depth_mm=fill_depth_mm,
                    mass_mg=mass_mg,
                    peak_pressure_mpa=pressure_mpa,
                    peak_force_kn=force_kn,
                    relative_density=rel_density,
                    porosity=porosity,
                    tensile_strength_mpa=tensile,
                    thickness_mm=thickness_mm,
                    accepted=accepted,
                    rejection_reason=reason,
                )
            )

            if e_stop_triggered:
                break

            # Closed-loop PI weight-control adjustment on dosing cam fill depth
            mass_err_mg = p.target_mass_mg - mass_mg
            integral_err_mm += 0.0008 * mass_err_mg
            integral_err_mm = max(-1.5, min(1.5, integral_err_mm))
            fill_depth_mm = max(
                4.0,
                min(18.0, fill_depth_mm + 0.0045 * mass_err_mg + 0.25 * integral_err_mm),
            )

        masses = np.array([t.mass_mg for t in tablets], dtype=np.float64)
        mean_mass = float(np.mean(masses))
        mass_rsd = float(100.0 * np.std(masses, ddof=1) / mean_mass) if len(masses) > 1 else 0.0
        mean_tensile = float(np.mean([t.tensile_strength_mpa for t in tablets]))
        peak_force = float(max(t.peak_force_kn for t in tablets))
        accept_rate = 100.0 * sum(1 for t in tablets if t.accepted) / len(tablets)

        return PillPressRunResult(
            tablets=tablets,
            mean_mass_mg=mean_mass,
            mass_rsd_pct=mass_rsd,
            mean_tensile_mpa=mean_tensile,
            peak_force_kn=peak_force,
            acceptance_rate_pct=accept_rate,
            e_stop_triggered=e_stop_triggered,
            e_stop_tablet_index=e_stop_idx,
        )
