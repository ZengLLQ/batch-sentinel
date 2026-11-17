"""Configuration dataclasses and physical unit validation for BatchSentinel.

Every parameter carries explicit SI or pharmaceutical engineering units and
enforces thermodynamic, kinetic, or safety bounds during instantiation so
downstream ODE integrators and PLC scan loops cannot silently accept
non-physical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


def _require_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value!r}")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


@dataclass(frozen=True)
class PLCScanConfig:
    """Deterministic IEC 61131-3 OB1 scan-cycle timing configuration.

    Attributes:
        scan_period_s: Fixed cyclic interrupt period in seconds (typically 0.01 - 0.5 s).
        watchdog_timeout_s: Maximum allowed execution time per scan before watchdog fault.
        jitter_std_s: Simulated Gaussian scan execution jitter standard deviation (s).
        max_loop_iterations: Static upper bound enforced on FOR/WHILE loops.
        seed: Random seed for reproducible jitter and sensor noise simulation.
    """

    scan_period_s: float = 0.05
    watchdog_timeout_s: float = 0.045
    jitter_std_s: float = 0.001
    max_loop_iterations: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        _require_positive(self.scan_period_s, "scan_period_s")
        _require_positive(self.watchdog_timeout_s, "watchdog_timeout_s")
        _require_non_negative(self.jitter_std_s, "jitter_std_s")
        if self.max_loop_iterations < 1:
            raise ValueError("max_loop_iterations must be >= 1")


@dataclass(frozen=True)
class BioreactorParams:
    """Kinetic, mass-transfer, and thermal parameters for a fed-batch/perfusion bioreactor.

    Units follow standard biochemical engineering conventions:
    - Biomass X, substrate S, lactate L_inh, mAb product P: g/L
    - Dissolved oxygen C_O2: mg/L (saturation ~ 7.2 mg/L at 37 C)
    - Volume V: L
    - Temperature T_c: degrees Celsius
    - Time: hours (h)
    """

    mu_max: float = 0.048              # 1/h maximum specific growth rate (CHO cells)
    k_s: float = 0.35                  # g/L Monod half-saturation constant for glucose
    k_i_sub: float = 18.0              # g/L Haldane substrate inhibition constant
    k_o2: float = 0.25                 # mg/L dissolved oxygen Monod constant
    k_i_lac: float = 4.5               # g/L lactate growth inhibition constant
    k_d_base: float = 0.004            # 1/h basal cell death rate
    y_xs: float = 0.52                 # g biomass / g glucose yield
    m_s: float = 0.018                 # g glucose / (g biomass * h) maintenance coefficient
    y_lac_s: float = 0.42              # g lactate / g glucose byproduct yield
    q_p_max: float = 0.024             # g mAb / (g biomass * h) specific productivity
    alpha_growth_prod: float = 0.12    # growth-associated product coefficient
    q_o2_spec: float = 0.14            # g O2 / (g biomass * h) specific oxygen uptake
    c_o2_star: float = 7.2             # mg/L oxygen saturation concentration at 37 C
    kla_base: float = 48.0             # 1/h baseline volumetric mass transfer coefficient
    s_feed: float = 250.0              # g/L concentrated glucose feed
    v_max: float = 2000.0              # L working vessel capacity
    t_opt_c: float = 37.0              # degC optimal culture temperature
    ua_jacket: float = 420.0           # W/K jacket heat transfer UA
    cp_broth: float = 4180.0           # J/(kg*K) broth heat capacity
    rho_broth: float = 1.02            # kg/L broth density
    delta_h_met: float = 14500.0       # J/g glucose metabolic heat release
    buffer_capacity: float = 0.045     # mol/(L*pH) Henderson-Hasselbalch effective capacity
    base_conc_mol_l: float = 1.0       # mol/L Na2CO3/NaOH base titrant concentration

    def __post_init__(self) -> None:
        _require_positive(self.mu_max, "mu_max")
        _require_positive(self.k_s, "k_s")
        _require_positive(self.k_i_sub, "k_i_sub")
        _require_positive(self.y_xs, "y_xs")
        _require_positive(self.v_max, "v_max")
        _require_positive(self.c_o2_star, "c_o2_star")
        _require_positive(self.buffer_capacity, "buffer_capacity")


@dataclass(frozen=True)
class PillPressParams:
    """Rotary tablet press Heckel/Kawakita powder compaction and turret parameters.

    Attributes:
        k_heckel: Heckel slope constant (1/MPa), inverse of mean yield pressure P_y.
        a_heckel: Heckel intercept representing initial die packing ln(1/(1 - D0)).
        sigma0_mpa: Zero-porosity Ryshkewitch-Duckworth tensile strength (MPa).
        b_ryshkewitch: Porosity sensitivity exponent (dimensionless).
        kawakita_a: Kawakita maximum volume reduction ratio (dimensionless, < 1).
        kawakita_b: Kawakita reciprocal pressure constant (1/MPa).
        tablet_area_mm2: Tablet cross-sectional punch face area (mm^2).
        die_diameter_mm: Die bore diameter (mm).
        rho_true_g_cm3: True crystalline density of powder blend (g/cm^3).
        target_mass_mg: Target tablet mass (mg).
        mass_tolerance_pct: USP <905> relative weight acceptance band (%).
        max_force_kn: Hardware punch overload E-stop threshold (kN).
        frame_stiffness_kn_mm: Turret/column elastic frame stiffness (kN/mm).
        n_stations: Number of punch stations around rotary turret.
    """

    k_heckel: float = 0.0085
    a_heckel: float = 0.58
    sigma0_mpa: float = 9.4
    b_ryshkewitch: float = 6.2
    kawakita_a: float = 0.42
    kawakita_b: float = 0.032
    tablet_area_mm2: float = 78.54
    die_diameter_mm: float = 10.0
    rho_true_g_cm3: float = 1.48
    target_mass_mg: float = 500.0
    mass_tolerance_pct: float = 5.0
    max_force_kn: float = 38.0
    frame_stiffness_kn_mm: float = 140.0
    n_stations: int = 36

    def __post_init__(self) -> None:
        _require_positive(self.k_heckel, "k_heckel")
        _require_positive(self.tablet_area_mm2, "tablet_area_mm2")
        _require_positive(self.rho_true_g_cm3, "rho_true_g_cm3")
        _require_positive(self.target_mass_mg, "target_mass_mg")
        _require_positive(self.max_force_kn, "max_force_kn")
        if not (0.0 < self.kawakita_a < 1.0):
            raise ValueError("kawakita_a must lie strictly in (0, 1)")
        if self.n_stations < 1:
            raise ValueError("n_stations must be >= 1")


@dataclass(frozen=True)
class CrystallizerParams:
    """Population balance (Method of Moments) cooling crystallizer parameters.

    Solubility curve: C*(T) = sol_a + sol_b * exp(sol_c * T_c) in g solute / kg solvent.
    """

    sol_a: float = 12.0               # g/kg baseline solubility
    sol_b: float = 18.0               # g/kg pre-exponential solubility term
    sol_c: float = 0.032              # 1/degC exponential thermal coefficient
    k_b: float = 2.8e9                # #/(kg_solvent * min) nucleation rate constant
    b_nuc: float = 2.2                # nucleation supersaturation exponent
    j_magma: float = 0.65             # secondary nucleation magma exponent on mu_3
    k_g: float = 42.0                 # um/min crystal growth rate constant
    g_growth: float = 1.35            # growth supersaturation exponent
    k_d_fines: float = 18.0           # um/min dissolution rate when undersaturated
    rho_crystal: float = 1350.0       # kg/m^3 crystal density
    k_v_shape: float = 0.5236         # volume shape factor (pi/6 for spheres)
    solvent_mass_kg: float = 250.0    # kg solvent charge
    ua_jacket_w_k: float = 850.0      # W/K cooling jacket UA
    heat_cap_j_k: float = 650000.0    # J/K total vessel thermal capacitance

    def __post_init__(self) -> None:
        _require_positive(self.k_b, "k_b")
        _require_positive(self.k_g, "k_g")
        _require_positive(self.rho_crystal, "rho_crystal")
        _require_positive(self.solvent_mass_kg, "solvent_mass_kg")


@dataclass(frozen=True)
class FillFinishParams:
    """Aseptic fill-finish cleanroom and 6-DOF robotics parameters."""

    target_fill_ml: float = 10.0
    fill_tolerance_ml: float = 0.25
    pump_flow_ml_s: float = 4.5
    iso5_min_dp_pa: float = 15.0      # Minimum Grade A / ISO 5 differential pressure (Pa)
    nominal_dp_pa: float = 22.0       # Target cleanroom positive pressure differential (Pa)
    hvac_supply_gain: float = 1.8     # Pa/s nominal HVAC pressurization gain
    leak_tau_s: float = 12.0          # s enclosure leakage time constant
    door_leak_conductance: float = 0.45  # 1/s pressure bleed when airlock door opens
    max_joint_vel_rad_s: float = 2.6
    ik_damping: float = 0.025
    ik_max_iters: int = 80
    ik_tol_m: float = 1e-4

    def __post_init__(self) -> None:
        _require_positive(self.target_fill_ml, "target_fill_ml")
        _require_positive(self.iso5_min_dp_pa, "iso5_min_dp_pa")
        _require_positive(self.ik_tol_m, "ik_tol_m")


@dataclass(frozen=True)
class PATConfig:
    """Process Analytical Technology (PAT) PCA/PLS statistical control parameters."""

    n_components: int = 3
    alpha: float = 0.01               # Significance level for Hotelling T^2 & SPE UCL (99%)
    standardize: bool = True

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if not (0.0 < self.alpha < 0.5):
            raise ValueError("alpha must lie strictly in (0, 0.5)")


@dataclass(frozen=True)
class StabilityConfig:
    """ICH Q1A / Q1E Arrhenius stability & Mean Kinetic Temperature configuration."""

    pre_exponential_per_day: float = 1.25e11  # 1/day Arrhenius A
    activation_energy_j_mol: float = 83144.0  # J/mol (classic ICH MKT ~83.144 kJ/mol)
    reaction_order: int = 1                   # 0 = zero-order, 1 = first-order
    release_spec_pct: float = 90.0            # Potency shelf-life threshold (% label claim)
    storage_target_c: float = 5.0             # Cold-chain nominal storage temperature (degC)

    def __post_init__(self) -> None:
        _require_positive(self.pre_exponential_per_day, "pre_exponential_per_day")
        _require_positive(self.activation_energy_j_mol, "activation_energy_j_mol")
        if self.reaction_order not in (0, 1):
            raise ValueError("reaction_order must be 0 or 1")
        if not (50.0 <= self.release_spec_pct < 100.0):
            raise ValueError("release_spec_pct must be in [50, 100)")


@dataclass
class PlantScenarioConfig:
    """Top-level configuration aggregating an end-to-end GMP manufacturing campaign."""

    batch_id: str = "BS-MAB-2026-001"
    product_name: str = "Recombinant mAb-X101"
    gtin14: str = "00312345678906"
    bioreactor_hours: float = 120.0
    crystallizer_minutes: float = 180.0
    pill_press_tablets: int = 60
    vial_count: int = 24
    inject_fault: str = "none"        # none | do_drop | press_overload | door_breach | cold_excursion
    plc: PLCScanConfig = field(default_factory=PLCScanConfig)
    bioreactor: BioreactorParams = field(default_factory=BioreactorParams)
    pill_press: PillPressParams = field(default_factory=PillPressParams)
    crystallizer: CrystallizerParams = field(default_factory=CrystallizerParams)
    fill_finish: FillFinishParams = field(default_factory=FillFinishParams)
    pat: PATConfig = field(default_factory=PATConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    metadata: Dict[str, str] = field(default_factory=dict)
    operator_ids: List[str] = field(default_factory=lambda: ["OP-QA-104", "OP-MFG-219"])
