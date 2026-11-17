"""BatchSentinel: Automated pharmaceutical manufacturing, PLC verification, and GMP traceability."""

from .config import (
    BioreactorParams,
    CrystallizerParams,
    FillFinishParams,
    PATConfig,
    PLCScanConfig,
    PillPressParams,
    PlantScenarioConfig,
    StabilityConfig,
)
from .gmp import (
    ElectronicBatchRecord,
    LotGenealogyDAG,
    MaterialLot,
    SerializationHierarchy,
    StabilityModel,
    build_merkle_proof,
    compute_gs1_check_digit,
    compute_merkle_root,
    validate_gtin14,
    verify_merkle_proof,
)
from .isa88 import (
    ISA88BatchExecutive,
    S88Operation,
    S88Phase,
    S88Procedure,
    S88State,
    S88UnitProcedure,
    compute_sip_f0_lethality,
)
from .pat import MultivariatePATMonitor, PLSQualityPredictor
from .plant import BatchSentinelPlant, PlantCampaignResult
from .plc import (
    PLCRuntime,
    PLCStaticVerifier,
    SafetyInterlockRule,
    parse_ladder_rungs,
    parse_structured_text,
)
from .twins import (
    AsepticFillLineTwin,
    BioreactorTwin,
    CrystallizerTwin,
    PillPressTwin,
    SixDOFRobotArm,
    integrate_rkf45,
)

__version__ = "0.1.0"

__all__ = [
    "AsepticFillLineTwin",
    "BatchSentinelPlant",
    "BioreactorParams",
    "BioreactorTwin",
    "CrystallizerParams",
    "CrystallizerTwin",
    "ElectronicBatchRecord",
    "FillFinishParams",
    "ISA88BatchExecutive",
    "LotGenealogyDAG",
    "MaterialLot",
    "MultivariatePATMonitor",
    "PATConfig",
    "PLCScanConfig",
    "PLCRuntime",
    "PLCStaticVerifier",
    "PLSQualityPredictor",
    "PillPressParams",
    "PillPressTwin",
    "PlantCampaignResult",
    "PlantScenarioConfig",
    "S88Operation",
    "S88Phase",
    "S88Procedure",
    "S88State",
    "S88UnitProcedure",
    "SafetyInterlockRule",
    "SerializationHierarchy",
    "SixDOFRobotArm",
    "StabilityConfig",
    "StabilityModel",
    "__version__",
    "build_merkle_proof",
    "compute_gs1_check_digit",
    "compute_merkle_root",
    "compute_sip_f0_lethality",
    "integrate_rkf45",
    "parse_ladder_rungs",
    "parse_structured_text",
    "validate_gtin14",
    "verify_merkle_proof",
]
