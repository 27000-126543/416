"""
Numerical model framework for atmosphere-ocean-land-sea ice coupling.
"""

from .coupled_model import (
    CoupledModel,
    ModelComponent,
    AtmosphericModel,
    OceanModel,
    LandModel,
    SeaIceModel,
    CouplingScheduler,
    CouplingType,
    ModelState,
    ModelTimeStepper,
)
from .adaptive_mesh import (
    AdaptiveMesh,
    RefinementCriterion,
    ExtremeWeatherDetector,
    MeshRefinementResult,
    GridCell,
)

__all__ = [
    "CoupledModel",
    "ModelComponent",
    "AtmosphericModel",
    "OceanModel",
    "LandModel",
    "SeaIceModel",
    "CouplingScheduler",
    "CouplingType",
    "ModelState",
    "ModelTimeStepper",
    "AdaptiveMesh",
    "RefinementCriterion",
    "ExtremeWeatherDetector",
    "MeshRefinementResult",
    "GridCell",
]
