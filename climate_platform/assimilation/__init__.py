"""
Data assimilation module.
Implements Ensemble Kalman Filter (EnKF), 3DVar, and 4DVar methods.
"""

from .assimilation import (
    DataAssimilationEngine,
    AssimilationMethod,
    EnKF,
    ThreeDVar,
    FourDVar,
    Localization,
    CovarianceInflation,
    AssimilationResult,
    ObservationOperator,
    Observation,
)

__all__ = [
    "DataAssimilationEngine",
    "AssimilationMethod",
    "EnKF",
    "ThreeDVar",
    "FourDVar",
    "Localization",
    "CovarianceInflation",
    "AssimilationResult",
    "ObservationOperator",
    "Observation",
]
