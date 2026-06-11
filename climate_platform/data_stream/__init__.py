"""
Data stream processing module for multi-source meteorological data.
"""

from .ingestion import DataIngestionManager, DataSource, SatelliteSource, GroundStationSource, OceanBuoySource
from .cleaning import DataCleaner, QualityControlResult
from .interpolation import SpatialInterpolator, TemporalInterpolator, InterpolationResult

__all__ = [
    "DataIngestionManager",
    "DataSource",
    "SatelliteSource",
    "GroundStationSource",
    "OceanBuoySource",
    "DataCleaner",
    "QualityControlResult",
    "SpatialInterpolator",
    "TemporalInterpolator",
    "InterpolationResult",
]
