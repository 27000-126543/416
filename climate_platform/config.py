"""
Platform configuration management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import os


@dataclass
class SystemConfig:
    name: str = "Global Climate Simulation Platform"
    version: str = "1.0.0"
    log_level: str = "INFO"
    data_dir: str = "./data"
    result_dir: str = "./results"
    scratch_dir: str = "./scratch"


@dataclass
class DataSourceConfig:
    type: str
    formats: List[str]
    max_retention_days: int
    quality_threshold: float


@dataclass
class DataStreamConfig:
    ingestion_rate_limit_tbps: float = 1.0
    buffer_size_mb: int = 1024
    parallel_workers: int = 64
    sources: Dict[str, DataSourceConfig] = field(default_factory=dict)


@dataclass
class DataAssimilationConfig:
    method: str = "enkf"
    ensemble_size: int = 100
    localization_radius_km: float = 500.0
    inflation_factor: float = 1.05
    covariance_inflation: str = "multiplicative"
    time_window_hours: int = 6


@dataclass
class ModelCouplingConfig:
    atmosphere_ocean: bool = True
    atmosphere_land: bool = True
    ocean_seaice: bool = True
    fully_coupled: bool = True
    coupling_interval_seconds: int = 900


@dataclass
class ResolutionConfig:
    base_resolution_deg: float = 0.25
    max_refinement_level: int = 4
    min_cell_size_km: float = 1.0


@dataclass
class RefinementCriterion:
    variable: str
    threshold: float


@dataclass
class ExtremeWeatherFocus:
    type: str
    detection_vars: List[str]
    focus_radius_km: float


@dataclass
class AdaptiveMeshConfig:
    enabled: bool = True
    refinement_criteria: List[RefinementCriterion] = field(default_factory=list)
    extreme_weather_focus: List[ExtremeWeatherFocus] = field(default_factory=list)


@dataclass
class NumericalModelConfig:
    coupling: ModelCouplingConfig = field(default_factory=ModelCouplingConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    adaptive_mesh: AdaptiveMeshConfig = field(default_factory=AdaptiveMeshConfig)


@dataclass
class ForecastHorizons:
    short_range_days: int = 10
    medium_range_days: int = 30
    seasonal_months: int = 6
    long_term_years: int = 30


@dataclass
class EnsembleForecastConfig:
    enabled: bool = True
    ensemble_size: int = 50
    perturbation_method: str = "bred"
    breeding_cycles: int = 5
    forecast_horizons: ForecastHorizons = field(default_factory=ForecastHorizons)
    probability_calibration: str = "bayesian"


@dataclass
class TimeFrequencyAnalysisConfig:
    methods: List[str] = field(default_factory=lambda: ["wavelet", "fft", "hilbert_huang"])
    wavelet: str = "morlet"
    scales_per_octave: int = 12


@dataclass
class CausalInferenceConfig:
    method: str = "pc_stable"
    significance_level: float = 0.05
    max_lag: int = 10


@dataclass
class PatternMatchingConfig:
    similarity_metric: str = "cosine"
    top_k_results: int = 20
    temporal_weight: float = 0.6
    spatial_weight: float = 0.4


@dataclass
class ClimateFingerprintConfig:
    time_frequency_analysis: TimeFrequencyAnalysisConfig = field(default_factory=TimeFrequencyAnalysisConfig)
    causal_inference: CausalInferenceConfig = field(default_factory=CausalInferenceConfig)
    pattern_matching: PatternMatchingConfig = field(default_factory=PatternMatchingConfig)


@dataclass
class ResultValidationConfig:
    enabled: bool = True
    checksum_algorithm: str = "sha256"


@dataclass
class ParameterSweepConfig:
    method: str = "latin_hypercube"
    max_samples: int = 1000
    optimization_method: str = "bayesian"
    optimization_iterations: int = 100


@dataclass
class OrchestrationConfig:
    engine: str = "dask"
    max_parallel_tasks: int = 1024
    checkpoint_interval_seconds: int = 3600
    retry_attempts: int = 3
    retry_backoff_seconds: int = 60
    dependency_resolution: str = "topological"
    result_validation: ResultValidationConfig = field(default_factory=ResultValidationConfig)
    parameter_sweep: ParameterSweepConfig = field(default_factory=ParameterSweepConfig)


@dataclass
class TenantQuota:
    storage_tb: float = 10.0
    compute_hours_month: float = 1000.0
    concurrent_jobs: int = 10


@dataclass
class TenantInfo:
    id: str
    name: str
    role: str
    quota: TenantQuota = field(default_factory=TenantQuota)


@dataclass
class CollaborationConfig:
    enabled: bool = True
    data_sharing: str = "controlled"


@dataclass
class DynamicScalingConfig:
    enabled: bool = True
    scaling_policy: str = "auto"
    min_workers: int = 16
    max_workers: int = 1024
    scale_up_threshold: float = 0.75
    scale_down_threshold: float = 0.25
    cooldown_seconds: int = 300


@dataclass
class MultiTenantConfig:
    enabled: bool = True
    sandbox_isolation: bool = True
    default_quota: TenantQuota = field(default_factory=TenantQuota)
    tenants: List[TenantInfo] = field(default_factory=list)
    collaboration: CollaborationConfig = field(default_factory=CollaborationConfig)
    dynamic_scaling: DynamicScalingConfig = field(default_factory=DynamicScalingConfig)


@dataclass
class VisualizationModules:
    wind_streamlines: bool = True
    isosurfaces: bool = True
    volume_rendering: bool = True
    statistics: bool = True


@dataclass
class TemporalPlaybackConfig:
    enabled: bool = True
    default_fps: int = 24


@dataclass
class VisualizationConfig:
    backend: str = "pyvista"
    interactive: bool = True
    render_quality: str = "high"
    output_formats: List[str] = field(default_factory=lambda: ["html", "png", "mp4", "glb"])
    modules: VisualizationModules = field(default_factory=VisualizationModules)
    temporal_playback: TemporalPlaybackConfig = field(default_factory=TemporalPlaybackConfig)


@dataclass
class QualityChecks:
    range_check: bool = True
    gradient_check: bool = True
    spatial_consistency: bool = True
    temporal_consistency: bool = True
    buddy_check: bool = True


@dataclass
class VersioningConfig:
    enabled: bool = True
    algorithm: str = "semver"
    metadata_tracking: bool = True
    provenance_graph: bool = True


@dataclass
class QualityControlConfig:
    enabled: bool = True
    checks: QualityChecks = field(default_factory=QualityChecks)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)


@dataclass
class PlatformConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    data_stream: DataStreamConfig = field(default_factory=DataStreamConfig)
    data_assimilation: DataAssimilationConfig = field(default_factory=DataAssimilationConfig)
    numerical_model: NumericalModelConfig = field(default_factory=NumericalModelConfig)
    ensemble_forecast: EnsembleForecastConfig = field(default_factory=EnsembleForecastConfig)
    climate_fingerprint: ClimateFingerprintConfig = field(default_factory=ClimateFingerprintConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    multi_tenant: MultiTenantConfig = field(default_factory=MultiTenantConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)


def _dict_to_dataclass(data: Dict[str, Any], dataclass_type) -> Any:
    if not isinstance(data, dict):
        return data
    kwargs = {}
    type_hints = dataclass_type.__dataclass_fields__
    for key, value in data.items():
        if key in type_hints:
            field_type = type_hints[key].type
            if hasattr(field_type, '__dataclass_fields__'):
                if isinstance(value, list):
                    inner_type = field_type.__args__[0] if hasattr(field_type, '__args__') else field_type
                    kwargs[key] = [_dict_to_dataclass(item, inner_type) for item in value]
                else:
                    kwargs[key] = _dict_to_dataclass(value, field_type)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                inner_type = field_type.__args__[0]
                if hasattr(inner_type, '__dataclass_fields__'):
                    kwargs[key] = [_dict_to_dataclass(item, inner_type) for item in value]
                else:
                    kwargs[key] = value
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is dict:
                inner_types = field_type.__args__
                if len(inner_types) >= 2 and hasattr(inner_types[1], '__dataclass_fields__'):
                    kwargs[key] = {k: _dict_to_dataclass(v, inner_types[1]) for k, v in value.items()}
                else:
                    kwargs[key] = value
            else:
                kwargs[key] = value
    return dataclass_type(**kwargs)


def load_config(config_path: Optional[str] = None) -> PlatformConfig:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    
    config_path = Path(config_path)
    if not config_path.exists():
        return PlatformConfig()
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    
    return _dict_to_dataclass(raw_config, PlatformConfig)
