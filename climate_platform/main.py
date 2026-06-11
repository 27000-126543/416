"""
Main entry point for the Global Climate Simulation Platform.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from .config import PlatformConfig, load_config
from .data_stream import (
    DataIngestionManager,
    SatelliteSource,
    GroundStationSource,
    OceanBuoySource,
    DataCleaner,
    SpatialInterpolator,
    TemporalInterpolator,
)
from .assimilation import DataAssimilationEngine, AssimilationMethod, Observation
from .model import CoupledModel, CouplingType, AdaptiveMesh, ExtremeWeatherDetector
from .ensemble import EnsembleForecast, PerturbationMethod, ForecastHorizon
from .fingerprint import ClimateFingerprintEngine
from .orchestration import (
    OrchestrationEngine,
    Task,
    Workflow,
    ParameterRange,
)
from .multi_tenant import (
    TenantManager,
    TenantRole,
    ResourceQuota,
    Permission,
    ResourceType,
)
from .visualization import VisualizationEngine, RenderBackend, OutputFormat
from .quality_control import QualityControlEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("climate_platform")


class ClimateSimulationPlatform:
    def __init__(self, config_path: Optional[str] = None):
        self.config: PlatformConfig = load_config(config_path)
        self._setup_directories()

        self.ingestion_manager: Optional[DataIngestionManager] = None
        self.data_cleaner: Optional[DataCleaner] = None
        self.spatial_interpolator: Optional[SpatialInterpolator] = None
        self.temporal_interpolator: Optional[TemporalInterpolator] = None
        self.assimilation_engine: Optional[DataAssimilationEngine] = None
        self.coupled_model: Optional[CoupledModel] = None
        self.adaptive_mesh: Optional[AdaptiveMesh] = None
        self.extreme_weather_detector: Optional[ExtremeWeatherDetector] = None
        self.ensemble_forecast: Optional[EnsembleForecast] = None
        self.fingerprint_engine: Optional[ClimateFingerprintEngine] = None
        self.orchestration_engine: Optional[OrchestrationEngine] = None
        self.tenant_manager: Optional[TenantManager] = None
        self.visualization_engine: Optional[VisualizationEngine] = None
        self.qc_engine: Optional[QualityControlEngine] = None

        self._initialized = False

    def _setup_directories(self):
        for dir_name in ["data_dir", "result_dir", "scratch_dir"]:
            dir_path = Path(getattr(self.config.system, dir_name))
            dir_path.mkdir(parents=True, exist_ok=True)

    def initialize(self):
        ds_cfg = self.config.data_stream
        self.ingestion_manager = DataIngestionManager(
            rate_limit_tbps=ds_cfg.ingestion_rate_limit_tbps,
            buffer_size_mb=ds_cfg.buffer_size_mb,
            parallel_workers=ds_cfg.parallel_workers,
        )

        for source_id, source_cfg in ds_cfg.sources.items():
            if source_cfg.type == "satellite_remote_sensing":
                source = SatelliteSource(
                    source_id=source_id,
                    satellite_name=source_id,
                    quality_threshold=source_cfg.quality_threshold,
                    max_retention_days=source_cfg.max_retention_days,
                )
            elif source_cfg.type == "surface_observation":
                source = GroundStationSource(
                    source_id=source_id,
                    station_id=source_id,
                    station_name=source_id,
                    latitude=0.0,
                    longitude=0.0,
                    quality_threshold=source_cfg.quality_threshold,
                    max_retention_days=source_cfg.max_retention_days,
                )
            elif source_cfg.type == "ocean_mooring":
                source = OceanBuoySource(
                    source_id=source_id,
                    buoy_id=source_id,
                    latitude=0.0,
                    longitude=0.0,
                    quality_threshold=source_cfg.quality_threshold,
                    max_retention_days=source_cfg.max_retention_days,
                )
            else:
                continue
            self.ingestion_manager.register_source(source)

        self.data_cleaner = DataCleaner(
            quality_threshold=0.85,
            enable_range_check=True,
            enable_gradient_check=True,
            enable_spatial_consistency=True,
            enable_temporal_consistency=True,
            enable_buddy_check=True,
            auto_clean=True,
        )

        self.spatial_interpolator = SpatialInterpolator()
        self.temporal_interpolator = TemporalInterpolator()

        da_cfg = self.config.data_assimilation
        self.assimilation_engine = DataAssimilationEngine(
            method=AssimilationMethod(da_cfg.method),
            ensemble_size=da_cfg.ensemble_size,
            localization_radius_km=da_cfg.localization_radius_km,
            inflation_factor=da_cfg.inflation_factor,
            time_window_hours=da_cfg.time_window_hours,
        )

        nm_cfg = self.config.numerical_model
        coupling_type = CouplingType.FULLY_COUPLED if nm_cfg.coupling.fully_coupled else CouplingType.TWO_WAY
        self.coupled_model = CoupledModel(
            coupling_type=coupling_type,
            coupling_interval_seconds=nm_cfg.coupling.coupling_interval_seconds,
        )
        self.coupled_model.initialize()

        self.adaptive_mesh = AdaptiveMesh(
            base_resolution_deg=nm_cfg.resolution.base_resolution_deg,
            max_refinement_level=nm_cfg.resolution.max_refinement_level,
            min_cell_size_km=nm_cfg.resolution.min_cell_size_km,
        )
        self.adaptive_mesh.initialize()

        self.extreme_weather_detector = ExtremeWeatherDetector()

        ef_cfg = self.config.ensemble_forecast
        self.ensemble_forecast = EnsembleForecast(
            ensemble_size=ef_cfg.ensemble_size,
            perturbation_method=PerturbationMethod(ef_cfg.perturbation_method),
        )

        self.fingerprint_engine = ClimateFingerprintEngine(
            tf_methods=self.config.climate_fingerprint.time_frequency_analysis.methods,
            causal_method=self.config.climate_fingerprint.causal_inference.method,
            similarity_metric=self.config.climate_fingerprint.pattern_matching.similarity_metric,
        )

        orch_cfg = self.config.orchestration
        self.orchestration_engine = OrchestrationEngine(
            max_parallel_tasks=orch_cfg.max_parallel_tasks,
            checkpoint_interval_seconds=orch_cfg.checkpoint_interval_seconds,
            retry_attempts_default=orch_cfg.retry_attempts,
            retry_backoff_seconds=orch_cfg.retry_backoff_seconds,
            checksum_algorithm=orch_cfg.result_validation.checksum_algorithm,
        )

        mt_cfg = self.config.multi_tenant
        self.tenant_manager = TenantManager(
            base_sandbox_path="./sandboxes",
            default_quota=ResourceQuota(
                storage_tb=mt_cfg.default_quota.storage_tb,
                compute_hours_month=mt_cfg.default_quota.compute_hours_month,
                concurrent_jobs=mt_cfg.default_quota.concurrent_jobs,
            ),
            enable_sandbox_isolation=mt_cfg.sandbox_isolation,
        )

        for tenant_cfg in mt_cfg.tenants:
            try:
                self.tenant_manager.create_tenant(
                    tenant_id=tenant_cfg.id,
                    name=tenant_cfg.name,
                    role=TenantRole(tenant_cfg.role),
                )
            except ValueError:
                pass

        viz_cfg = self.config.visualization
        self.visualization_engine = VisualizationEngine(
            backend=RenderBackend(viz_cfg.backend),
            interactive=viz_cfg.interactive,
            render_quality=viz_cfg.render_quality,
        )

        qc_cfg = self.config.quality_control
        self.qc_engine = QualityControlEngine(
            enable_range_check=qc_cfg.checks.range_check,
            enable_gradient_check=qc_cfg.checks.gradient_check,
            enable_spatial_consistency=qc_cfg.checks.spatial_consistency,
            enable_temporal_consistency=qc_cfg.checks.temporal_consistency,
            enable_buddy_check=qc_cfg.checks.buddy_check,
        )

        self._initialized = True
        logger.info(f"Climate Simulation Platform v{self.config.system.version} initialized")

    def run_simulation(
        self,
        start_time: Optional[datetime] = None,
        duration: timedelta = timedelta(days=1),
        tenant_id: Optional[str] = None,
        with_ensemble: bool = False,
        ensemble_size: Optional[int] = None,
    ):
        if not self._initialized:
            self.initialize()

        st = start_time or datetime(2020, 1, 1)
        logger.info(f"Starting simulation: {st} + {duration}")

        states = self.coupled_model.initialize(start_time=st)

        if with_ensemble:
            if ensemble_size:
                self.ensemble_forecast.ensemble_size = ensemble_size
            control_state = states["atmosphere"].data
            self.ensemble_forecast.generate_initial_ensemble(control_state, start_time=st)
            logger.info(f"Ensemble forecast initialized with {self.ensemble_forecast.ensemble_size} members")

        outputs = self.coupled_model.run(duration)

        combined = self.coupled_model.get_combined_state()

        passed, qc_report = self.qc_engine.validate(combined)
        logger.info(f"QC passed: {passed}, pass rate: {qc_report['overall_pass_rate']:.4f}")

        version, _ = self.qc_engine.create_version(
            dataset=combined,
            dataset_name="simulation_output",
            change_type="minor",
            change_log=f"Simulation run from {st} for {duration}",
        )
        if version:
            logger.info(f"Created version: {version.version_number}")

        if tenant_id:
            result_dir = Path(self.config.system.result_dir) / tenant_id
            result_dir.mkdir(parents=True, exist_ok=True)
            output_path = result_dir / f"simulation_{st.strftime('%Y%m%d_%H%M%S')}.nc"
            combined.to_netcdf(str(output_path))
            logger.info(f"Results saved to {output_path}")

        return combined, qc_report

    def run_data_assimilation_cycle(
        self,
        background: xr.Dataset,
        observations: list,
    ):
        if not self._initialized:
            self.initialize()

        result = self.assimilation_engine.assimilate(
            background=background,
            observations=observations,
        )
        logger.info(
            f"Assimilation complete: {result.observations_assimilated} obs assimilated, "
            f"{result.observations_rejected} rejected in {result.computation_time:.2f}s"
        )
        return result

    def find_similar_climate_events(
        self,
        query_dataset: xr.Dataset,
        top_k: int = 10,
    ):
        if not self._initialized:
            self.initialize()

        return self.fingerprint_engine.search_similar_events(query_dataset, top_k=top_k)

    def generate_visualizations(
        self,
        dataset: xr.Dataset,
        output_dir: Optional[str] = None,
        variables: Optional[list] = None,
    ):
        if not self._initialized:
            self.initialize()

        out_dir = output_dir or self.config.system.result_dir
        return self.visualization_engine.render_all(dataset, variables, out_dir)

    def parameter_sensitivity_analysis(
        self,
        model_fn,
        parameters: list,
        n_samples: int = 100,
        objective_fn=None,
    ):
        if not self._initialized:
            self.initialize()

        return self.orchestration_engine.run_parameter_sweep(
            base_task_fn=model_fn,
            parameters=parameters,
            n_samples=n_samples,
            objective_fn=objective_fn,
        )

    def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "version": self.config.system.version,
            "system_name": self.config.system.name,
            "tenants": (
                len(self.tenant_manager.tenants) if self.tenant_manager else 0
            ),
            "registered_sources": (
                len(self.ingestion_manager.get_sources()) if self.ingestion_manager else 0
            ),
            "active_workers": (
                self.tenant_manager.get_worker_count() if self.tenant_manager else 0
            ),
        }

    async def start_realtime(self):
        if not self._initialized:
            self.initialize()

        logger.info("Starting real-time data ingestion and processing")
        await self.ingestion_manager.start()

        try:
            while True:
                await asyncio.sleep(60)
                stats = self.ingestion_manager.get_stats()
                logger.info(
                    f"Ingestion stats: {stats.total_chunks} chunks, "
                    f"{stats.total_bytes / 1e9:.2f} GB, "
                    f"{stats.avg_ingestion_rate_mbps:.2f} Mbps"
                )
        except asyncio.CancelledError:
            await self.ingestion_manager.stop()


def create_default_config() -> PlatformConfig:
    return PlatformConfig()


def main():
    parser = argparse.ArgumentParser(
        description="Global Climate Simulation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "realtime", "status", "version"],
        default="status",
        help="Operation mode",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=1,
        help="Simulation duration in days",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run ensemble forecast",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=10,
        help="Ensemble size",
    )
    parser.add_argument(
        "--tenant",
        type=str,
        default=None,
        help="Tenant ID for sandboxed execution",
    )

    args = parser.parse_args()

    platform = ClimateSimulationPlatform(config_path=args.config)

    if args.mode == "version":
        print(f"Global Climate Simulation Platform v{platform.config.system.version}")
        return 0

    platform.initialize()

    if args.mode == "status":
        status = platform.get_status()
        print(json.dumps(status, indent=2, default=str))
        return 0

    if args.mode == "simulation":
        result, qc = platform.run_simulation(
            duration=timedelta(days=args.duration),
            tenant_id=args.tenant,
            with_ensemble=args.ensemble,
            ensemble_size=args.ensemble_size,
        )
        print(f"Simulation complete. Variables: {list(result.data_vars.keys())}")
        print(f"QC Pass Rate: {qc['overall_pass_rate']:.2%}")
        if qc["issues"]:
            print("QC Issues:")
            for issue in qc["issues"][:5]:
                print(f"  - {issue}")
        return 0

    if args.mode == "realtime":
        try:
            asyncio.run(platform.start_realtime())
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        return 0

    return 1


import json

if __name__ == "__main__":
    sys.exit(main())
