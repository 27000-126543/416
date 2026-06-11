"""
Multi-source data ingestion module.
Handles satellite remote sensing, ground observation stations, and ocean buoys.
"""

import asyncio
import collections
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    SATELLITE = "satellite_remote_sensing"
    GROUND_STATION = "surface_observation"
    OCEAN_BUOY = "ocean_mooring"
    REANALYSIS = "reanalysis"
    RADAR = "radar"
    LIDAR = "lidar"


class DataFormat(Enum):
    NETCDF4 = "netcdf4"
    HDF5 = "hdf5"
    GRIB2 = "grib2"
    CSV = "csv"
    BUFR = "bufr"
    ZARR = "zarr"


@dataclass
class DataChunk:
    source_id: str
    source_type: DataSourceType
    timestamp: datetime
    data: xr.Dataset
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_size_bytes: int = 0


@dataclass
class IngestionStats:
    total_chunks: int = 0
    total_bytes: int = 0
    failed_chunks: int = 0
    avg_ingestion_rate_mbps: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


class DataSource:
    def __init__(
        self,
        source_id: str,
        source_type: DataSourceType,
        formats: List[DataFormat],
        quality_threshold: float = 0.85,
        max_retention_days: int = 30,
    ):
        self.source_id = source_id
        self.source_type = source_type
        self.formats = formats
        self.quality_threshold = quality_threshold
        self.max_retention_days = max_retention_days
        self._callbacks: List[Callable[[DataChunk], None]] = []
        self._is_active = False

    def subscribe(self, callback: Callable[[DataChunk], None]):
        self._callbacks.append(callback)

    def _emit(self, chunk: DataChunk):
        for callback in self._callbacks:
            try:
                callback(chunk)
            except Exception as e:
                logger.error(f"Callback error in source {self.source_id}: {e}")

    async def start(self):
        self._is_active = True
        logger.info(f"Started data source: {self.source_id}")

    async def stop(self):
        self._is_active = False
        logger.info(f"Stopped data source: {self.source_id}")

    @property
    def is_active(self) -> bool:
        return self._is_active

    def _check_retention(self, timestamp: datetime) -> bool:
        cutoff = datetime.now() - timedelta(days=self.max_retention_days)
        return timestamp >= cutoff

    def _estimate_size(self, data: xr.Dataset) -> int:
        return sum(v.nbytes for v in data.data_vars.values())


class SatelliteSource(DataSource):
    def __init__(
        self,
        source_id: str,
        satellite_name: str,
        orbit_type: str = "polar",
        spectral_channels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            source_id=source_id,
            source_type=DataSourceType.SATELLITE,
            formats=[DataFormat.NETCDF4, DataFormat.HDF5, DataFormat.GRIB2],
            **kwargs,
        )
        self.satellite_name = satellite_name
        self.orbit_type = orbit_type
        self.spectral_channels = spectral_channels or [
            "visible", "near_infrared", "thermal_infrared", "microwave"
        ]
        self._scan_interval = timedelta(minutes=15)

    async def ingest_from_file(self, file_path: str) -> Optional[DataChunk]:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Satellite file not found: {file_path}")
            return None

        try:
            ds = xr.open_dataset(path)
            timestamp = pd.to_datetime(ds.attrs.get("time_coverage_start", datetime.now()))
            
            if not self._check_retention(timestamp):
                logger.warning(f"Satellite data too old, skipping: {file_path}")
                ds.close()
                return None

            chunk = DataChunk(
                source_id=self.source_id,
                source_type=self.source_type,
                timestamp=timestamp,
                data=ds,
                metadata={
                    "satellite": self.satellite_name,
                    "orbit_type": self.orbit_type,
                    "spectral_channels": self.spectral_channels,
                    "file_path": str(path),
                },
                chunk_size_bytes=self._estimate_size(ds),
            )
            self._emit(chunk)
            return chunk
        except Exception as e:
            logger.error(f"Failed to ingest satellite data {file_path}: {e}")
            return None

    async def ingest_from_memory(self, data: xr.Dataset, timestamp: Optional[datetime] = None) -> Optional[DataChunk]:
        ts = timestamp or datetime.now()
        if not self._check_retention(ts):
            return None

        chunk = DataChunk(
            source_id=self.source_id,
            source_type=self.source_type,
            timestamp=ts,
            data=data,
            metadata={
                "satellite": self.satellite_name,
                "orbit_type": self.orbit_type,
                "spectral_channels": self.spectral_channels,
            },
            chunk_size_bytes=self._estimate_size(data),
        )
        self._emit(chunk)
        return chunk


class GroundStationSource(DataSource):
    def __init__(
        self,
        source_id: str,
        station_id: str,
        station_name: str,
        latitude: float,
        longitude: float,
        elevation_m: float = 0.0,
        variables: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            source_id=source_id,
            source_type=DataSourceType.GROUND_STATION,
            formats=[DataFormat.CSV, DataFormat.NETCDF4, DataFormat.BUFR],
            **kwargs,
        )
        self.station_id = station_id
        self.station_name = station_name
        self.latitude = latitude
        self.longitude = longitude
        self.elevation_m = elevation_m
        self.variables = variables or [
            "temperature_2m", "relative_humidity", "pressure_surface",
            "wind_speed_10m", "wind_direction_10m", "precipitation"
        ]

    async def ingest_observation(self, observation: Dict[str, Any], timestamp: Optional[datetime] = None) -> Optional[DataChunk]:
        ts = timestamp or datetime.now()
        if not self._check_retention(ts):
            return None

        try:
            data = {k: np.array([v]) for k, v in observation.items() if k in self.variables}
            ds = xr.Dataset(
                data_vars={k: (["time", "station"], v.reshape(1, 1)) for k, v in data.items()},
                coords={
                    "time": [ts],
                    "station": [self.station_id],
                    "latitude": ("station", [self.latitude]),
                    "longitude": ("station", [self.longitude]),
                    "elevation": ("station", [self.elevation_m]),
                },
            )
            chunk = DataChunk(
                source_id=self.source_id,
                source_type=self.source_type,
                timestamp=ts,
                data=ds,
                metadata={
                    "station_id": self.station_id,
                    "station_name": self.station_name,
                    "elevation_m": self.elevation_m,
                },
                chunk_size_bytes=self._estimate_size(ds),
            )
            self._emit(chunk)
            return chunk
        except Exception as e:
            logger.error(f"Failed to ingest ground observation at {self.station_id}: {e}")
            return None

    async def ingest_batch(self, observations: pd.DataFrame) -> Optional[DataChunk]:
        if observations.empty:
            return None

        ts = pd.to_datetime(observations.index[0]).to_pydatetime() if observations.index.name == "time" else datetime.now()
        if not self._check_retention(ts):
            return None

        try:
            available_vars = [v for v in self.variables if v in observations.columns]
            ds = xr.Dataset(
                data_vars={v: (["time", "station"], observations[v].values.reshape(-1, 1)) for v in available_vars},
                coords={
                    "time": observations.index.values,
                    "station": [self.station_id],
                    "latitude": ("station", [self.latitude]),
                    "longitude": ("station", [self.longitude]),
                    "elevation": ("station", [self.elevation_m]),
                },
            )
            chunk = DataChunk(
                source_id=self.source_id,
                source_type=self.source_type,
                timestamp=ts,
                data=ds,
                metadata={
                    "station_id": self.station_id,
                    "station_name": self.station_name,
                    "num_observations": len(observations),
                },
                chunk_size_bytes=self._estimate_size(ds),
            )
            self._emit(chunk)
            return chunk
        except Exception as e:
            logger.error(f"Failed to ingest batch at {self.station_id}: {e}")
            return None


class OceanBuoySource(DataSource):
    def __init__(
        self,
        source_id: str,
        buoy_id: str,
        latitude: float,
        longitude: float,
        water_depth_m: Optional[float] = None,
        measurement_depth_m: List[float] = None,
        **kwargs,
    ):
        super().__init__(
            source_id=source_id,
            source_type=DataSourceType.OCEAN_BUOY,
            formats=[DataFormat.NETCDF4, DataFormat.CSV],
            **kwargs,
        )
        self.buoy_id = buoy_id
        self.latitude = latitude
        self.longitude = longitude
        self.water_depth_m = water_depth_m
        self.measurement_depth_m = measurement_depth_m or [0.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

    async def ingest_profile(
        self,
        temperature_profile: Dict[float, float],
        salinity_profile: Optional[Dict[float, float]] = None,
        current_profile: Optional[Dict[float, Tuple[float, float]]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[DataChunk]:
        ts = timestamp or datetime.now()
        if not self._check_retention(ts):
            return None

        try:
            depths = sorted(temperature_profile.keys())
            temp_values = np.array([temperature_profile[d] for d in depths])

            data_vars = {
                "water_temperature": (["time", "depth", "buoy"], temp_values.reshape(1, -1, 1)),
            }

            if salinity_profile:
                sal_values = np.array([salinity_profile.get(d, np.nan) for d in depths])
                data_vars["salinity"] = (["time", "depth", "buoy"], sal_values.reshape(1, -1, 1))

            if current_profile:
                u_values = np.array([current_profile.get(d, (np.nan, np.nan))[0] for d in depths])
                v_values = np.array([current_profile.get(d, (np.nan, np.nan))[1] for d in depths])
                data_vars["current_u"] = (["time", "depth", "buoy"], u_values.reshape(1, -1, 1))
                data_vars["current_v"] = (["time", "depth", "buoy"], v_values.reshape(1, -1, 1))

            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": [ts],
                    "depth": depths,
                    "buoy": [self.buoy_id],
                    "latitude": ("buoy", [self.latitude]),
                    "longitude": ("buoy", [self.longitude]),
                    "water_depth": ("buoy", [self.water_depth_m or np.nan]),
                },
            )
            chunk = DataChunk(
                source_id=self.source_id,
                source_type=self.source_type,
                timestamp=ts,
                data=ds,
                metadata={
                    "buoy_id": self.buoy_id,
                    "water_depth_m": self.water_depth_m,
                    "num_depths": len(depths),
                },
                chunk_size_bytes=self._estimate_size(ds),
            )
            self._emit(chunk)
            return chunk
        except Exception as e:
            logger.error(f"Failed to ingest ocean profile at {self.buoy_id}: {e}")
            return None


class DataIngestionManager:
    def __init__(
        self,
        rate_limit_tbps: float = 1.0,
        buffer_size_mb: int = 1024,
        parallel_workers: int = 64,
    ):
        self.rate_limit_tbps = rate_limit_tbps
        self.buffer_size_mb = buffer_size_mb
        self.parallel_workers = parallel_workers
        self._sources: Dict[str, DataSource] = {}
        self._buffer: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._stats = IngestionStats()
        self._source_stats: Dict[str, IngestionStats] = {}
        self._rate_window = collections.deque()
        self._is_running = False
        self._semaphore = asyncio.Semaphore(parallel_workers)
        self._bytes_processed = 0
        self._start_time = None

    def register_source(self, source: DataSource):
        self._sources[source.source_id] = source
        source.subscribe(self._on_data_received)
        logger.info(f"Registered data source: {source.source_id} ({source.source_type.value})")

    def unregister_source(self, source_id: str):
        if source_id in self._sources:
            source = self._sources[source_id]
            asyncio.create_task(source.stop())
            del self._sources[source_id]
            logger.info(f"Unregistered data source: {source_id}")

    async def _on_data_received(self, chunk: DataChunk):
        self._stats.total_chunks += 1
        self._stats.total_bytes += chunk.chunk_size_bytes
        self._stats.last_update = datetime.now()

        source_id = chunk.source_id
        if source_id not in self._source_stats:
            self._source_stats[source_id] = IngestionStats()
        src_stats = self._source_stats[source_id]
        src_stats.total_chunks += 1
        src_stats.total_bytes += chunk.chunk_size_bytes
        src_stats.last_update = datetime.now()

        now = datetime.now()
        self._rate_window.append((now, chunk.chunk_size_bytes))
        while self._rate_window and (now - self._rate_window[0][0]).total_seconds() > 10.0:
            self._rate_window.popleft()
        if len(self._rate_window) > 1:
            elapsed = (now - self._rate_window[0][0]).total_seconds()
            if elapsed > 0:
                total_bytes_in_window = sum(b for _, b in self._rate_window)
                self._stats.avg_ingestion_rate_mbps = (total_bytes_in_window * 8) / (elapsed * 1e6)
                src_stats.avg_ingestion_rate_mbps = self._stats.avg_ingestion_rate_mbps

        await self._buffer.put(chunk)

    async def start(self):
        self._is_running = True
        self._start_time = datetime.now()
        self._bytes_processed = 0
        logger.info("Starting data ingestion manager")

        tasks = []
        for source in self._sources.values():
            tasks.append(asyncio.create_task(source.start()))
        await asyncio.gather(*tasks)

        asyncio.create_task(self._process_buffer())
        asyncio.create_task(self._update_stats())

    async def stop(self):
        self._is_running = False
        for source in self._sources.values():
            await source.stop()
        logger.info("Stopped data ingestion manager")

    async def _process_buffer(self):
        while self._is_running:
            try:
                chunk = await asyncio.wait_for(self._buffer.get(), timeout=1.0)
                async with self._semaphore:
                    await self._process_chunk(chunk)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing buffer: {e}")

    async def _process_chunk(self, chunk: DataChunk):
        try:
            self._bytes_processed += chunk.chunk_size_bytes
        except Exception as e:
            self._stats.failed_chunks += 1
            logger.error(f"Error processing chunk from {chunk.source_id}: {e}")

    async def _update_stats(self):
        while self._is_running:
            if self._start_time:
                elapsed = (datetime.now() - self._start_time).total_seconds()
                if elapsed > 0:
                    self._stats.avg_ingestion_rate_mbps = (self._bytes_processed * 8) / (elapsed * 1e6)
            self._stats.last_update = datetime.now()
            await asyncio.sleep(5)

    def get_stats(self) -> IngestionStats:
        return self._stats

    def get_source_stats(self, source_id: str) -> Optional[IngestionStats]:
        return self._source_stats.get(source_id)

    def get_all_source_stats(self) -> Dict[str, IngestionStats]:
        return self._source_stats.copy()

    def get_sources(self) -> Dict[str, DataSource]:
        return self._sources.copy()

    @property
    def is_running(self) -> bool:
        return self._is_running
