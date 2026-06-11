"""
Quality control checks and data versioning with provenance tracking.
"""

import hashlib
import json
import logging
import pickle
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class QCResult:
    check_name: str
    variable: str
    passed: bool
    total_points: int = 0
    failed_points: int = 0
    error_details: List[str] = field(default_factory=list)
    failure_indices: Optional[np.ndarray] = None
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.total_points == 0:
            return 1.0
        return 1.0 - self.failed_points / self.total_points


@dataclass
class ValidationRecord:
    record_id: str
    data_source: str
    passed: bool
    overall_pass_rate: float
    total_points: int
    failed_points: int
    min_pass_rate: float
    timestamp: datetime
    variables: List[str]
    variable_details: Dict[str, Dict[str, Any]]
    issues: List[str]


class QualityCheck(ABC):
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def check(self, dataset: xr.Dataset, variable: str) -> QCResult:
        pass


class RangeCheck(QualityCheck):
    UNIT_MAP = {
        "temperature": "K",
        "sea_surface_temperature": "K",
        "water_temperature": "K",
        "skin_temperature": "K",
        "sea_ice_temperature": "K",
        "relative_humidity": "%",
        "specific_humidity": "kg/kg",
        "pressure": "Pa",
        "surface_pressure": "Pa",
        "wind_speed": "m/s",
        "u_wind": "m/s",
        "v_wind": "m/s",
        "wind_direction": "degree",
        "precipitation": "mm",
        "salinity": "PSU",
        "sea_ice_concentration": "fraction",
        "sea_ice_thickness": "m",
        "soil_moisture": "m3/m3",
        "albedo": "fraction",
        "cloud_cover": "fraction",
        "geopotential_height": "m",
        "shortwave_radiation": "W/m2",
        "sensible_heat_flux": "W/m2",
        "latent_heat_flux": "W/m2",
        "evapotranspiration": "mm",
    }

    RANGE_KELVIN = {
        "temperature": (180.0, 330.0),
        "temperature_2m": (180.0, 320.0),
        "water_temperature": (271.0, 305.0),
        "sea_surface_temperature": (271.0, 305.0),
        "skin_temperature": (200.0, 330.0),
        "sea_ice_temperature": (200.0, 273.0),
        "pressure": (87000.0, 108500.0),
        "surface_pressure": (87000.0, 108500.0),
        "pressure_sea_level": (87000.0, 108500.0),
        "wind_speed": (0.0, 100.0),
        "wind_speed_10m": (0.0, 85.0),
        "wind_direction": (0.0, 360.0),
        "wind_direction_10m": (0.0, 360.0),
        "u_wind": (-150.0, 150.0),
        "v_wind": (-150.0, 150.0),
        "relative_humidity": (0.0, 100.0),
        "specific_humidity": (0.0, 0.05),
        "precipitation": (0.0, 1000.0),
        "salinity": (0.0, 42.0),
        "sea_ice_concentration": (0.0, 1.0),
        "sea_ice_thickness": (0.0, 50.0),
        "soil_moisture": (0.0, 1.0),
        "albedo": (0.0, 1.0),
        "cloud_cover": (0.0, 1.0),
        "geopotential_height": (0.0, 50000.0),
        "shortwave_radiation": (0.0, 1500.0),
        "sensible_heat_flux": (-500.0, 500.0),
        "latent_heat_flux": (-500.0, 500.0),
        "evapotranspiration": (0.0, 100.0),
    }

    PHYSICAL_RANGES = {
        "temperature": (-90.0, 60.0),
        "temperature_2m": (-80.0, 55.0),
        "water_temperature": (-2.0, 40.0),
        "relative_humidity": (0.0, 100.0),
        "specific_humidity": (0.0, 0.05),
        "pressure": (87000.0, 108500.0),
        "wind_speed": (0.0, 100.0),
        "wind_direction": (0.0, 360.0),
        "precipitation": (0.0, 1000.0),
        "salinity": (0.0, 42.0),
        "sea_ice_concentration": (0.0, 1.0),
        "geopotential_height": (0.0, 50000.0),
    }

    def __init__(self, custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None, **kwargs):
        super().__init__(name="range_check", **kwargs)
        self.ranges = self.RANGE_KELVIN.copy()
        if custom_ranges:
            self.ranges.update(custom_ranges)

    def check(self, dataset: xr.Dataset, variable: str) -> QCResult:
        result = QCResult(check_name=self.name, variable=variable, passed=True)

        if variable not in dataset.data_vars:
            result.passed = True
            result.error_details.append(f"Variable {variable} not found, skipping")
            return result

        data = dataset[variable].values
        valid = np.isfinite(data)
        result.total_points = int(np.sum(valid))

        min_val, max_val = None, None
        for key in self.ranges:
            if key in variable.lower():
                min_val, max_val = self.ranges[key]
                break

        if min_val is None:
            result.passed = True
            result.error_details.append(f"No range defined for {variable}, skipping")
            return result

        out_of_range = valid & ((data < min_val) | (data > max_val))
        result.failed_points = int(np.sum(out_of_range))
        result.failure_indices = np.where(out_of_range)

        if np.any(valid):
            result.stats = {
                "min": float(np.nanmin(data[valid])),
                "max": float(np.nanmax(data[valid])),
                "unit": self.UNIT_MAP.get(variable, "unknown"),
                "range_min": min_val,
                "range_max": max_val,
            }

        result.passed = result.failed_points == 0
        if not result.passed:
            result.error_details.append(
                f"{result.failed_points}/{result.total_points} values outside range [{min_val}, {max_val}]"
            )

        return result


class GradientCheck(QualityCheck):
    def __init__(self, threshold: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(name="gradient_check", **kwargs)
        self.thresholds = threshold or {
            "temperature": 15.0,
            "pressure": 500.0,
            "wind": 30.0,
            "humidity": 50.0,
        }

    def check(self, dataset: xr.Dataset, variable: str) -> QCResult:
        result = QCResult(check_name=self.name, variable=variable, passed=True)

        if variable not in dataset.data_vars:
            result.passed = True
            return result

        data = dataset[variable].values
        lat_name = None
        lon_name = None
        for ln in ["lat", "latitude"]:
            if ln in dataset[variable].coords:
                lat_name = ln
                break
        for ln in ["lon", "longitude"]:
            if ln in dataset[variable].coords:
                lon_name = ln
                break

        if lat_name is None or lon_name is None or data.ndim < 2:
            result.passed = True
            result.error_details.append("Not enough spatial dimensions, skipping")
            return result

        dims = dataset[variable].dims
        lat_axis = dims.index(lat_name) if lat_name in dims else None
        lon_axis = dims.index(lon_name) if lon_name in dims else None

        if lat_axis is None or lon_axis is None:
            result.passed = True
            return result

        threshold = 5.0
        for key, val in self.thresholds.items():
            if key in variable.lower():
                threshold = val
                break

        try:
            valid = np.isfinite(data)
            result.total_points = int(np.sum(valid))

            if data.shape[lat_axis] >= 2 and data.shape[lon_axis] >= 2:
                grad_lat = np.abs(np.gradient(data, axis=lat_axis))
                grad_lon = np.abs(np.gradient(data, axis=lon_axis))
                total_grad = np.sqrt(grad_lat ** 2 + grad_lon ** 2)
                extreme = valid & (total_grad > threshold)

                result.failed_points = int(np.sum(extreme))
                result.failure_indices = np.where(extreme)
                result.stats = {
                    "max_gradient": float(np.nanmax(total_grad)),
                    "mean_gradient": float(np.nanmean(total_grad)),
                    "threshold": threshold,
                }

            result.passed = result.failed_points == 0
            if not result.passed:
                result.error_details.append(
                    f"{result.failed_points}/{result.total_points} points exceed gradient threshold {threshold}"
                )
        except Exception as e:
            result.error_details.append(f"Error computing gradient: {e}")

        return result


class SpatialConsistencyCheck(QualityCheck):
    def __init__(self, zscore_threshold: float = 6.0, **kwargs):
        super().__init__(name="spatial_consistency", **kwargs)
        self.zscore_threshold = zscore_threshold

    def check(self, dataset: xr.Dataset, variable: str) -> QCResult:
        result = QCResult(check_name=self.name, variable=variable, passed=True)

        if variable not in dataset.data_vars:
            return result

        data = dataset[variable].values
        valid = np.isfinite(data)
        result.total_points = int(np.sum(valid))

        if not np.any(valid) or data.size < 10:
            result.passed = True
            return result

        try:
            mean = np.nanmean(data)
            std = np.nanstd(data)

            if std > 0:
                zscores = np.abs((data - mean) / std)
                outliers = valid & (zscores > self.zscore_threshold)
                result.failed_points = int(np.sum(outliers))
                result.failure_indices = np.where(outliers)
                result.stats = {
                    "mean": float(mean),
                    "std": float(std),
                    "zscore_threshold": self.zscore_threshold,
                    "max_zscore": float(np.nanmax(zscores)),
                }

            result.passed = result.failed_points == 0
            if not result.passed:
                result.error_details.append(
                    f"{result.failed_points}/{result.total_points} spatial outliers detected (z > {self.zscore_threshold})"
                )
        except Exception as e:
            result.error_details.append(f"Error in spatial consistency check: {e}")

        return result


class TemporalConsistencyCheck(QualityCheck):
    def __init__(self, zscore_threshold: float = 5.0, **kwargs):
        super().__init__(name="temporal_consistency", **kwargs)
        self.zscore_threshold = zscore_threshold

    def check(self, dataset: xr.Dataset, variable: str) -> QCResult:
        result = QCResult(check_name=self.name, variable=variable, passed=True)

        if variable not in dataset.data_vars:
            return result

        data = dataset[variable].values
        dims = dataset[variable].dims

        if "time" not in dims:
            result.error_details.append("No time dimension, skipping")
            return result

        time_axis = dims.index("time")
        if data.shape[time_axis] < 5:
            result.error_details.append("Insufficient time steps, skipping")
            return result

        try:
            valid = np.isfinite(data)
            result.total_points = int(np.sum(valid))

            second_diff = np.diff(data, n=2, axis=time_axis)
            valid_diff = np.isfinite(second_diff)

            if np.any(valid_diff):
                std = np.nanstd(second_diff)
                if std > 0:
                    zscores = np.abs(second_diff / std)
                    spikes = valid_diff & (zscores > self.zscore_threshold)
                    result.failed_points = int(np.sum(spikes))
                    result.stats = {
                        "std_second_diff": float(std),
                        "zscore_threshold": self.zscore_threshold,
                    }

            result.passed = result.failed_points == 0
            if not result.passed:
                result.error_details.append(
                    f"{result.failed_points} temporal discontinuities detected"
                )
        except Exception as e:
            result.error_details.append(f"Error in temporal consistency check: {e}")

        return result


class BuddyCheck(QualityCheck):
    def __init__(
        self,
        num_neighbors: int = 5,
        threshold_multiplier: float = 3.0,
        **kwargs,
    ):
        super().__init__(name="buddy_check", **kwargs)
        self.num_neighbors = num_neighbors
        self.threshold_multiplier = threshold_multiplier

    def check(self, dataset: xr.Dataset, variable: str) -> QCResult:
        result = QCResult(check_name=self.name, variable=variable, passed=True)

        if variable not in dataset.data_vars:
            return result

        data = dataset[variable].values
        valid = np.isfinite(data)
        result.total_points = int(np.sum(valid))

        if data.ndim < 2 or not np.any(valid):
            return result

        try:
            from scipy.ndimage import generic_filter

            def _neighbor_check(window):
                center = window[len(window) // 2]
                neighbors = np.concatenate([window[:len(window)//2], window[len(window)//2 + 1:]])
                valid_neighbors = neighbors[np.isfinite(neighbors)]

                if not np.isfinite(center) or len(valid_neighbors) < 3:
                    return 0.0

                neighbor_mean = np.mean(valid_neighbors)
                neighbor_std = np.std(valid_neighbors) if len(valid_neighbors) > 1 else 1.0

                if neighbor_std == 0:
                    return 0.0 if abs(center - neighbor_mean) < 1e-6 else 1.0

                z = abs(center - neighbor_mean) / neighbor_std
                return 1.0 if z > self.threshold_multiplier else 0.0

            window_size = min(3, min(data.shape[:2]))
            if data.ndim >= 2:
                footprint = np.ones((window_size, window_size))
                flags = generic_filter(data, _neighbor_check, footprint=footprint, mode='reflect')
                failed = valid & (flags > 0.5)
                result.failed_points = int(np.sum(failed))
                result.failure_indices = np.where(failed)

            result.passed = result.failed_points == 0
            if not result.passed:
                result.error_details.append(
                    f"{result.failed_points}/{result.total_points} failed buddy check"
                )
        except Exception as e:
            result.error_details.append(f"Error in buddy check: {e}")

        return result


@dataclass
class ProvenanceNode:
    node_id: str
    node_type: str
    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None


@dataclass
class ProvenanceEdge:
    source_id: str
    target_id: str
    edge_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProvenanceGraph:
    def __init__(self):
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: List[ProvenanceEdge] = []
        self._incoming: Dict[str, Set[str]] = {}
        self._outgoing: Dict[str, Set[str]] = {}

    def add_node(self, node: ProvenanceNode):
        self.nodes[node.node_id] = node
        if node.node_id not in self._incoming:
            self._incoming[node.node_id] = set()
        if node.node_id not in self._outgoing:
            self._outgoing[node.node_id] = set()

    def add_edge(self, edge: ProvenanceEdge):
        self.edges.append(edge)
        self._outgoing[edge.source_id].add(edge.target_id)
        self._incoming[edge.target_id].add(edge.source_id)

    def connect(self, source_id: str, target_id: str, edge_type: str = "derived_from", **kwargs):
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        edge = ProvenanceEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            metadata=kwargs,
        )
        self.add_edge(edge)

    def get_lineage(self, node_id: str, direction: str = "upstream") -> List[ProvenanceNode]:
        result = []
        visited = set()

        def _traverse(nid: str):
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            result.append(self.nodes[nid])
            if direction == "upstream":
                for src in self._incoming.get(nid, set()):
                    _traverse(src)
            else:
                for tgt in self._outgoing.get(nid, set()):
                    _traverse(tgt)

        _traverse(node_id)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "name": n.name,
                    "timestamp": n.timestamp.isoformat(),
                    "metadata": n.metadata,
                    "checksum": n.checksum,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type,
                    "timestamp": e.timestamp.isoformat(),
                    "metadata": e.metadata,
                }
                for e in self.edges
            ]
        }

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ProvenanceGraph":
        graph = cls()
        with open(filepath, "r") as f:
            data = json.load(f)

        for node_data in data["nodes"]:
            node = ProvenanceNode(
                node_id=node_data["node_id"],
                node_type=node_data["node_type"],
                name=node_data["name"],
                timestamp=datetime.fromisoformat(node_data["timestamp"]),
                metadata=node_data.get("metadata", {}),
                checksum=node_data.get("checksum"),
            )
            graph.add_node(node)

        for edge_data in data["edges"]:
            edge = ProvenanceEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=edge_data["edge_type"],
                timestamp=datetime.fromisoformat(edge_data["timestamp"]),
                metadata=edge_data.get("metadata", {}),
            )
            graph.add_edge(edge)

        return graph


@dataclass
class DataVersion:
    version_id: str
    version_number: str
    dataset_name: str
    checksum: str
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.now)
    parent_version: Optional[str] = None
    change_log: str = ""
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_semver(self) -> Tuple[int, int, int]:
        parts = self.version_number.split(".")
        try:
            return (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
        except (ValueError, IndexError):
            return (0, 0, 0)


class VersionTracker:
    def __init__(self, algorithm: str = "semver"):
        self.algorithm = algorithm
        self.versions: Dict[str, List[DataVersion]] = {}
        self._version_counters: Dict[str, Tuple[int, int, int]] = {}

    def create_version(
        self,
        dataset_name: str,
        data: Any,
        change_type: str = "patch",
        change_log: str = "",
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataVersion:
        version_num = self._bump_version(dataset_name, change_type)
        checksum = self._compute_checksum(data)
        size = self._compute_size(data)
        variables = self._extract_variables(data)

        version = DataVersion(
            version_id=str(uuid.uuid4()),
            version_number=version_num,
            dataset_name=dataset_name,
            checksum=checksum,
            size_bytes=size,
            parent_version=parent_version,
            change_log=change_log,
            variables=variables,
            metadata=metadata or {},
        )

        if dataset_name not in self.versions:
            self.versions[dataset_name] = []
        self.versions[dataset_name].append(version)

        return version

    def _bump_version(self, dataset_name: str, change_type: str) -> str:
        if dataset_name not in self._version_counters:
            self._version_counters[dataset_name] = (1, 0, 0)
            return "1.0.0"

        major, minor, patch = self._version_counters[dataset_name]

        if change_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif change_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1

        self._version_counters[dataset_name] = (major, minor, patch)
        return f"{major}.{minor}.{patch}"

    @staticmethod
    def _compute_checksum(data: Any) -> str:
        try:
            if isinstance(data, xr.Dataset):
                hasher = hashlib.sha256()
                for var in sorted(data.data_vars):
                    hasher.update(data[var].values.tobytes())
                    hasher.update(var.encode())
                return hasher.hexdigest()
            elif isinstance(data, np.ndarray):
                return hashlib.sha256(data.tobytes()).hexdigest()
            elif isinstance(data, (dict, list, str)):
                return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
            else:
                serialized = pickle.dumps(data)
                return hashlib.sha256(serialized).hexdigest()
        except Exception:
            return str(uuid.uuid4())

    @staticmethod
    def _compute_size(data: Any) -> int:
        try:
            if isinstance(data, xr.Dataset):
                return sum(v.nbytes for v in data.data_vars.values())
            elif isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, (dict, list)):
                return len(pickle.dumps(data))
            else:
                serialized = pickle.dumps(data)
                return len(serialized)
        except Exception:
            return 0

    @staticmethod
    def _extract_variables(data: Any) -> List[str]:
        if isinstance(data, xr.Dataset):
            return list(data.data_vars)
        elif isinstance(data, dict):
            return list(data.keys())
        return []

    def get_versions(self, dataset_name: str) -> List[DataVersion]:
        return sorted(
            self.versions.get(dataset_name, []),
            key=lambda v: v.created_at,
        )

    def get_latest(self, dataset_name: str) -> Optional[DataVersion]:
        versions = self.get_versions(dataset_name)
        return versions[-1] if versions else None

    def get_version(self, dataset_name: str, version_number: str) -> Optional[DataVersion]:
        for v in self.get_versions(dataset_name):
            if v.version_number == version_number:
                return v
        return None

    def compare_versions(
        self,
        dataset_name: str,
        v1: str,
        v2: str,
    ) -> Dict[str, Any]:
        version1 = self.get_version(dataset_name, v1)
        version2 = self.get_version(dataset_name, v2)

        if version1 is None or version2 is None:
            return {"error": "Version not found"}

        result = {
            "version1": version1.version_number,
            "version2": version2.version_number,
            "size_change": version2.size_bytes - version1.size_bytes,
            "variables_added": list(set(version2.variables) - set(version1.variables)),
            "variables_removed": list(set(version1.variables) - set(version2.variables)),
            "checksum_changed": version1.checksum != version2.checksum,
            "time_elapsed": (version2.created_at - version1.created_at).total_seconds(),
        }
        return result

    def list_datasets(self) -> List[str]:
        return sorted(self.versions.keys())


class QualityControlEngine:
    def __init__(
        self,
        enable_range_check: bool = True,
        enable_gradient_check: bool = True,
        enable_spatial_consistency: bool = True,
        enable_temporal_consistency: bool = True,
        enable_buddy_check: bool = True,
    ):
        self.checks: List[QualityCheck] = []
        if enable_range_check:
            self.checks.append(RangeCheck())
        if enable_gradient_check:
            self.checks.append(GradientCheck())
        if enable_spatial_consistency:
            self.checks.append(SpatialConsistencyCheck())
        if enable_temporal_consistency:
            self.checks.append(TemporalConsistencyCheck())
        if enable_buddy_check:
            self.checks.append(BuddyCheck())

        self.version_tracker = VersionTracker()
        self.provenance = ProvenanceGraph()
        self._qc_history: List[Dict[str, Any]] = []
        self._validation_history: List[ValidationRecord] = []

    def add_check(self, check: QualityCheck):
        self.checks.append(check)

    def run_checks(
        self,
        dataset: xr.Dataset,
        variables: Optional[List[str]] = None,
        version: Optional[str] = None,
    ) -> Dict[str, List[QCResult]]:
        vars_to_check = variables or list(dataset.data_vars)
        all_results: Dict[str, List[QCResult]] = {}

        for var in vars_to_check:
            var_results = []
            for check in self.checks:
                if check.enabled:
                    result = check.check(dataset, var)
                    var_results.append(result)
            all_results[var] = var_results

        record = {
            "timestamp": datetime.now(),
            "dataset_id": str(id(dataset)),
            "variables": vars_to_check,
            "num_checks": len(self.checks),
            "all_passed": all(
                r.passed for results in all_results.values() for r in results
            ),
        }
        self._qc_history.append(record)
        return all_results

    def validate(
        self,
        dataset: xr.Dataset,
        variables: Optional[List[str]] = None,
        min_pass_rate: float = 0.95,
        data_source: str = "default",
    ) -> Tuple[bool, Dict[str, Any]]:
        results = self.run_checks(dataset, variables)
        issues = []
        overall_pass_rate = 1.0
        total = 0
        failed = 0
        vars_to_check = variables or list(dataset.data_vars)
        variable_details: Dict[str, Dict[str, Any]] = {}

        for var, var_results in results.items():
            var_total = 0
            var_failed = 0
            var_passed_checks = 0
            var_checks = []
            var_unit = "unknown"
            var_range_min = None
            var_range_max = None

            for r in var_results:
                total += max(1, r.total_points)
                failed += r.failed_points
                var_total += max(1, r.total_points)
                var_failed += r.failed_points
                if r.passed:
                    var_passed_checks += 1
                if not r.passed:
                    issues.append(f"{var}: {r.check_name} - {'; '.join(r.error_details)}")
                if "unit" in r.stats:
                    var_unit = r.stats["unit"]
                if "range_min" in r.stats:
                    var_range_min = r.stats["range_min"]
                if "range_max" in r.stats:
                    var_range_max = r.stats["range_max"]
                var_checks.append({
                    "check": r.check_name,
                    "passed": r.passed,
                    "pass_rate": r.pass_rate,
                    "failed_points": r.failed_points,
                    "total_points": r.total_points,
                })

            var_pass_rate = 1.0 - var_failed / var_total if var_total > 0 else 1.0
            variable_details[var] = {
                "pass_rate": var_pass_rate,
                "total_points": var_total,
                "failed_points": var_failed,
                "passed_checks": var_passed_checks,
                "total_checks": len(var_results),
                "unit": var_unit,
                "range_min": var_range_min,
                "range_max": var_range_max,
                "checks": var_checks,
            }

        overall_pass_rate = 1.0 - failed / total if total > 0 else 1.0
        passed = overall_pass_rate >= min_pass_rate

        record = ValidationRecord(
            record_id=str(uuid.uuid4()),
            data_source=data_source,
            passed=passed,
            overall_pass_rate=overall_pass_rate,
            total_points=total,
            failed_points=failed,
            min_pass_rate=min_pass_rate,
            timestamp=datetime.now(),
            variables=vars_to_check,
            variable_details=variable_details,
            issues=issues,
        )
        self._validation_history.append(record)

        report = {
            "record_id": record.record_id,
            "timestamp": record.timestamp.isoformat(),
            "data_source": data_source,
            "passed": passed,
            "overall_pass_rate": overall_pass_rate,
            "total_points": total,
            "failed_points": failed,
            "min_pass_rate": min_pass_rate,
            "issues": issues,
            "checks_run": len(self.checks),
            "variables": vars_to_check,
            "variable_details": variable_details,
            "results": {
                var: [
                    {
                        "check": r.check_name,
                        "passed": r.passed,
                        "pass_rate": r.pass_rate,
                        "failed_points": r.failed_points,
                        "unit": r.stats.get("unit", "unknown"),
                        "range_min": r.stats.get("range_min"),
                        "range_max": r.stats.get("range_max"),
                        "total_points": r.total_points,
                    }
                    for r in var_results
                ]
                for var, var_results in results.items()
            }
        }
        return passed, report

    def create_version(
        self,
        dataset: xr.Dataset,
        dataset_name: str,
        change_type: str = "patch",
        change_log: str = "",
        run_qc: bool = True,
    ) -> Tuple[Optional[DataVersion], Dict[str, Any]]:
        qc_report = {}
        if run_qc:
            passed, qc_report = self.validate(dataset)
            if not passed:
                logger.warning(f"QC failed for {dataset_name}: {qc_report.get('issues', [])[:5]}")

        qc_node = ProvenanceNode(
            node_id=f"qc_{uuid.uuid4().hex[:8]}",
            node_type="qc_report",
            name=f"QC for {dataset_name}",
            metadata={"passed": qc_report.get("passed", True), **qc_report},
        )
        self.provenance.add_node(qc_node)

        version = self.version_tracker.create_version(
            dataset_name=dataset_name,
            data=dataset,
            change_type=change_type,
            change_log=change_log,
            metadata=qc_report,
        )

        version_node = ProvenanceNode(
            node_id=version.version_id,
            node_type="dataset_version",
            name=f"{dataset_name} v{version.version_number}",
            checksum=version.checksum,
            metadata={"variables": version.variables, "size_bytes": version.size_bytes},
        )
        self.provenance.add_node(version_node)
        self.provenance.connect(qc_node.node_id, version_node.node_id, "validated")

        if version.parent_version:
            parent_node = ProvenanceNode(
                node_id=f"parent_{version.parent_version}",
                node_type="parent_version",
                name=f"Parent version {version.parent_version}",
            )
            self.provenance.add_node(parent_node)
            self.provenance.connect(parent_node.node_id, version_node.node_id, "derived_from")

        return version, qc_report

    def track_processing_step(
        self,
        step_name: str,
        input_datasets: List[str],
        output_dataset: str,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        step_node = ProvenanceNode(
            node_id=f"step_{uuid.uuid4().hex[:8]}",
            node_type="processing_step",
            name=step_name,
            metadata=parameters or {},
        )
        self.provenance.add_node(step_node)

        for input_id in input_datasets:
            if input_id in self.provenance.nodes:
                self.provenance.connect(input_id, step_node.node_id, "input_to")

        output_node = ProvenanceNode(
            node_id=f"output_{uuid.uuid4().hex[:8]}",
            node_type="dataset",
            name=output_dataset,
        )
        self.provenance.add_node(output_node)
        self.provenance.connect(step_node.node_id, output_node.node_id, "produced")

    def get_history(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if dataset_name is None:
            return self._qc_history.copy()
        return [h for h in self._qc_history if dataset_name in str(h.get("variables", []))]

    def query_history(
        self,
        data_source: Optional[str] = None,
        variable: Optional[str] = None,
        passed: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ValidationRecord]:
        filtered = self._validation_history

        if data_source is not None:
            filtered = [r for r in filtered if r.data_source == data_source]

        if variable is not None:
            filtered = [r for r in filtered if variable in r.variables]

        if passed is not None:
            filtered = [r for r in filtered if r.passed == passed]

        if start_time is not None:
            filtered = [r for r in filtered if r.timestamp >= start_time]

        if end_time is not None:
            filtered = [r for r in filtered if r.timestamp <= end_time]

        filtered = sorted(filtered, key=lambda r: r.timestamp, reverse=True)
        return filtered[:limit]

    def get_validation_history(self, limit: Optional[int] = None) -> List[ValidationRecord]:
        records = sorted(self._validation_history, key=lambda r: r.timestamp, reverse=True)
        if limit is not None:
            return records[:limit]
        return records
