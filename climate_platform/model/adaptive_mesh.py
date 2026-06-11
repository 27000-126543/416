"""
Adaptive mesh refinement and extreme weather detection.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class ExtremeWeatherType(Enum):
    TYPHOON = "typhoon"
    HEATWAVE = "heatwave"
    COLD_WAVE = "cold_wave"
    HEAVY_RAIN = "heavy_rain"
    DROUGHT = "drought"
    STORM_SURGE = "storm_surge"
    TORNADO = "tornado"


@dataclass
class GridCell:
    cell_id: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    refinement_level: int = 0
    is_refined: bool = False
    children: List[int] = field(default_factory=list)
    parent: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2)

    @property
    def area_deg2(self) -> float:
        return (self.lat_max - self.lat_min) * (self.lon_max - self.lon_min)

    def contains(self, lat: float, lon: float) -> bool:
        return (self.lat_min <= lat < self.lat_max) and (self.lon_min <= lon < self.lon_max)


@dataclass
class RefinementCriterion:
    variable: str
    threshold: float
    operator: str = ">"
    spatial_gradient: bool = False
    weight: float = 1.0


@dataclass
class ExtremeWeatherEvent:
    event_type: ExtremeWeatherType
    center_lat: float
    center_lon: float
    radius_km: float
    intensity: float
    timestamp: datetime
    variables: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    track: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class MeshRefinementResult:
    mesh: "AdaptiveMesh"
    refined_cells: List[int] = field(default_factory=list)
    coarsened_cells: List[int] = field(default_factory=list)
    extreme_events: List[ExtremeWeatherEvent] = field(default_factory=list)
    refinement_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveMesh:
    def __init__(
        self,
        base_resolution_deg: float = 0.25,
        max_refinement_level: int = 4,
        min_cell_size_km: float = 1.0,
        criteria: Optional[List[RefinementCriterion]] = None,
    ):
        self.base_resolution_deg = base_resolution_deg
        self.max_refinement_level = max_refinement_level
        self.min_cell_size_km = min_cell_size_km
        self.criteria = criteria or [
            RefinementCriterion(variable="temperature_gradient", threshold=5.0),
            RefinementCriterion(variable="vorticity", threshold=1e-4),
            RefinementCriterion(variable="pressure_change", threshold=100.0),
        ]

        self.cells: Dict[int, GridCell] = {}
        self.level_cell_map: Dict[int, List[int]] = {}
        self._next_cell_id = 0
        self._initialized = False

    def initialize(self, lat_range: Tuple[float, float] = (-90, 90), lon_range: Tuple[float, float] = (0, 360)):
        self.cells.clear()
        self.level_cell_map.clear()
        self._next_cell_id = 0

        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

        lat_steps = int((lat_max - lat_min) / self.base_resolution_deg)
        lon_steps = int((lon_max - lon_min) / self.base_resolution_deg)

        level_0_cells = []
        for i in range(lat_steps):
            cell_lat_min = lat_min + i * self.base_resolution_deg
            cell_lat_max = min(lat_min + (i + 1) * self.base_resolution_deg, lat_max)
            for j in range(lon_steps):
                cell_lon_min = lon_min + j * self.base_resolution_deg
                cell_lon_max = min(lon_min + (j + 1) * self.base_resolution_deg, lon_max)
                cell_id = self._next_cell_id
                self.cells[cell_id] = GridCell(
                    cell_id=cell_id,
                    lat_min=cell_lat_min,
                    lat_max=cell_lat_max,
                    lon_min=cell_lon_min,
                    lon_max=cell_lon_max,
                    refinement_level=0,
                )
                level_0_cells.append(cell_id)
                self._next_cell_id += 1

        self.level_cell_map[0] = level_0_cells
        self._initialized = True
        logger.info(f"Adaptive mesh initialized: {len(self.cells)} base cells")

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    @property
    def active_cells(self) -> List[int]:
        return [cid for cid, cell in self.cells.items() if not cell.is_refined]

    def get_level_cells(self, level: int) -> List[int]:
        return self.level_cell_map.get(level, [])

    def get_cell_at(self, lat: float, lon: float, level: Optional[int] = None) -> Optional[GridCell]:
        if level is None:
            for cid in self.active_cells:
                cell = self.cells[cid]
                if cell.contains(lat, lon):
                    return cell
        else:
            for cid in self.level_cell_map.get(level, []):
                cell = self.cells[cid]
                if cell.contains(lat, lon):
                    return cell
        return None

    def refine_cell(self, cell_id: int) -> List[int]:
        if cell_id not in self.cells:
            return []

        parent = self.cells[cell_id]
        if parent.refinement_level >= self.max_refinement_level:
            return []
        if parent.is_refined:
            return parent.children

        min_size_km = self._cell_size_km(parent)
        if min_size_km <= self.min_cell_size_km:
            return []

        child_level = parent.refinement_level + 1
        lat_mid = (parent.lat_min + parent.lat_max) / 2
        lon_mid = (parent.lon_min + parent.lon_max) / 2

        corners = [
            (parent.lat_min, lat_mid, parent.lon_min, lon_mid),
            (parent.lat_min, lat_mid, lon_mid, parent.lon_max),
            (lat_mid, parent.lat_max, parent.lon_min, lon_mid),
            (lat_mid, parent.lat_max, lon_mid, parent.lon_max),
        ]

        child_ids = []
        for lat_min, lat_max, lon_min, lon_max in corners:
            child_id = self._next_cell_id
            self.cells[child_id] = GridCell(
                cell_id=child_id,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                refinement_level=child_level,
                parent=cell_id,
            )
            child_ids.append(child_id)
            self._next_cell_id += 1

        parent.is_refined = True
        parent.children = child_ids

        if child_level not in self.level_cell_map:
            self.level_cell_map[child_level] = []
        self.level_cell_map[child_level].extend(child_ids)

        return child_ids

    def coarsen_cell(self, cell_id: int) -> bool:
        if cell_id not in self.cells:
            return False

        cell = self.cells[cell_id]
        if cell.parent is None or not cell.is_refined:
            return False

        parent = self.cells[cell.parent]
        for child_id in parent.children:
            if child_id in self.cells:
                child = self.cells[child_id]
                if child.is_refined:
                    return False

        for child_id in parent.children:
            child = self.cells.pop(child_id, None)
            if child and child.refinement_level in self.level_cell_map:
                if child_id in self.level_cell_map[child.refinement_level]:
                    self.level_cell_map[child.refinement_level].remove(child_id)

        parent.is_refined = False
        parent.children = []
        return True

    def _cell_size_km(self, cell: GridCell) -> float:
        lat_center = (cell.lat_min + cell.lat_max) / 2
        lat_km = (cell.lat_max - cell.lat_min) * 111.0
        lon_km = (cell.lon_max - cell.lon_min) * 111.0 * np.cos(np.radians(lat_center))
        return min(lat_km, lon_km)

    def evaluate_criteria(self, dataset: xr.Dataset) -> List[int]:
        needs_refinement = []

        for cid in self.active_cells:
            cell = self.cells[cid]
            if cell.refinement_level >= self.max_refinement_level:
                continue

            cell_data = self._extract_cell_data(dataset, cell)
            for criterion in self.criteria:
                if self._check_criterion(cell_data, criterion):
                    needs_refinement.append(cid)
                    break

        return needs_refinement

    def _extract_cell_data(self, dataset: xr.Dataset, cell: GridCell) -> Dict[str, np.ndarray]:
        result = {}
        lat_name = None
        lon_name = None
        for ln in ["lat", "latitude"]:
            if ln in dataset.coords:
                lat_name = ln
                break
        for ln in ["lon", "longitude"]:
            if ln in dataset.coords:
                lon_name = ln
                break

        if lat_name is None or lon_name is None:
            return result

        lats = dataset[lat_name].values
        lons = dataset[lon_name].values

        lat_mask = (lats >= cell.lat_min) & (lats < cell.lat_max)
        lon_mask = (lons >= cell.lon_min) & (lons < cell.lon_max)

        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]

        if len(lat_indices) == 0 or len(lon_indices) == 0:
            return result

        for var in dataset.data_vars:
            var_data = dataset[var].values
            dims = dataset[var].dims

            lat_axis = dims.index(lat_name) if lat_name in dims else None
            lon_axis = dims.index(lon_name) if lon_name in dims else None

            if lat_axis is not None and lon_axis is not None:
                slicer = [slice(None)] * var_data.ndim
                slicer[lat_axis] = slice(lat_indices[0], lat_indices[-1] + 1)
                slicer[lon_axis] = slice(lon_indices[0], lon_indices[-1] + 1)
                result[var] = var_data[tuple(slicer)]

        return result

    def _check_criterion(self, cell_data: Dict[str, np.ndarray], criterion: RefinementCriterion) -> bool:
        var_name = criterion.variable
        if criterion.spatial_gradient:
            base_var = var_name.replace("_gradient", "")
            if base_var in cell_data:
                data = cell_data[base_var]
                if data.ndim >= 2 and data.shape[0] >= 2 and data.shape[1] >= 2:
                    grad_y = np.diff(data, axis=0)
                    grad_x = np.diff(data, axis=1)
                    gradient = np.sqrt(grad_y[:, :-1] ** 2 + grad_x[:-1, :] ** 2)
                    max_grad = np.nanmax(gradient) if np.any(np.isfinite(gradient)) else 0
                    return self._compare(max_grad, criterion.threshold, criterion.operator)
        else:
            if var_name in cell_data:
                data = cell_data[var_name]
                max_val = np.nanmax(data) if np.any(np.isfinite(data)) else 0
                return self._compare(max_val, criterion.threshold, criterion.operator)

        return False

    @staticmethod
    def _compare(value: float, threshold: float, operator: str) -> bool:
        if operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 1e-10
        else:
            return value > threshold

    def refine_around_point(
        self, lat: float, lon: float, radius_km: float, max_level: Optional[int] = None
    ) -> List[int]:
        max_lvl = max_level if max_level is not None else self.max_refinement_level
        refined = set()
        active = self.active_cells.copy()

        for cid in active:
            cell = self.cells[cid]
            dist = self._distance_to_cell(lat, lon, cell)
            if dist <= radius_km:
                new_ids = self._recursive_refine(cid, max_lvl)
                refined.update(new_ids)

        return list(refined)

    def _recursive_refine(self, cell_id: int, target_level: int) -> List[int]:
        refined = []
        cell = self.cells[cell_id]
        if cell.refinement_level >= target_level:
            return [cell_id]

        children = self.refine_cell(cell_id)
        if not children:
            return [cell_id]

        for child_id in children:
            refined.extend(self._recursive_refine(child_id, target_level))

        return refined

    def _distance_to_cell(self, lat: float, lon: float, cell: GridCell) -> float:
        R = 6371.0
        clat, clon = cell.center

        lat1_rad = np.radians(lat)
        lat2_rad = np.radians(clat)
        dlat = lat2_rad - lat1_rad
        dlon = np.radians(clon - lon)

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def get_mesh_statistics(self) -> Dict[str, Any]:
        total = len(self.cells)
        active = len(self.active_cells)
        level_counts = {level: len(cells) for level, cells in self.level_cell_map.items()}
        active_by_level = {}
        for level in self.level_cell_map:
            active_by_level[level] = sum(
                1 for cid in self.level_cell_map[level]
                if cid in self.cells and not self.cells[cid].is_refined
            )

        sizes = []
        for cid in self.active_cells:
            sizes.append(self._cell_size_km(self.cells[cid]))

        return {
            "total_cells": total,
            "active_cells": active,
            "refined_cells": total - active,
            "levels": level_counts,
            "active_by_level": active_by_level,
            "min_cell_size_km": min(sizes) if sizes else 0,
            "max_cell_size_km": max(sizes) if sizes else 0,
            "mean_cell_size_km": np.mean(sizes) if sizes else 0,
            "effective_resolution_ratio": (1 / (active / len(self.level_cell_map.get(0, [1])))) if active > 0 else 1,
        }


class ExtremeWeatherDetector:
    def __init__(
        self,
        event_configs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.event_configs = event_configs or [
            {
                "type": ExtremeWeatherType.TYPHOON,
                "detection_vars": ["vorticity", "pressure", "wind_speed"],
                "thresholds": {"vorticity": 1e-4, "pressure": 99000, "wind_speed": 17},
                "focus_radius_km": 1000.0,
            },
            {
                "type": ExtremeWeatherType.HEATWAVE,
                "detection_vars": ["temperature_2m", "humidity"],
                "thresholds": {"temperature_2m": 305.15, "humidity": 0.6},
                "focus_radius_km": 2000.0,
            },
            {
                "type": ExtremeWeatherType.HEAVY_RAIN,
                "detection_vars": ["precipitation"],
                "thresholds": {"precipitation": 0.0001},
                "focus_radius_km": 500.0,
            },
        ]
        self._event_history: List[ExtremeWeatherEvent] = []

    def detect(self, dataset: xr.Dataset, timestamp: Optional[datetime] = None) -> List[ExtremeWeatherEvent]:
        events = []
        ts = timestamp or datetime.now()

        lat_name = None
        lon_name = None
        for ln in ["lat", "latitude"]:
            if ln in dataset.coords:
                lat_name = ln
                break
        for ln in ["lon", "longitude"]:
            if ln in dataset.coords:
                lon_name = ln
                break

        if lat_name is None or lon_name is None:
            return events

        lats = dataset[lat_name].values
        lons = dataset[lon_name].values

        for config in self.event_configs:
            event_type = config["type"]
            detection_vars = config["detection_vars"]
            thresholds = config["thresholds"]
            radius_km = config["focus_radius_km"]

            candidates = self._find_candidates(dataset, lats, lons, detection_vars, thresholds, lat_name, lon_name)

            for (lat_idx, lon_idx), intensity, var_values in candidates:
                event = ExtremeWeatherEvent(
                    event_type=event_type,
                    center_lat=float(lats[lat_idx]),
                    center_lon=float(lons[lon_idx]),
                    radius_km=radius_km,
                    intensity=float(intensity),
                    timestamp=ts,
                    variables=var_values,
                    confidence=min(1.0, intensity),
                )
                events.append(event)

        events = self._cluster_events(events)
        self._event_history.extend(events)
        return events

    def _find_candidates(
        self,
        dataset: xr.Dataset,
        lats: np.ndarray,
        lons: np.ndarray,
        variables: List[str],
        thresholds: Dict[str, float],
        lat_name: str,
        lon_name: str,
    ) -> List[Tuple[Tuple[int, int], float, Dict[str, float]]]:
        candidates = []

        for var in variables:
            if var not in dataset.data_vars:
                continue

        masks = []
        for var, threshold in thresholds.items():
            if var in dataset.data_vars:
                var_data = dataset[var]
                dims = var_data.dims
                lat_axis = dims.index(lat_name) if lat_name in dims else None
                lon_axis = dims.index(lon_name) if lon_name in dims else None

                if lat_axis is not None and lon_axis is not None:
                    slicer = [0] * var_data.ndim
                    slicer[lat_axis] = slice(None)
                    slicer[lon_axis] = slice(None)
                    data_2d = var_data.values[tuple(slicer)]
                    masks.append(data_2d > threshold if "vorticity" in var or "wind" in var else data_2d > threshold)

        if not masks:
            return candidates

        combined_mask = masks[0]
        for m in masks[1:]:
            if m.shape == combined_mask.shape:
                combined_mask = combined_mask & m

        if combined_mask.size == 0:
            return candidates

        thresholded = np.where(combined_mask)
        for lat_idx, lon_idx in zip(thresholded[0], thresholded[1]):
            var_values = {}
            intensity = 0.0
            n_vars = 0
            for var, threshold in thresholds.items():
                if var in dataset.data_vars:
                    var_data = dataset[var]
                    dims = var_data.dims
                    lat_axis = dims.index(lat_name) if lat_name in dims else None
                    lon_axis = dims.index(lon_name) if lon_name in dims else None
                    if lat_axis is not None and lon_axis is not None:
                        slicer = [0] * var_data.ndim
                        slicer[lat_axis] = lat_idx
                        slicer[lon_axis] = lon_idx
                        val = float(var_data.values[tuple(slicer)])
                        var_values[var] = val
                        if threshold > 0:
                            intensity += abs(val) / threshold
                        n_vars += 1
            if n_vars > 0:
                intensity /= n_vars
                candidates.append(((int(lat_idx), int(lon_idx)), intensity, var_values))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:20]

    def _cluster_events(self, events: List[ExtremeWeatherEvent]) -> List[ExtremeWeatherEvent]:
        if len(events) <= 1:
            return events

        clustered = []
        used = set()

        for i, event in enumerate(events):
            if i in used:
                continue
            cluster = [event]
            used.add(i)

            for j in range(i + 1, len(events)):
                if j in used:
                    continue
                if events[j].event_type == event.event_type:
                    dist = self._haversine(
                        event.center_lat, event.center_lon,
                        events[j].center_lat, events[j].center_lon
                    )
                    if dist < (event.radius_km + events[j].radius_km) / 2:
                        cluster.append(events[j])
                        used.add(j)

            if len(cluster) == 1:
                clustered.append(event)
            else:
                avg_lat = np.mean([e.center_lat for e in cluster])
                avg_lon = np.mean([e.center_lon for e in cluster])
                max_intensity = max(e.intensity for e in cluster)
                max_radius = max(e.radius_km for e in cluster)

                merged = ExtremeWeatherEvent(
                    event_type=cluster[0].event_type,
                    center_lat=float(avg_lat),
                    center_lon=float(avg_lon),
                    radius_km=max_radius,
                    intensity=float(max_intensity),
                    timestamp=cluster[0].timestamp,
                    variables={k: v for e in cluster for k, v in e.variables.items()},
                    confidence=max(e.confidence for e in cluster),
                    track=[(e.center_lat, e.center_lon) for e in cluster],
                )
                clustered.append(merged)

        return clustered

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = lat2_rad - lat1_rad
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    def get_event_history(self, event_type: Optional[ExtremeWeatherType] = None) -> List[ExtremeWeatherEvent]:
        if event_type is None:
            return self._event_history.copy()
        return [e for e in self._event_history if e.event_type == event_type]


class AdaptiveMeshRefiner:
    def __init__(
        self,
        mesh: AdaptiveMesh,
        detector: ExtremeWeatherDetector,
        focus_weight: float = 2.0,
    ):
        self.mesh = mesh
        self.detector = detector
        self.focus_weight = focus_weight

    def refine(
        self,
        dataset: xr.Dataset,
        timestamp: Optional[datetime] = None,
    ) -> MeshRefinementResult:
        ts = timestamp or datetime.now()

        events = self.detector.detect(dataset, ts)

        criterion_refinements = self.mesh.evaluate_criteria(dataset)

        refined_ids = set()
        for cid in criterion_refinements:
            new_ids = self.mesh.refine_cell(cid)
            refined_ids.update(new_ids)
            refined_ids.add(cid)

        for event in events:
            event_refined = self.mesh.refine_around_point(
                event.center_lat, event.center_lon, event.radius_km
            )
            refined_ids.update(event_refined)

        stats = self.mesh.get_mesh_statistics()
        stats["extreme_events_detected"] = len(events)
        stats["cells_refined_by_criteria"] = len(criterion_refinements)

        return MeshRefinementResult(
            mesh=self.mesh,
            refined_cells=list(refined_ids),
            extreme_events=events,
            refinement_stats=stats,
            timestamp=ts,
        )
