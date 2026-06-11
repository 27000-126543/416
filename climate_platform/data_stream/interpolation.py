"""
Spatial and temporal interpolation for meteorological data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import interpolate, ndimage

logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    LINEAR = "linear"
    NEAREST = "nearest"
    CUBIC = "cubic"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    KRIGING = "kriging"
    IDW = "idw"
    SPLINE = "spline"
    CONSERVATIVE = "conservative"


@dataclass
class InterpolationResult:
    dataset: xr.Dataset
    method: str
    source_resolution: Optional[Tuple[float, ...]] = None
    target_resolution: Optional[Tuple[float, ...]] = None
    variables_interpolated: List[str] = field(default_factory=list)
    computation_time: float = 0.0
    error_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class SpatialInterpolator:
    def __init__(
        self,
        method: InterpolationMethod = InterpolationMethod.BILINEAR,
        fill_value: float = np.nan,
        extrapolate: bool = False,
    ):
        self.method = method
        self.fill_value = fill_value
        self.extrapolate = extrapolate

    def _get_lat_lon(self, ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[int]]:
        lat = None
        lon = None
        lat_dim = None
        lon_dim = None

        for lat_name in ["lat", "latitude", "y"]:
            if lat_name in ds.coords:
                lat = ds[lat_name].values
                lat_dim = lat_name
                break

        for lon_name in ["lon", "longitude", "x"]:
            if lon_name in ds.coords:
                lon = ds[lon_name].values
                lon_dim = lon_name
                break

        return lat, lon, lat_dim, lon_dim

    def interpolate_to_grid(
        self,
        ds: xr.Dataset,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        variables: Optional[List[str]] = None,
    ) -> InterpolationResult:
        import time
        start_time = time.time()

        src_lat, src_lon, src_lat_dim, src_lon_dim = self._get_lat_lon(ds)
        if src_lat is None or src_lon is None:
            raise ValueError("Source dataset missing latitude/longitude coordinates")

        vars_to_interp = variables or list(ds.data_vars)
        result_ds = xr.Dataset()

        for coord_name, coord in ds.coords.items():
            if coord_name not in [src_lat_dim, src_lon_dim]:
                result_ds[coord_name] = coord

        result_ds[src_lat_dim] = (src_lat_dim, target_lat)
        result_ds[src_lon_dim] = (src_lon_dim, target_lon)

        error_metrics: Dict[str, Dict[str, float]] = {}

        for var_name in vars_to_interp:
            if var_name not in ds.data_vars:
                continue

            var_data = ds[var_name]
            interpolated = self._interpolate_variable(
                var_data, src_lat, src_lon, target_lat, target_lon,
                src_lat_dim, src_lon_dim
            )

            new_dims = list(var_data.dims)
            result_ds[var_name] = (new_dims, interpolated)

            error_metrics[var_name] = self._compute_error_metrics(
                var_data.values, interpolated
            )

        elapsed = time.time() - start_time

        src_res = (
            float(np.mean(np.diff(src_lat))) if len(src_lat) > 1 else None,
            float(np.mean(np.diff(src_lon))) if len(src_lon) > 1 else None,
        )
        tgt_res = (
            float(np.mean(np.diff(target_lat))) if len(target_lat) > 1 else None,
            float(np.mean(np.diff(target_lon))) if len(target_lon) > 1 else None,
        )

        return InterpolationResult(
            dataset=result_ds,
            method=self.method.value,
            source_resolution=src_res,
            target_resolution=tgt_res,
            variables_interpolated=vars_to_interp,
            computation_time=elapsed,
            error_metrics=error_metrics,
        )

    def _interpolate_variable(
        self,
        var_data: xr.DataArray,
        src_lat: np.ndarray,
        src_lon: np.ndarray,
        tgt_lat: np.ndarray,
        tgt_lon: np.ndarray,
        lat_dim: Optional[str],
        lon_dim: Optional[str],
    ) -> np.ndarray:
        data = var_data.values
        dims = var_data.dims

        if lat_dim not in dims or lon_dim not in dims:
            return data

        lat_axis = dims.index(lat_dim)
        lon_axis = dims.index(lon_dim)

        src_lon_grid, src_lat_grid = np.meshgrid(src_lon, src_lat)
        tgt_lon_grid, tgt_lat_grid = np.meshgrid(tgt_lon, tgt_lat)

        src_points = np.column_stack([src_lat_grid.ravel(), src_lon_grid.ravel()])
        tgt_points = np.column_stack([tgt_lat_grid.ravel(), tgt_lon_grid.ravel()])

        result_shape = list(data.shape)
        result_shape[lat_axis] = len(tgt_lat)
        result_shape[lon_axis] = len(tgt_lon)
        result = np.full(result_shape, self.fill_value, dtype=data.dtype)

        non_spatial_axes = [i for i in range(data.ndim) if i not in [lat_axis, lon_axis]]

        if non_spatial_axes:
            iterator_shape = [data.shape[i] for i in non_spatial_axes]
            for idx in np.ndindex(*iterator_shape):
                slicer = [slice(None)] * data.ndim
                for i, axis in enumerate(non_spatial_axes):
                    slicer[axis] = idx[i]
                slice_data = data[tuple(slicer)]
                interpolated_slice = self._interpolate_slice(
                    slice_data, src_lat, src_lon, tgt_lat, tgt_lon
                )
                result[tuple(slicer)] = interpolated_slice
        else:
            result = self._interpolate_slice(
                data, src_lat, src_lon, tgt_lat, tgt_lon
            )

        return result

    def _interpolate_slice(
        self,
        data_2d: np.ndarray,
        src_lat: np.ndarray,
        src_lon: np.ndarray,
        tgt_lat: np.ndarray,
        tgt_lon: np.ndarray,
    ) -> np.ndarray:
        if data_2d.shape[0] != len(src_lat) or data_2d.shape[1] != len(src_lon):
            return data_2d

        valid_mask = np.isfinite(data_2d)
        if not np.any(valid_mask):
            return np.full((len(tgt_lat), len(tgt_lon)), self.fill_value)

        if self.method == InterpolationMethod.NEAREST:
            return self._nearest_interp(data_2d, src_lat, src_lon, tgt_lat, tgt_lon)
        elif self.method == InterpolationMethod.BILINEAR or self.method == InterpolationMethod.LINEAR:
            return self._bilinear_interp(data_2d, src_lat, src_lon, tgt_lat, tgt_lon)
        elif self.method == InterpolationMethod.BICUBIC or self.method == InterpolationMethod.CUBIC:
            return self._bicubic_interp(data_2d, src_lat, src_lon, tgt_lat, tgt_lon)
        elif self.method == InterpolationMethod.IDW:
            return self._idw_interp(data_2d, src_lat, src_lon, tgt_lat, tgt_lon)
        else:
            return self._bilinear_interp(data_2d, src_lat, src_lon, tgt_lat, tgt_lon)

    def _nearest_interp(
        self, data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray,
        tgt_lat: np.ndarray, tgt_lon: np.ndarray
    ) -> np.ndarray:
        data_filled = np.nan_to_num(data, nan=np.nanmean(data) if np.any(np.isfinite(data)) else 0)
        try:
            f = interpolate.RegularGridInterpolator(
                (src_lat, src_lon), data_filled, method="nearest",
                bounds_error=False, fill_value=self.fill_value
            )
            tgt_lon_grid, tgt_lat_grid = np.meshgrid(tgt_lon, tgt_lat)
            pts = np.column_stack([tgt_lat_grid.ravel(), tgt_lon_grid.ravel()])
            return f(pts).reshape(len(tgt_lat), len(tgt_lon))
        except Exception:
            return np.full((len(tgt_lat), len(tgt_lon)), self.fill_value)

    def _bilinear_interp(
        self, data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray,
        tgt_lat: np.ndarray, tgt_lon: np.ndarray
    ) -> np.ndarray:
        data_filled = data.copy()
        if np.any(~np.isfinite(data_filled)):
            valid_mask = np.isfinite(data_filled)
            mean_val = np.nanmean(data_filled) if np.any(valid_mask) else 0
            data_filled[~valid_mask] = mean_val
        try:
            f = interpolate.RegularGridInterpolator(
                (src_lat, src_lon), data_filled, method="linear",
                bounds_error=False, fill_value=self.fill_value
            )
            tgt_lon_grid, tgt_lat_grid = np.meshgrid(tgt_lon, tgt_lat)
            pts = np.column_stack([tgt_lat_grid.ravel(), tgt_lon_grid.ravel()])
            return f(pts).reshape(len(tgt_lat), len(tgt_lon))
        except Exception:
            return np.full((len(tgt_lat), len(tgt_lon)), self.fill_value)

    def _bicubic_interp(
        self, data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray,
        tgt_lat: np.ndarray, tgt_lon: np.ndarray
    ) -> np.ndarray:
        data_filled = data.copy()
        if np.any(~np.isfinite(data_filled)):
            valid_mask = np.isfinite(data_filled)
            mean_val = np.nanmean(data_filled) if np.any(valid_mask) else 0
            data_filled[~valid_mask] = mean_val
        try:
            f = interpolate.RectBivariateSpline(src_lat, src_lon, data_filled, kx=3, ky=3)
            return f(tgt_lat, tgt_lon)
        except Exception:
            return self._bilinear_interp(data, src_lat, src_lon, tgt_lat, tgt_lon)

    def _idw_interp(
        self, data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray,
        tgt_lat: np.ndarray, tgt_lon: np.ndarray, power: float = 2.0
    ) -> np.ndarray:
        src_lon_grid, src_lat_grid = np.meshgrid(src_lon, src_lat)
        tgt_lon_grid, tgt_lat_grid = np.meshgrid(tgt_lon, tgt_lat)

        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            return np.full((len(tgt_lat), len(tgt_lon)), self.fill_value)

        src_pts = np.column_stack([src_lat_grid[valid_mask], src_lon_grid[valid_mask]])
        src_vals = data[valid_mask]

        result = np.zeros((len(tgt_lat), len(tgt_lon)))
        weights_sum = np.zeros_like(result)

        for i, (lat, val) in enumerate(zip(src_pts, src_vals)):
            dx = tgt_lon_grid - lat[1]
            dy = tgt_lat_grid - lat[0]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            dist = np.maximum(dist, 1e-10)
            w = 1.0 / (dist ** power)
            result += w * val
            weights_sum += w

        return np.where(weights_sum > 0, result / weights_sum, self.fill_value)

    def _compute_error_metrics(self, source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        if source.size != target.size:
            return {}

        src_flat = source.ravel()
        tgt_flat = target.ravel()
        valid = np.isfinite(src_flat) & np.isfinite(tgt_flat)

        if not np.any(valid):
            return {}

        diff = src_flat[valid] - tgt_flat[valid]
        return {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "max_error": float(np.max(np.abs(diff))),
        }

    def regrid_conservative(
        self,
        ds: xr.Dataset,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        variables: Optional[List[str]] = None,
    ) -> InterpolationResult:
        import time
        start_time = time.time()

        src_lat, src_lon, src_lat_dim, src_lon_dim = self._get_lat_lon(ds)
        if src_lat is None or src_lon is None:
            raise ValueError("Source dataset missing latitude/longitude coordinates")

        vars_to_interp = variables or list(ds.data_vars)
        result_ds = xr.Dataset()

        for coord_name, coord in ds.coords.items():
            if coord_name not in [src_lat_dim, src_lon_dim]:
                result_ds[coord_name] = coord

        result_ds[src_lat_dim] = (src_lat_dim, target_lat)
        result_ds[src_lon_dim] = (src_lon_dim, target_lon)

        src_lat_edges = self._cell_edges(src_lat)
        src_lon_edges = self._cell_edges(src_lon)
        tgt_lat_edges = self._cell_edges(target_lat)
        tgt_lon_edges = self._cell_edges(target_lon)

        error_metrics: Dict[str, Dict[str, float]] = {}

        for var_name in vars_to_interp:
            if var_name not in ds.data_vars:
                continue

            var_data = ds[var_name].values
            dims = ds[var_name].dims

            if src_lat_dim in dims and src_lon_dim in dims:
                lat_axis = dims.index(src_lat_dim)
                lon_axis = dims.index(src_lon_dim)
                new_shape = list(var_data.shape)
                new_shape[lat_axis] = len(target_lat)
                new_shape[lon_axis] = len(target_lon)
                regridded = np.zeros(new_shape)

                non_spatial_axes = [i for i in range(var_data.ndim) if i not in [lat_axis, lon_axis]]
                if non_spatial_axes:
                    iterator_shape = [var_data.shape[i] for i in non_spatial_axes]
                    for idx in np.ndindex(*iterator_shape):
                        slicer = [slice(None)] * var_data.ndim
                        for i, axis in enumerate(non_spatial_axes):
                            slicer[axis] = idx[i]
                        slice_data = var_data[tuple(slicer)]
                        regridded[tuple(slicer)] = self._conserve_regrid_slice(
                            slice_data, src_lat_edges, src_lon_edges,
                            tgt_lat_edges, tgt_lon_edges
                        )
                else:
                    regridded = self._conserve_regrid_slice(
                        var_data, src_lat_edges, src_lon_edges,
                        tgt_lat_edges, tgt_lon_edges
                    )

                result_ds[var_name] = (dims, regridded)
                error_metrics[var_name] = self._compute_error_metrics(var_data, regridded)

        elapsed = time.time() - start_time
        return InterpolationResult(
            dataset=result_ds,
            method="conservative",
            source_resolution=(float(np.mean(np.diff(src_lat))), float(np.mean(np.diff(src_lon)))),
            target_resolution=(float(np.mean(np.diff(target_lat))), float(np.mean(np.diff(target_lon)))),
            variables_interpolated=vars_to_interp,
            computation_time=elapsed,
            error_metrics=error_metrics,
        )

    @staticmethod
    def _cell_edges(centers: np.ndarray) -> np.ndarray:
        if len(centers) < 2:
            delta = 1.0
        else:
            delta = np.diff(centers).mean()
        edges = np.zeros(len(centers) + 1)
        edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
        edges[0] = centers[0] - delta / 2
        edges[-1] = centers[-1] + delta / 2
        return edges

    @staticmethod
    def _conserve_regrid_slice(
        data: np.ndarray,
        src_lat_edges: np.ndarray,
        src_lon_edges: np.ndarray,
        tgt_lat_edges: np.ndarray,
        tgt_lon_edges: np.ndarray,
    ) -> np.ndarray:
        result = np.zeros((len(tgt_lat_edges) - 1, len(tgt_lon_edges) - 1))
        result_weights = np.zeros_like(result)

        for i in range(len(tgt_lat_edges) - 1):
            for j in range(len(tgt_lon_edges) - 1):
                lat_overlap = np.minimum(src_lat_edges[1:], tgt_lat_edges[i + 1]) - np.maximum(src_lat_edges[:-1], tgt_lat_edges[i])
                lon_overlap = np.minimum(src_lon_edges[1:], tgt_lon_edges[j + 1]) - np.maximum(src_lon_edges[:-1], tgt_lon_edges[j])
                lat_overlap = np.maximum(lat_overlap, 0)
                lon_overlap = np.maximum(lon_overlap, 0)
                weights = np.outer(lat_overlap, lon_overlap)

                valid_data = np.nan_to_num(data, nan=0.0)
                if weights.sum() > 0:
                    result[i, j] = np.sum(valid_data * weights) / weights.sum()
                    result_weights[i, j] = weights.sum()

        return np.where(result_weights > 0, result, np.nan)


class TemporalInterpolator:
    def __init__(
        self,
        method: InterpolationMethod = InterpolationMethod.LINEAR,
        fill_value: float = np.nan,
    ):
        self.method = method
        self.fill_value = fill_value

    def interpolate_to_times(
        self,
        ds: xr.Dataset,
        target_times: np.ndarray,
        variables: Optional[List[str]] = None,
    ) -> InterpolationResult:
        import time
        start_time = time.time()

        if "time" not in ds.coords:
            raise ValueError("Dataset missing time coordinate")

        src_times = ds["time"].values
        vars_to_interp = variables or list(ds.data_vars)

        result_ds = xr.Dataset()
        for coord_name, coord in ds.coords.items():
            if coord_name != "time":
                result_ds[coord_name] = coord
        result_ds["time"] = ("time", target_times)

        error_metrics: Dict[str, Dict[str, float]] = {}

        for var_name in vars_to_interp:
            if var_name not in ds.data_vars:
                continue

            var_data = ds[var_name]
            dims = var_data.dims

            if "time" not in dims:
                result_ds[var_name] = var_data
                continue

            time_axis = dims.index("time")
            result_shape = list(var_data.shape)
            result_shape[time_axis] = len(target_times)
            result = np.full(result_shape, self.fill_value, dtype=var_data.dtype)

            non_time_axes = [i for i in range(var_data.ndim) if i != time_axis]
            if non_time_axes:
                iterator_shape = [var_data.shape[i] for i in non_time_axes]
                for idx in np.ndindex(*iterator_shape):
                    slicer = [slice(None)] * var_data.ndim
                    for i, axis in enumerate(non_time_axes):
                        slicer[axis] = idx[i]
                    series = var_data.values[tuple(slicer)]
                    result[tuple(slicer)] = self._interpolate_series(
                        src_times, series, target_times
                    )
            else:
                result = self._interpolate_series(src_times, var_data.values, target_times)

            result_ds[var_name] = (dims, result)
            if var_data.size == result.size:
                error_metrics[var_name] = self._compute_error_metrics(var_data.values, result)

        elapsed = time.time() - start_time
        src_dt = float(np.mean(np.diff(src_times.astype("datetime64[h]").astype(float)))) if len(src_times) > 1 else None
        tgt_dt = float(np.mean(np.diff(target_times.astype("datetime64[h]").astype(float)))) if len(target_times) > 1 else None

        return InterpolationResult(
            dataset=result_ds,
            method=self.method.value,
            source_resolution=(src_dt,) if src_dt else None,
            target_resolution=(tgt_dt,) if tgt_dt else None,
            variables_interpolated=vars_to_interp,
            computation_time=elapsed,
            error_metrics=error_metrics,
        )

    def _interpolate_series(
        self,
        src_times: np.ndarray,
        src_values: np.ndarray,
        tgt_times: np.ndarray,
    ) -> np.ndarray:
        src_times_numeric = src_times.astype("datetime64[s]").astype(float)
        tgt_times_numeric = tgt_times.astype("datetime64[s]").astype(float)

        valid_mask = np.isfinite(src_values)
        if not np.any(valid_mask):
            return np.full(len(tgt_times), self.fill_value)

        src_t_valid = src_times_numeric[valid_mask]
        src_v_valid = src_values[valid_mask]

        if len(src_v_valid) < 2:
            return np.full(len(tgt_times), src_v_valid[0] if len(src_v_valid) == 1 else self.fill_value)

        try:
            if self.method == InterpolationMethod.NEAREST:
                f = interpolate.interp1d(src_t_valid, src_v_valid, kind="nearest",
                                         bounds_error=False, fill_value="extrapolate" if self._can_extrapolate(src_t_valid, src_v_valid) else self.fill_value)
            elif self.method == InterpolationMethod.CUBIC or self.method == InterpolationMethod.SPLINE:
                if len(src_v_valid) >= 4:
                    f = interpolate.CubicSpline(src_t_valid, src_v_valid, extrapolate=True)
                else:
                    f = interpolate.interp1d(src_t_valid, src_v_valid, kind="linear",
                                             bounds_error=False, fill_value="extrapolate")
            else:
                f = interpolate.interp1d(src_t_valid, src_v_valid, kind="linear",
                                         bounds_error=False, fill_value="extrapolate")

            result = f(tgt_times_numeric)
            return np.where(np.isfinite(result), result, self.fill_value)
        except Exception:
            return np.full(len(tgt_times), self.fill_value)

    @staticmethod
    def _can_extrapolate(times: np.ndarray, values: np.ndarray) -> bool:
        return len(times) >= 2 and np.std(values) > 0

    @staticmethod
    def _compute_error_metrics(source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        if source.size != target.size:
            return {}

        src_flat = source.ravel()
        tgt_flat = target.ravel()
        valid = np.isfinite(src_flat) & np.isfinite(tgt_flat)

        if not np.any(valid):
            return {}

        diff = src_flat[valid] - tgt_flat[valid]
        return {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "max_error": float(np.max(np.abs(diff))),
        }
