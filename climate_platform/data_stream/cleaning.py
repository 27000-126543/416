"""
Real-time data cleaning module with quality control.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QCFailure:
    variable: str
    check_name: str
    message: str
    indices: Optional[np.ndarray] = None
    num_failures: int = 0


@dataclass
class QualityControlResult:
    dataset: xr.Dataset
    passed: bool
    overall_quality: float
    variable_quality: Dict[str, float] = field(default_factory=dict)
    failures: List[QCFailure] = field(default_factory=list)
    cleaned: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def num_failures(self) -> int:
        return len(self.failures)

    @property
    def failure_summary(self) -> Dict[str, int]:
        summary = {}
        for f in self.failures:
            key = f"{f.variable}:{f.check_name}"
            summary[key] = summary.get(key, 0) + f.num_failures
        return summary


class DataCleaner:
    PHYSICAL_RANGES = {
        "temperature": (180.0, 330.0),
        "temperature_2m": (180.0, 320.0),
        "temperature_surface": (180.0, 330.0),
        "water_temperature": (271.0, 305.0),
        "sea_surface_temperature": (271.0, 305.0),
        "skin_temperature": (200.0, 330.0),
        "sea_ice_temperature": (200.0, 273.0),
        "relative_humidity": (0.0, 100.0),
        "specific_humidity": (0.0, 0.05),
        "pressure": (87000.0, 108500.0),
        "pressure_surface": (87000.0, 108500.0),
        "pressure_sea_level": (87000.0, 108500.0),
        "surface_pressure": (87000.0, 108500.0),
        "wind_speed": (0.0, 150.0),
        "wind_speed_10m": (0.0, 85.0),
        "u_wind": (-150.0, 150.0),
        "v_wind": (-150.0, 150.0),
        "wind_direction": (0.0, 360.0),
        "wind_direction_10m": (0.0, 360.0),
        "precipitation": (0.0, 1000.0),
        "salinity": (0.0, 42.0),
        "geopotential_height": (0.0, 50000.0),
        "vorticity": (-1e-2, 1e-2),
        "divergence": (-1e-3, 1e-3),
        "sea_ice_concentration": (0.0, 1.0),
        "sea_ice_thickness": (0.0, 50.0),
        "soil_moisture": (0.0, 1.0),
        "albedo": (0.0, 1.0),
        "cloud_cover": (0.0, 1.0),
    }

    def __init__(
        self,
        quality_threshold: float = 0.85,
        enable_range_check: bool = True,
        enable_gradient_check: bool = True,
        enable_spatial_consistency: bool = True,
        enable_temporal_consistency: bool = True,
        enable_buddy_check: bool = True,
        auto_clean: bool = True,
    ):
        self.quality_threshold = quality_threshold
        self.enable_range_check = enable_range_check
        self.enable_gradient_check = enable_gradient_check
        self.enable_spatial_consistency = enable_spatial_consistency
        self.enable_temporal_consistency = enable_temporal_consistency
        self.enable_buddy_check = enable_buddy_check
        self.auto_clean = auto_clean
        self._history: Dict[str, List[xr.Dataset]] = {}

    def check_range(self, ds: xr.Dataset, var_name: str, data: np.ndarray) -> Optional[QCFailure]:
        if var_name not in self.PHYSICAL_RANGES:
            return None

        min_val, max_val = self.PHYSICAL_RANGES[var_name]
        valid = np.isfinite(data)
        out_of_range = valid & ((data < min_val) | (data > max_val))

        if np.any(out_of_range):
            bad_count = int(np.sum(out_of_range))
            return QCFailure(
                variable=var_name,
                check_name="range_check",
                message=f"Values outside physical range [{min_val}, {max_val}]",
                indices=np.where(out_of_range),
                num_failures=bad_count,
            )
        return None

    def check_gradient(
        self,
        ds: xr.Dataset,
        var_name: str,
        data: np.ndarray,
        max_gradient: Optional[float] = None,
    ) -> Optional[QCFailure]:
        if data.ndim < 2:
            return None

        if max_gradient is None:
            if "temperature" in var_name:
                max_gradient = 15.0
            elif "pressure" in var_name:
                max_gradient = 500.0
            elif "wind" in var_name:
                max_gradient = 30.0
            else:
                max_gradient = np.nanstd(data) * 5 if np.nanstd(data) > 0 else 1.0

        try:
            if "lat" in ds.coords and "lon" in ds.coords:
                lat_axis = ds.dims.index("lat") if "lat" in ds.dims else None
                lon_axis = ds.dims.index("lon") if "lon" in ds.dims else None

                if lat_axis is not None and lon_axis is not None:
                    grad_lat = np.abs(np.gradient(data, axis=lat_axis))
                    grad_lon = np.abs(np.gradient(data, axis=lon_axis))
                    total_grad = np.sqrt(grad_lat ** 2 + grad_lon ** 2)

                    extreme_grad = total_grad > max_gradient
                    if np.any(extreme_grad):
                        bad_count = int(np.sum(extreme_grad))
                        return QCFailure(
                            variable=var_name,
                            check_name="gradient_check",
                            message=f"Spatial gradient exceeds {max_gradient}",
                            indices=np.where(extreme_grad),
                            num_failures=bad_count,
                        )
        except Exception:
            pass
        return None

    def check_temporal_consistency(
        self,
        ds: xr.Dataset,
        var_name: str,
        data: np.ndarray,
    ) -> Optional[QCFailure]:
        if "time" not in ds.dims or ds.sizes["time"] < 3:
            return None

        time_axis = ds.dims.index("time")
        try:
            diff1 = np.diff(data, axis=time_axis)
            diff2 = np.diff(diff1, axis=time_axis)

            std_diff = np.nanstd(diff2) if np.nanstd(diff2) > 0 else 1.0
            threshold = std_diff * 5
            spikes = np.abs(diff2) > threshold

            if np.any(spikes):
                bad_count = int(np.sum(spikes))
                return QCFailure(
                    variable=var_name,
                    check_name="temporal_consistency",
                    message=f"Temporal discontinuities detected (threshold={threshold:.4f})",
                    indices=np.where(spikes),
                    num_failures=bad_count,
                )
        except Exception:
            pass
        return None

    def check_spatial_consistency(
        self,
        ds: xr.Dataset,
        var_name: str,
        data: np.ndarray,
    ) -> Optional[QCFailure]:
        if data.ndim < 2:
            return None

        try:
            mean = np.nanmean(data)
            std = np.nanstd(data)
            if std == 0 or not np.isfinite(std):
                return None

            z_scores = np.abs((data - mean) / std)
            outliers = z_scores > 6

            if np.any(outliers):
                bad_count = int(np.sum(outliers))
                return QCFailure(
                    variable=var_name,
                    check_name="spatial_consistency",
                    message="Spatial outliers detected (z-score > 6 sigma)",
                    indices=np.where(outliers),
                    num_failures=bad_count,
                )
        except Exception:
            pass
        return None

    def _clean_variable(self, data: np.ndarray, failures: List[QCFailure]) -> np.ndarray:
        cleaned = data.copy()
        for failure in failures:
            if failure.indices is not None and len(failure.indices) > 0:
                cleaned[failure.indices] = np.nan
        return cleaned

    def _interpolate_nans(self, data: np.ndarray, ds: xr.Dataset, var_name: str) -> np.ndarray:
        if not np.any(np.isnan(data)):
            return data

        cleaned = data.copy()

        try:
            if "time" in ds.dims and data.ndim >= 1:
                time_axis = ds.dims.index("time")
                if data.ndim == 1:
                    x = np.arange(len(cleaned))
                    valid = ~np.isnan(cleaned)
                    if np.sum(valid) >= 2:
                        cleaned = np.interp(x, x[valid], cleaned[valid])
                elif data.ndim > 1:
                    other_shape = list(cleaned.shape)
                    other_shape.pop(time_axis)
                    for idx in np.ndindex(*other_shape):
                        slicer = list(idx)
                        slicer.insert(time_axis, slice(None))
                        series = cleaned[tuple(slicer)]
                        valid = ~np.isnan(series)
                        if np.sum(valid) >= 2 and np.sum(~valid) > 0:
                            x = np.arange(len(series))
                            cleaned[tuple(slicer)] = np.interp(x, x[valid], series[valid])
        except Exception as e:
            logger.debug(f"Temporal interpolation failed for {var_name}: {e}")

        if not np.any(np.isnan(cleaned)):
            return cleaned

        try:
            has_lat = "lat" in ds.dims
            has_lon = "lon" in ds.dims
            if has_lat and has_lon and cleaned.ndim >= 2:
                lat_axis = ds.dims.index("lat")
                lon_axis = ds.dims.index("lon")

                for _ in range(10):
                    if not np.any(np.isnan(cleaned)):
                        break
                    new_cleaned = cleaned.copy()
                    nan_mask = np.isnan(cleaned)

                    for axis in sorted([lat_axis, lon_axis]):
                        shift_fwd = np.roll(cleaned, 1, axis=axis)
                        shift_bwd = np.roll(cleaned, -1, axis=axis)
                        with np.errstate(invalid="ignore"):
                            avg_axis = np.where(
                                np.isfinite(shift_fwd) & np.isfinite(shift_bwd),
                                (shift_fwd + shift_bwd) / 2,
                                np.where(np.isfinite(shift_fwd), shift_fwd, shift_bwd)
                            )
                        new_mask = np.isnan(new_cleaned) & ~np.isnan(avg_axis)
                        new_cleaned[new_mask] = avg_axis[new_mask]

                    cleaned = new_cleaned
        except Exception as e:
            logger.debug(f"Spatial interpolation failed for {var_name}: {e}")

        if np.any(np.isnan(cleaned)):
            try:
                valid_mean = np.nanmean(cleaned)
                if np.isfinite(valid_mean):
                    cleaned[np.isnan(cleaned)] = valid_mean
            except Exception:
                pass

        return cleaned

    def run_qc(self, ds: xr.Dataset) -> QualityControlResult:
        failures: List[QCFailure] = []
        variable_quality: Dict[str, float] = {}

        for var_name in ds.data_vars:
            var_failures: List[QCFailure] = []
            data = ds[var_name].values

            if self.enable_range_check:
                f = self.check_range(ds, var_name, data)
                if f:
                    var_failures.append(f)

            if self.enable_gradient_check:
                f = self.check_gradient(ds, var_name, data)
                if f:
                    var_failures.append(f)

            if self.enable_spatial_consistency:
                f = self.check_spatial_consistency(ds, var_name, data)
                if f:
                    var_failures.append(f)

            if self.enable_temporal_consistency:
                f = self.check_temporal_consistency(ds, var_name, data)
                if f:
                    var_failures.append(f)

            failures.extend(var_failures)

            total_points = data.size
            failed_points = sum(f.num_failures for f in var_failures)
            if total_points > 0:
                variable_quality[var_name] = max(0.0, 1.0 - failed_points / total_points)
            else:
                variable_quality[var_name] = 1.0

        overall_quality = (
            np.mean(list(variable_quality.values())) if variable_quality else 1.0
        )
        passed = overall_quality >= self.quality_threshold

        result = QualityControlResult(
            dataset=ds,
            passed=passed,
            overall_quality=overall_quality,
            variable_quality=variable_quality,
            failures=failures,
            cleaned=False,
        )

        if self.auto_clean:
            result = self.clean(result)

        if var_name:
            history_key = "default"
            if history_key not in self._history:
                self._history[history_key] = []
            self._history[history_key].append(ds)

        return result

    def clean(self, qc_result: QualityControlResult) -> QualityControlResult:
        if not qc_result.failures:
            qc_result.cleaned = True
            return qc_result

        ds = qc_result.dataset
        cleaned_ds = ds.copy()

        var_failures: Dict[str, List[QCFailure]] = {}
        for f in qc_result.failures:
            if f.variable not in var_failures:
                var_failures[f.variable] = []
            var_failures[f.variable].append(f)

        for var_name, failures in var_failures.items():
            if var_name in cleaned_ds.data_vars:
                data = cleaned_ds[var_name].values
                cleaned_data = self._clean_variable(data, failures)
                cleaned_data = self._interpolate_nans(cleaned_data, cleaned_ds, var_name)
                cleaned_ds[var_name].values = cleaned_data

        return QualityControlResult(
            dataset=cleaned_ds,
            passed=qc_result.passed,
            overall_quality=qc_result.overall_quality,
            variable_quality=qc_result.variable_quality,
            failures=[],
            cleaned=True,
        )

    def validate_dataset(self, ds: xr.Dataset) -> Tuple[bool, List[str]]:
        issues = []

        if not isinstance(ds, xr.Dataset):
            return False, ["Input is not an xarray Dataset"]

        if len(ds.data_vars) == 0:
            issues.append("Dataset contains no data variables")

        for var_name, var in ds.data_vars.items():
            if np.all(np.isnan(var.values)):
                issues.append(f"Variable '{var_name}' contains only NaN values")
            if var.size == 0:
                issues.append(f"Variable '{var_name}' is empty")

        return len(issues) == 0, issues
