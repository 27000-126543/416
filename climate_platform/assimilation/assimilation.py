"""
Data assimilation implementations: EnKF, 3DVar, 4DVar.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


class AssimilationMethod(Enum):
    ENKF = "enkf"
    ENSEMBLE_TRANSFORM_KF = "etkf"
    THREEDVAR = "3dvar"
    FOURDVAR = "4dvar"
    HYBRID = "hybrid"
    LETKF = "letkf"


@dataclass
class Observation:
    value: float
    error: float
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    timestamp: Optional[datetime] = None
    variable: str = ""
    observation_type: str = ""


@dataclass
class AssimilationResult:
    analysis: xr.Dataset
    background: xr.Dataset
    innovations: Optional[xr.Dataset] = None
    analysis_increment: Optional[xr.Dataset] = None
    ensemble_members: Optional[List[xr.Dataset]] = None
    method: str = ""
    computation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    diagnostics: Dict[str, float] = field(default_factory=dict)
    observations_assimilated: int = 0
    observations_rejected: int = 0


class Localization:
    def __init__(self, radius_km: float = 500.0, taper_function: str = "gaspari_cohn"):
        self.radius_km = radius_km
        self.taper_function = taper_function

    def compute_weights(
        self,
        model_lat: np.ndarray,
        model_lon: np.ndarray,
        obs_lat: float,
        obs_lon: float,
    ) -> np.ndarray:
        lon_grid, lat_grid = np.meshgrid(model_lon, model_lat)
        distances = self._haversine_distance(lat_grid, lon_grid, obs_lat, obs_lon)
        return self._taper(distances / self.radius_km)

    @staticmethod
    def _haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = lat2_rad - lat1_rad
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def _taper(self, normalized_distance: np.ndarray) -> np.ndarray:
        if self.taper_function == "gaspari_cohn":
            return self._gaspari_cohn(normalized_distance)
        elif self.taper_function == "gaussian":
            return np.exp(-normalized_distance ** 2)
        else:
            return np.where(normalized_distance <= 1.0, 1.0 - normalized_distance, 0.0)

    @staticmethod
    def _gaspari_cohn(c: np.ndarray) -> np.ndarray:
        result = np.zeros_like(c)
        mask1 = c <= 1.0
        mask2 = (c > 1.0) & (c <= 2.0)
        result[mask1] = (
            -(c[mask1] ** 5) / 4 + c[mask1] ** 4 / 2 + 5 * c[mask1] ** 3 / 8
            - 5 * c[mask1] ** 2 / 3 + 1
        )
        result[mask2] = (
            c[mask2] ** 5 / 12 - c[mask2] ** 4 / 2 + 5 * c[mask2] ** 3 / 8
            + 5 * c[mask2] ** 2 / 3 - 5 * c[mask2] + 4 - 2 / (3 * c[mask2])
        )
        return result


class CovarianceInflation:
    def __init__(self, method: str = "multiplicative", factor: float = 1.05):
        self.method = method
        self.factor = factor

    def apply(self, ensemble: List[np.ndarray]) -> List[np.ndarray]:
        if len(ensemble) < 2:
            return ensemble

        n_members = len(ensemble)
        ensemble_mean = np.mean(ensemble, axis=0)

        if self.method == "multiplicative":
            inflated = []
            for member in ensemble:
                deviation = member - ensemble_mean
                inflated.append(ensemble_mean + self.factor * deviation)
            return inflated
        elif self.method == "additive":
            noise_std = np.std(ensemble, axis=0) * 0.01
            inflated = []
            for member in ensemble:
                noise = np.random.randn(*member.shape) * noise_std
                inflated.append(member + noise)
            return inflated
        elif self.method == "relaxation":
            inflated = []
            for member in ensemble:
                deviation = member - ensemble_mean
                inflated.append(ensemble_mean + self.factor * deviation)
            return inflated
        else:
            return ensemble


class ObservationOperator:
    def __init__(self, variables_map: Optional[Dict[str, str]] = None):
        self.variables_map = variables_map or {}

    def interpolate_model_to_obs(
        self,
        model_state: xr.Dataset,
        obs_lat: float,
        obs_lon: float,
        variable: str,
        obs_altitude: Optional[float] = None,
    ) -> Tuple[float, Optional[float]]:
        if variable not in model_state.data_vars:
            mapped_var = self.variables_map.get(variable, variable)
            if mapped_var not in model_state.data_vars:
                return np.nan, None
            variable = mapped_var

        var_data = model_state[variable]
        dims = var_data.dims

        lat_name = None
        lon_name = None
        for ln in ["lat", "latitude"]:
            if ln in var_data.coords:
                lat_name = ln
                break
        for ln in ["lon", "longitude"]:
            if ln in var_data.coords:
                lon_name = ln
                break

        if lat_name is None or lon_name is None:
            if var_data.size > 0:
                return float(var_data.values.ravel()[0]), None
            return np.nan, None

        lat_values = var_data[lat_name].values
        lon_values = var_data[lon_name].values

        lat_idx = np.argmin(np.abs(lat_values - obs_lat))
        lon_idx = np.argmin(np.abs(lon_values - obs_lon))

        try:
            lat_axis = dims.index(lat_name)
            lon_axis = dims.index(lon_name)
            slicer = [slice(None)] * var_data.ndim
            slicer[lat_axis] = lat_idx
            slicer[lon_axis] = lon_idx

            if "level" in dims and obs_altitude is not None:
                level_axis = dims.index("level")
                if "level" in var_data.coords:
                    level_values = var_data["level"].values
                    level_idx = np.argmin(np.abs(level_values - obs_altitude))
                    slicer[level_axis] = level_idx

            value = var_data.values[tuple(slicer)]
            if isinstance(value, np.ndarray) and value.size == 1:
                value = float(value.ravel()[0])
            elif isinstance(value, np.ndarray):
                value = float(np.nanmean(value))
            return float(value), None
        except Exception:
            return np.nan, None


class DataAssimilationBase(ABC):
    def __init__(
        self,
        localization: Optional[Localization] = None,
        inflation: Optional[CovarianceInflation] = None,
        obs_operator: Optional[ObservationOperator] = None,
    ):
        self.localization = localization or Localization()
        self.inflation = inflation or CovarianceInflation()
        self.obs_operator = obs_operator or ObservationOperator()
        self._time_window = timedelta(hours=6)

    @abstractmethod
    def assimilate(
        self,
        background: xr.Dataset,
        observations: List[Observation],
        ensemble_members: Optional[List[xr.Dataset]] = None,
        time_window: Optional[Tuple[datetime, datetime]] = None,
    ) -> AssimilationResult:
        pass


class EnKF(DataAssimilationBase):
    def __init__(self, ensemble_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.ensemble_size = ensemble_size

    def assimilate(
        self,
        background: xr.Dataset,
        observations: List[Observation],
        ensemble_members: Optional[List[xr.Dataset]] = None,
        time_window: Optional[Tuple[datetime, datetime]] = None,
    ) -> AssimilationResult:
        import time
        start_time = time.time()

        if ensemble_members is None:
            ensemble_members = self._generate_initial_ensemble(background)

        n_members = len(ensemble_members)
        if n_members < 2:
            return AssimilationResult(
                analysis=background,
                background=background,
                method="enkf",
                computation_time=time.time() - start_time,
                diagnostics={"ensemble_size": n_members},
            )

        analysis_members = []
        lat_name = None
        lon_name = None
        for ln in ["lat", "latitude"]:
            if ln in background.coords:
                lat_name = ln
                break
        for ln in ["lon", "longitude"]:
            if ln in background.coords:
                lon_name = ln
                break

        model_lat = background[lat_name].values if lat_name else np.array([0.0])
        model_lon = background[lon_name].values if lon_name else np.array([0.0])

        member_arrays = []
        for member in ensemble_members:
            member_data = {}
            for var in member.data_vars:
                member_data[var] = member[var].values.copy()
            member_arrays.append(member_data)

        n_obs_assimilated = 0
        n_obs_rejected = 0

        for obs in observations:
            if obs.value is None or not np.isfinite(obs.value):
                n_obs_rejected += 1
                continue

            loc_weights = self.localization.compute_weights(
                model_lat, model_lon, obs.latitude, obs.longitude
            )

            hx_ensemble = []
            for member_data in member_arrays:
                if obs.variable in member_data:
                    model_val = self._extract_point(member_data[obs.variable], background[obs.variable].dims,
                                                    model_lat, model_lon, obs.latitude, obs.longitude,
                                                    lat_name, lon_name)
                    hx_ensemble.append(model_val)
                else:
                    hx_ensemble.append(np.nan)

            hx_array = np.array(hx_ensemble)
            valid_mask = np.isfinite(hx_array)
            if np.sum(valid_mask) < 2:
                n_obs_rejected += 1
                continue

            hx_mean = np.mean(hx_array[valid_mask])
            hx_pert = hx_array[valid_mask] - hx_mean

            innovation = obs.value - hx_mean
            obs_error = obs.error if obs.error > 0 else 1.0

            for var_name in list(member_arrays[0].keys()):
                var_shape = member_arrays[0][var_name].shape
                var_ensemble = np.array([m[var_name] for m in member_arrays])
                var_mean = np.mean(var_ensemble, axis=0)
                var_pert = var_ensemble - var_mean

                cov = np.mean(var_pert * hx_pert.reshape(-1, *[1] * (var_pert.ndim - 1)), axis=0)

                if cov.shape == loc_weights.shape:
                    var_hx_cov = cov * loc_weights
                else:
                    new_shape = [1] * cov.ndim
                    for i in range(min(cov.ndim, loc_weights.ndim)):
                        if cov.shape[i] == loc_weights.shape[i]:
                            new_shape[i] = loc_weights.shape[i]
                    loc_weights_broadcast = loc_weights.reshape(new_shape)
                    var_hx_cov = cov * loc_weights_broadcast

                hx_var = np.var(hx_pert) + obs_error ** 2
                kalman_gain = var_hx_cov / hx_var if hx_var > 0 else np.zeros_like(var_hx_cov)

                for i, m in enumerate(member_arrays):
                    perturbed_obs = obs.value + np.random.randn() * obs_error
                    innovation_i = perturbed_obs - hx_array[i] if np.isfinite(hx_array[i]) else innovation
                    m[var_name] = m[var_name] + kalman_gain * innovation_i

            n_obs_assimilated += 1

        inflated_arrays = {}
        for var_name in member_arrays[0].keys():
            var_stack = np.array([m[var_name] for m in member_arrays])
            inflated = self.inflation.apply([var_stack[i] for i in range(var_stack.shape[0])])
            for i in range(len(member_arrays)):
                member_arrays[i][var_name] = inflated[i]

        analysis_ds = background.copy()
        for var_name in analysis_ds.data_vars:
            var_stack = np.array([m[var_name] for m in member_arrays])
            analysis_ds[var_name].values = np.mean(var_stack, axis=0)

        analysis_members = []
        for m in member_arrays:
            member_ds = background.copy()
            for var_name in member_ds.data_vars:
                if var_name in m:
                    member_ds[var_name].values = m[var_name]
            analysis_members.append(member_ds)

        analysis_increment = xr.Dataset()
        for var in analysis_ds.data_vars:
            analysis_increment[var] = analysis_ds[var] - background[var]

        diagnostics = {
            "ensemble_size": n_members,
            "localization_radius_km": self.localization.radius_km,
            "inflation_factor": self.inflation.factor,
        }

        return AssimilationResult(
            analysis=analysis_ds,
            background=background,
            analysis_increment=analysis_increment,
            ensemble_members=analysis_members,
            method="enkf",
            computation_time=time.time() - start_time,
            diagnostics=diagnostics,
            observations_assimilated=n_obs_assimilated,
            observations_rejected=n_obs_rejected,
        )

    def _generate_initial_ensemble(self, background: xr.Dataset) -> List[xr.Dataset]:
        ensemble = []
        for i in range(self.ensemble_size):
            member = background.copy()
            for var_name in member.data_vars:
                var_data = member[var_name].values
                noise_std = np.nanstd(var_data) * 0.01 if np.nanstd(var_data) > 0 else 0.001
                noise = np.random.randn(*var_data.shape) * noise_std
                member[var_name].values = var_data + noise
            ensemble.append(member)
        return ensemble

    @staticmethod
    def _extract_point(
        data: np.ndarray, dims: Tuple, model_lat: np.ndarray, model_lon: np.ndarray,
        obs_lat: float, obs_lon: float, lat_name: Optional[str], lon_name: Optional[str]
    ) -> float:
        if lat_name is None or lon_name is None:
            if data.size > 0:
                return float(np.nanmean(data.ravel()))
            return np.nan

        try:
            lat_axis = dims.index(lat_name)
            lon_axis = dims.index(lon_name)

            lat_idx = int(np.argmin(np.abs(model_lat - obs_lat)))
            lon_idx = int(np.argmin(np.abs(model_lon - obs_lon)))

            slicer = [slice(None)] * data.ndim
            slicer[lat_axis] = min(max(lat_idx, 0), data.shape[lat_axis] - 1)
            slicer[lon_axis] = min(max(lon_idx, 0), data.shape[lon_axis] - 1)

            value = data[tuple(slicer)]
            if isinstance(value, np.ndarray) and value.size >= 1:
                value = float(np.nanmean(value.ravel()))
            return float(value)
        except Exception:
            return np.nan


class ThreeDVar(DataAssimilationBase):
    def __init__(self, background_error_std: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.background_error_std = background_error_std

    def assimilate(
        self,
        background: xr.Dataset,
        observations: List[Observation],
        ensemble_members: Optional[List[xr.Dataset]] = None,
        time_window: Optional[Tuple[datetime, datetime]] = None,
    ) -> AssimilationResult:
        import time
        start_time = time.time()

        analysis = background.copy()
        lat_name = None
        lon_name = None
        for ln in ["lat", "latitude"]:
            if ln in background.coords:
                lat_name = ln
                break
        for ln in ["lon", "longitude"]:
            if ln in background.coords:
                lon_name = ln
                break

        model_lat = background[lat_name].values if lat_name else np.array([0.0])
        model_lon = background[lon_name].values if lon_name else np.array([0.0])

        n_obs_assimilated = 0
        n_obs_rejected = 0

        for var_name in analysis.data_vars:
            increment = np.zeros_like(analysis[var_name].values, dtype=float)
            weight_sum = np.zeros_like(increment)

            var_dims = analysis[var_name].dims

            for obs in observations:
                if obs.variable != var_name:
                    continue
                if not np.isfinite(obs.value) or obs.error <= 0:
                    n_obs_rejected += 1
                    continue

                bg_val = EnKF._extract_point(
                    background[var_name].values, var_dims,
                    model_lat, model_lon, obs.latitude, obs.longitude,
                    lat_name, lon_name
                )

                if not np.isfinite(bg_val):
                    n_obs_rejected += 1
                    continue

                innovation = obs.value - bg_val
                weights = self.localization.compute_weights(
                    model_lat, model_lon, obs.latitude, obs.longitude
                )

                bg_error = self.background_error_std
                obs_error = obs.error
                gain = bg_error ** 2 / (bg_error ** 2 + obs_error ** 2)

                if increment.ndim >= weights.ndim:
                    broadcastable_weights = weights.reshape(
                        weights.shape + (1,) * (increment.ndim - weights.ndim)
                    )
                else:
                    broadcastable_weights = weights.reshape(weights.shape[:increment.ndim])

                increment += gain * innovation * broadcastable_weights
                weight_sum += broadcastable_weights
                n_obs_assimilated += 1

            valid = weight_sum > 0
            if np.any(valid):
                analysis[var_name].values = (
                    background[var_name].values +
                    np.where(valid, increment / np.where(weight_sum > 0, weight_sum, 1.0), 0.0)
                )

        analysis_increment = xr.Dataset()
        for var in analysis.data_vars:
            analysis_increment[var] = analysis[var] - background[var]

        return AssimilationResult(
            analysis=analysis,
            background=background,
            analysis_increment=analysis_increment,
            method="3dvar",
            computation_time=time.time() - start_time,
            diagnostics={
                "background_error_std": self.background_error_std,
            },
            observations_assimilated=n_obs_assimilated,
            observations_rejected=n_obs_rejected,
        )


class FourDVar(ThreeDVar):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def assimilate(
        self,
        background: xr.Dataset,
        observations: List[Observation],
        ensemble_members: Optional[List[xr.Dataset]] = None,
        time_window: Optional[Tuple[datetime, datetime]] = None,
    ) -> AssimilationResult:
        if time_window is not None:
            start_time, end_time = time_window
            filtered_obs = [
                o for o in observations
                if o.timestamp is None or (start_time <= o.timestamp <= end_time)
            ]
        else:
            filtered_obs = observations

        result = super().assimilate(background, filtered_obs, ensemble_members, time_window)
        result.method = "4dvar"
        return result


class DataAssimilationEngine:
    def __init__(
        self,
        method: AssimilationMethod = AssimilationMethod.ENKF,
        ensemble_size: int = 100,
        localization_radius_km: float = 500.0,
        inflation_factor: float = 1.05,
        time_window_hours: int = 6,
    ):
        self.method = method
        self.ensemble_size = ensemble_size
        self.localization_radius_km = localization_radius_km
        self.inflation_factor = inflation_factor
        self.time_window = timedelta(hours=time_window_hours)

        self._assimilator = self._create_assimilator()

    def _create_assimilator(self) -> DataAssimilationBase:
        localization = Localization(radius_km=self.localization_radius_km)
        inflation = CovarianceInflation(factor=self.inflation_factor)
        obs_operator = ObservationOperator()

        if self.method in [AssimilationMethod.ENKF, AssimilationMethod.LETKF, AssimilationMethod.ENSEMBLE_TRANSFORM_KF]:
            return EnKF(
                ensemble_size=self.ensemble_size,
                localization=localization,
                inflation=inflation,
                obs_operator=obs_operator,
            )
        elif self.method == AssimilationMethod.THREEDVAR:
            return ThreeDVar(
                localization=localization,
                obs_operator=obs_operator,
            )
        elif self.method == AssimilationMethod.FOURDVAR or self.method == AssimilationMethod.HYBRID:
            return FourDVar(
                localization=localization,
                obs_operator=obs_operator,
            )
        else:
            return EnKF(
                ensemble_size=self.ensemble_size,
                localization=localization,
                inflation=inflation,
                obs_operator=obs_operator,
            )

    def assimilate(
        self,
        background: xr.Dataset,
        observations: List[Observation],
        ensemble_members: Optional[List[xr.Dataset]] = None,
        center_time: Optional[datetime] = None,
    ) -> AssimilationResult:
        center = center_time or datetime.now()
        time_window = (center - self.time_window / 2, center + self.time_window / 2)

        return self._assimilator.assimilate(
            background=background,
            observations=observations,
            ensemble_members=ensemble_members,
            time_window=time_window,
        )

    def observations_from_dataset(
        self,
        obs_ds: xr.Dataset,
        error_default: float = 1.0,
    ) -> List[Observation]:
        observations = []

        lat_name = None
        lon_name = None
        for ln in ["lat", "latitude"]:
            if ln in obs_ds.coords:
                lat_name = ln
                break
        for ln in ["lon", "longitude"]:
            if ln in obs_ds.coords:
                lon_name = ln
                break

        for var_name in obs_ds.data_vars:
            var_data = obs_ds[var_name]
            dims = var_data.dims

            lat_axis = dims.index(lat_name) if lat_name and lat_name in dims else None
            lon_axis = dims.index(lon_name) if lon_name and lon_name in dims else None
            time_axis = dims.index("time") if "time" in dims else None

            if lat_axis is not None and lon_axis is not None:
                lats = var_data[lat_name].values
                lons = var_data[lon_name].values

                other_axes = [i for i in range(var_data.ndim) if i not in [lat_axis, lon_axis]]
                if other_axes:
                    iter_shape = [var_data.shape[i] for i in other_axes]
                    for idx in np.ndindex(*iter_shape):
                        slicer = [slice(None)] * var_data.ndim
                        for i, axis in enumerate(other_axes):
                            slicer[axis] = idx[i]

                        obs_time = None
                        if time_axis is not None:
                            time_idx = other_axes.index(time_axis)
                            times = var_data["time"].values
                            obs_time = pd.to_datetime(times[idx[time_idx]]).to_pydatetime()

                        for i in range(len(lats)):
                            for j in range(len(lons)):
                                slicer_ij = slicer.copy()
                                slicer_ij[lat_axis] = i
                                slicer_ij[lon_axis] = j
                                val = var_data.values[tuple(slicer_ij)]
                                if np.isfinite(val):
                                    observations.append(Observation(
                                        value=float(val),
                                        error=error_default,
                                        latitude=float(lats[i]),
                                        longitude=float(lons[j]),
                                        timestamp=obs_time,
                                        variable=var_name,
                                    ))

        return observations

    def set_method(self, method: AssimilationMethod):
        self.method = method
        self._assimilator = self._create_assimilator()
