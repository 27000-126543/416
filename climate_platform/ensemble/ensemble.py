"""
Ensemble forecasting with probabilistic predictions.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy import stats

logger = logging.getLogger(__name__)


class PerturbationMethod(Enum):
    BRED = "bred"
    LAGGED_AVERAGE = "lagged_average"
    STOCHASTIC = "stochastic"
    SINGULAR_VECTORS = "singular_vectors"
    RANDOM = "random"
    MULTI_MODEL = "multi_model"
    INITIAL_CONDITION = "initial_condition"
    PHYSICS_PERTURBATION = "physics_perturbation"


class ForecastHorizon(Enum):
    SHORT_RANGE = "short_range"
    MEDIUM_RANGE = "medium_range"
    SEASONAL = "seasonal"
    LONG_TERM = "long_term"


@dataclass
class EnsembleMember:
    member_id: int
    data: xr.Dataset
    perturbation_method: PerturbationMethod
    perturbation_magnitude: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    valid_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def lead_time(self) -> timedelta:
        return self.valid_time - self.start_time


@dataclass
class ProbabilisticForecast:
    ensemble_mean: xr.Dataset
    ensemble_std: xr.Dataset
    percentiles: Dict[int, xr.Dataset]
    probabilities: Dict[str, xr.Dataset]
    members: List[EnsembleMember]
    horizon: ForecastHorizon
    calibration_method: str = "none"
    confidence_intervals: Dict[float, xr.Dataset] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_probability_above(self, variable: str, threshold: float) -> Optional[xr.DataArray]:
        key = f"{variable}_above_{threshold}"
        if key in self.probabilities:
            return self.probabilities[key]
        if variable in self.ensemble_mean.data_vars:
            member_data = np.array([m.data[variable].values for m in self.members])
            prob = np.mean(member_data > threshold, axis=0)
            return xr.DataArray(prob, dims=self.ensemble_mean[variable].dims,
                                coords=self.ensemble_mean[variable].coords)
        return None

    def get_probability_below(self, variable: str, threshold: float) -> Optional[xr.DataArray]:
        if variable in self.ensemble_mean.data_vars:
            member_data = np.array([m.data[variable].values for m in self.members])
            prob = np.mean(member_data < threshold, axis=0)
            return xr.DataArray(prob, dims=self.ensemble_mean[variable].dims,
                                coords=self.ensemble_mean[variable].coords)
        return None

    def get_probability_between(self, variable: str, low: float, high: float) -> Optional[xr.DataArray]:
        if variable in self.ensemble_mean.data_vars:
            member_data = np.array([m.data[variable].values for m in self.members])
            prob = np.mean((member_data >= low) & (member_data <= high), axis=0)
            return xr.DataArray(prob, dims=self.ensemble_mean[variable].dims,
                                coords=self.ensemble_mean[variable].coords)
        return None


class PerturbationStrategy(ABC):
    @abstractmethod
    def perturb(self, state: xr.Dataset, member_id: int, total_members: int) -> xr.Dataset:
        pass


class RandomPerturbation(PerturbationStrategy):
    def __init__(self, amplitude: float = 0.01, relative: bool = True):
        self.amplitude = amplitude
        self.relative = relative

    def perturb(self, state: xr.Dataset, member_id: int, total_members: int) -> xr.Dataset:
        perturbed = state.copy()
        rng = np.random.RandomState(member_id + 42)
        for var in perturbed.data_vars:
            data = perturbed[var].values
            if self.relative and np.nanstd(data) > 0:
                noise = rng.randn(*data.shape) * self.amplitude * np.nanstd(data)
            else:
                noise = rng.randn(*data.shape) * self.amplitude
            perturbed[var].values = data + noise
        return perturbed


class BREDPerturbation(PerturbationStrategy):
    def __init__(
        self,
        breeding_cycles: int = 5,
        rescaling_interval: int = 6,
        amplitude: float = 0.05,
        growth_rate: float = 1.2,
    ):
        self.breeding_cycles = breeding_cycles
        self.rescaling_interval = rescaling_interval
        self.amplitude = amplitude
        self.growth_rate = growth_rate
        self._vectors: Dict[int, xr.Dataset] = {}

    def perturb(self, state: xr.Dataset, member_id: int, total_members: int) -> xr.Dataset:
        perturbed = state.copy()
        rng = np.random.RandomState(member_id + 100)

        if member_id not in self._vectors:
            vec = state.copy()
            for var in vec.data_vars:
                data = vec[var].values
                if np.nanstd(data) > 0:
                    vec[var].values = rng.randn(*data.shape) * self.amplitude * np.nanstd(data)
                else:
                    vec[var].values = rng.randn(*data.shape) * self.amplitude
            self._vectors[member_id] = vec

        for _ in range(self.breeding_cycles):
            vector = self._vectors[member_id]
            vec_norm = self._dataset_norm(vector)
            if vec_norm > 0:
                for var in vector.data_vars:
                    vector[var].values = (vector[var].values / vec_norm) * self.amplitude * self.growth_rate

        for var in perturbed.data_vars:
            if var in self._vectors[member_id].data_vars:
                perturbed[var].values = state[var].values + self._vectors[member_id][var].values

        return perturbed

    @staticmethod
    def _dataset_norm(ds: xr.Dataset) -> float:
        total = 0.0
        count = 0
        for var in ds.data_vars:
            data = ds[var].values
            finite_vals = data[np.isfinite(data)]
            if len(finite_vals) > 0:
                total += np.sum(finite_vals ** 2)
                count += len(finite_vals)
        return np.sqrt(total / count) if count > 0 else 1.0


class LaggedAverageEnsemble(PerturbationStrategy):
    def __init__(self, lag_hours: List[int] = None):
        self.lag_hours = lag_hours or [0, 6, 12, 18, 24]
        self._history: List[xr.Dataset] = []

    def add_to_history(self, state: xr.Dataset):
        self._history.append(state.copy())
        max_lag = max(self.lag_hours)
        while len(self._history) > max_lag + 1:
            self._history.pop(0)

    def perturb(self, state: xr.Dataset, member_id: int, total_members: int) -> xr.Dataset:
        lag_idx = member_id % len(self.lag_hours)
        lag = self.lag_hours[lag_idx]

        if lag == 0 or len(self._history) <= lag:
            return state.copy()

        base = self._history[-(lag + 1)].copy()
        for var in state.data_vars:
            if var in base.data_vars:
                base_data = base[var].values
                current_data = state[var].values
                blend = 0.5
                base[var].values = blend * base_data + (1 - blend) * current_data
        return base


class StochasticPerturbation(PerturbationStrategy):
    def __init__(
        self,
        spatial_correlation_km: float = 500.0,
        temporal_correlation_hours: float = 6.0,
        amplitude: float = 0.1,
    ):
        self.spatial_correlation_km = spatial_correlation_km
        self.temporal_correlation_hours = temporal_correlation_hours
        self.amplitude = amplitude
        self._noise_fields: Dict[int, np.ndarray] = {}

    def perturb(self, state: xr.Dataset, member_id: int, total_members: int) -> xr.Dataset:
        perturbed = state.copy()
        rng = np.random.RandomState(member_id + 200)

        lat_name = None
        lon_name = None
        for ln in ["lat", "latitude"]:
            if ln in state.coords:
                lat_name = ln
                break
        for ln in ["lon", "longitude"]:
            if ln in state.coords:
                lon_name = ln
                break

        if lat_name and lon_name:
            lats = state[lat_name].values
            lons = state[lon_name].values
            spatial_scale = int(self.spatial_correlation_km / 111 / max(abs(lats[1] - lats[0]) if len(lats) > 1 else 1,
                                                                         abs(lons[1] - lons[0]) if len(lons) > 1 else 1))
            spatial_scale = max(2, min(spatial_scale, min(len(lats), len(lons)) // 4))

        for var in perturbed.data_vars:
            data = perturbed[var].values
            noise = rng.randn(*data.shape)

            if lat_name and lon_name and data.ndim >= 2:
                try:
                    from scipy.ndimage import gaussian_filter
                    dims = state[var].dims
                    lat_axis = dims.index(lat_name) if lat_name in dims else None
                    lon_axis = dims.index(lon_name) if lon_name in dims else None
                    if lat_axis is not None and lon_axis is not None:
                        sigma = [0] * data.ndim
                        sigma[lat_axis] = spatial_scale
                        sigma[lon_axis] = spatial_scale
                        noise = gaussian_filter(noise, sigma=sigma)
                except Exception:
                    pass

            std = np.nanstd(data) if np.nanstd(data) > 0 else 1.0
            perturbed[var].values = data + noise * self.amplitude * std

        return perturbed


class ProbabilityCalibration:
    def __init__(self, method: str = "bayesian"):
        self.method = method
        self._calibration_params: Dict[str, Any] = {}

    def calibrate(self, forecast: ProbabilisticForecast, observations: Optional[xr.Dataset] = None) -> ProbabilisticForecast:
        if observations is None:
            return forecast

        calibrated = forecast
        if self.method == "bayesian":
            calibrated = self._bayesian_calibration(forecast, observations)
        elif self.method == "isotonic":
            calibrated = self._isotonic_calibration(forecast, observations)
        elif self.method == "reliability":
            calibrated = self._reliability_calibration(forecast, observations)

        calibrated.calibration_method = self.method
        return calibrated

    def _bayesian_calibration(self, forecast: ProbabilisticForecast, observations: xr.Dataset) -> ProbabilisticForecast:
        calibrated_mean = forecast.ensemble_mean.copy()
        calibrated_std = forecast.ensemble_std.copy()

        for var in forecast.ensemble_mean.data_vars:
            if var in observations.data_vars:
                prior_mean = forecast.ensemble_mean[var].values
                prior_std = forecast.ensemble_std[var].values
                obs_data = observations[var].values

                obs_var = np.nanvar(obs_data) if np.nanvar(obs_data) > 0 else 1.0
                prior_var = prior_std ** 2

                weight = prior_var / (prior_var + obs_var)
                calibrated_mean[var].values = weight * prior_mean + (1 - weight) * np.nanmean(obs_data)
                calibrated_std[var].values = np.sqrt(prior_var * (1 - weight))

        return ProbabilisticForecast(
            ensemble_mean=calibrated_mean,
            ensemble_std=calibrated_std,
            percentiles=forecast.percentiles,
            probabilities=forecast.probabilities,
            members=forecast.members,
            horizon=forecast.horizon,
            calibration_method="bayesian",
            confidence_intervals=forecast.confidence_intervals,
            timestamp=datetime.now(),
        )

    def _isotonic_calibration(self, forecast: ProbabilisticForecast, observations: xr.Dataset) -> ProbabilisticForecast:
        return forecast

    def _reliability_calibration(self, forecast: ProbabilisticForecast, observations: xr.Dataset) -> ProbabilisticForecast:
        return forecast


class EnsembleVerification:
    def __init__(self):
        self._metrics: Dict[str, Any] = {}

    def compute_crps(self, forecast: ProbabilisticForecast, observations: xr.Dataset) -> xr.Dataset:
        result = xr.Dataset()
        for var in forecast.ensemble_mean.data_vars:
            if var in observations.data_vars:
                member_data = np.array([m.data[var].values for m in forecast.members])
                obs_data = observations[var].values

                n_members = len(forecast.members)
                crps_ensemble = np.zeros_like(obs_data)

                for i in range(n_members):
                    for j in range(n_members):
                        crps_ensemble += np.abs(member_data[i] - member_data[j])
                crps_ensemble /= (2 * n_members ** 2)

                for i in range(n_members):
                    crps_ensemble -= np.abs(member_data[i] - obs_data) / n_members

                result[var] = xr.DataArray(crps_ensemble, dims=observations[var].dims,
                                           coords=observations[var].coords)
        return result

    def compute_brier_score(self, forecast: ProbabilisticForecast, observations: xr.Dataset,
                            thresholds: Dict[str, List[float]]) -> xr.Dataset:
        result = xr.Dataset()
        for var, thresh_list in thresholds.items():
            for thresh in thresh_list:
                if var in observations.data_vars:
                    prob = forecast.get_probability_above(var, thresh)
                    if prob is not None:
                        obs_binary = (observations[var].values > thresh).astype(float)
                        brier = (prob.values - obs_binary) ** 2
                        result[f"{var}_above_{thresh}"] = xr.DataArray(
                            brier, dims=prob.dims, coords=prob.coords
                        )
        return result

    def compute_spread_skill_ratio(self, forecast: ProbabilisticForecast, observations: xr.Dataset) -> Dict[str, float]:
        ratios = {}
        for var in forecast.ensemble_mean.data_vars:
            if var in observations.data_vars:
                spread = np.sqrt(np.nanmean(forecast.ensemble_std[var].values ** 2))
                error = np.sqrt(np.nanmean((forecast.ensemble_mean[var].values - observations[var].values) ** 2))
                ratios[var] = spread / error if error > 0 else float('inf')
        return ratios

    def compute_roc_curve(self, forecast: ProbabilisticForecast, observations: xr.Dataset,
                          variable: str, threshold: float, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prob = forecast.get_probability_above(variable, threshold)
        if prob is None or variable not in observations.data_vars:
            return np.array([]), np.array([]), np.array([])

        obs_binary = (observations[variable].values > threshold).astype(int).ravel()
        prob_flat = prob.values.ravel()
        valid = np.isfinite(prob_flat) & np.isfinite(obs_binary)
        obs_binary = obs_binary[valid]
        prob_flat = prob_flat[valid]

        tpr_list = []
        fpr_list = []
        thresholds_list = np.linspace(0, 1, n_points)

        for t in thresholds_list:
            predictions = (prob_flat >= t).astype(int)
            tp = np.sum((predictions == 1) & (obs_binary == 1))
            fp = np.sum((predictions == 1) & (obs_binary == 0))
            fn = np.sum((predictions == 0) & (obs_binary == 1))
            tn = np.sum((predictions == 0) & (obs_binary == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return np.array(tpr_list), np.array(fpr_list), thresholds_list


class EnsembleForecast:
    def __init__(
        self,
        ensemble_size: int = 50,
        perturbation_method: PerturbationMethod = PerturbationMethod.BRED,
        forecast_horizon: ForecastHorizon = ForecastHorizon.SHORT_RANGE,
    ):
        self.ensemble_size = ensemble_size
        self.perturbation_method = perturbation_method
        self.forecast_horizon = forecast_horizon

        self._strategies: Dict[PerturbationMethod, PerturbationStrategy] = {
            PerturbationMethod.RANDOM: RandomPerturbation(),
            PerturbationMethod.BRED: BREDPerturbation(),
            PerturbationMethod.STOCHASTIC: StochasticPerturbation(),
            PerturbationMethod.LAGGED_AVERAGE: LaggedAverageEnsemble(),
            PerturbationMethod.INITIAL_CONDITION: RandomPerturbation(amplitude=0.02),
            PerturbationMethod.PHYSICS_PERTURBATION: StochasticPerturbation(amplitude=0.15),
        }
        self.calibrator = ProbabilityCalibration()
        self.verifier = EnsembleVerification()
        self._members: List[EnsembleMember] = []

    def generate_initial_ensemble(self, control_state: xr.Dataset, start_time: Optional[datetime] = None) -> List[EnsembleMember]:
        st = start_time or datetime.now()
        strategy = self._strategies.get(self.perturbation_method, RandomPerturbation())
        self._members = []

        for i in range(self.ensemble_size):
            perturbed_ds = strategy.perturb(control_state, i, self.ensemble_size)
            member = EnsembleMember(
                member_id=i,
                data=perturbed_ds,
                perturbation_method=self.perturbation_method,
                perturbation_magnitude=getattr(strategy, 'amplitude', 0.0),
                start_time=st,
                valid_time=st,
                metadata={"ensemble_size": self.ensemble_size, "member_index": i},
            )
            self._members.append(member)

        logger.info(f"Generated ensemble of {self.ensemble_size} members using {self.perturbation_method.value}")
        return self._members

    def integrate_member(
        self,
        member: EnsembleMember,
        model_step_fn: Callable[[xr.Dataset, timedelta], xr.Dataset],
        lead_time: timedelta,
        dt: timedelta,
    ) -> EnsembleMember:
        current = member.data.copy()
        elapsed = timedelta(0)
        target = lead_time

        while elapsed < target:
            step_dt = min(dt, target - elapsed)
            current = model_step_fn(current, step_dt)
            elapsed += step_dt

        return EnsembleMember(
            member_id=member.member_id,
            data=current,
            perturbation_method=member.perturbation_method,
            perturbation_magnitude=member.perturbation_magnitude,
            start_time=member.start_time,
            valid_time=member.start_time + lead_time,
            metadata={**member.metadata, "lead_time_hours": lead_time.total_seconds() / 3600},
        )

    def integrate_ensemble(
        self,
        model_step_fn: Callable[[xr.Dataset, timedelta], xr.Dataset],
        lead_time: timedelta,
        dt: timedelta,
        parallel: bool = True,
    ) -> List[EnsembleMember]:
        results = []
        for member in self._members:
            result = self.integrate_member(member, model_step_fn, lead_time, dt)
            results.append(result)
        self._members = results
        return results

    def compute_probabilistic_forecast(
        self,
        members: Optional[List[EnsembleMember]] = None,
        percentiles: Optional[List[int]] = None,
        confidence_levels: Optional[List[float]] = None,
        calibration_obs: Optional[xr.Dataset] = None,
    ) -> ProbabilisticForecast:
        mems = members or self._members
        if not mems:
            raise ValueError("No ensemble members available")

        pcts = percentiles or [10, 25, 50, 75, 90]
        conf_levels = confidence_levels or [0.68, 0.90, 0.95]

        first_ds = mems[0].data
        ensemble_mean = first_ds.copy()
        ensemble_std = first_ds.copy()
        percentile_datasets: Dict[int, xr.Dataset] = {}
        confidence_intervals: Dict[float, xr.Dataset] = {}

        for var in first_ds.data_vars:
            all_data = np.array([m.data[var].values for m in mems])
            ensemble_mean[var].values = np.mean(all_data, axis=0)
            ensemble_std[var].values = np.std(all_data, axis=0)

            for p in pcts:
                if p not in percentile_datasets:
                    percentile_datasets[p] = first_ds.copy()
                percentile_datasets[p][var].values = np.percentile(all_data, p, axis=0)

            for cl in conf_levels:
                if cl not in confidence_intervals:
                    confidence_intervals[cl] = first_ds.copy()
                low_pct = (1 - cl) / 2 * 100
                high_pct = (1 + cl) / 2 * 100
                ci_low = np.percentile(all_data, low_pct, axis=0)
                ci_high = np.percentile(all_data, high_pct, axis=0)
                ci_values = np.stack([ci_low, ci_high], axis=0)
                if "ci" not in confidence_intervals[cl][var].dims:
                    try:
                        expanded = confidence_intervals[cl][var].expand_dims("ci", axis=0)
                        confidence_intervals[cl][var] = xr.DataArray(
                            ci_values,
                            dims=["ci"] + list(first_ds[var].dims),
                            coords={"ci": ["lower", "upper"], **{d: first_ds[var].coords[d] for d in first_ds[var].dims}}
                        )
                    except Exception:
                        pass

        probabilities: Dict[str, xr.Dataset] = {}

        forecast = ProbabilisticForecast(
            ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std,
            percentiles=percentile_datasets,
            probabilities=probabilities,
            members=mems,
            horizon=self.forecast_horizon,
            confidence_intervals=confidence_intervals,
            timestamp=datetime.now(),
        )

        if calibration_obs is not None:
            forecast = self.calibrator.calibrate(forecast, calibration_obs)

        return forecast

    def add_perturbation_strategy(self, method: PerturbationMethod, strategy: PerturbationStrategy):
        self._strategies[method] = strategy

    @property
    def members(self) -> List[EnsembleMember]:
        return self._members.copy()

    @property
    def effective_ensemble_size(self) -> float:
        if len(self._members) < 2:
            return float(len(self._members))

        n = len(self._members)
        correlations = []
        sample_var = None

        for var in self._members[0].data.data_vars:
            all_data = np.array([m.data[var].values.ravel() for m in self._members])
            valid = np.all(np.isfinite(all_data), axis=1)
            if np.sum(valid) >= 2:
                cov_matrix = np.cov(all_data[valid])
                trace = np.trace(cov_matrix)
                total_var = np.sum(cov_matrix)
                if total_var > 0:
                    avg_corr = (total_var - trace) / (n * (n - 1)) / (trace / n) if trace > 0 else 0
                    correlations.append(max(0, min(1, avg_corr)))

        if correlations:
            avg_corr = np.mean(correlations)
            return n / (1 + (n - 1) * avg_corr)
        return float(n)
