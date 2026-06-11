"""
Climate fingerprint analysis: time-frequency analysis, causal inference, pattern matching.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import signal, stats
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


@dataclass
class ClimateEvent:
    event_id: str
    start_time: datetime
    end_time: datetime
    description: str
    region: Optional[str] = None
    variables: Dict[str, float] = field(default_factory=dict)
    fingerprint: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time


@dataclass
class SimilarityResult:
    query_event: Optional[ClimateEvent]
    matched_event: ClimateEvent
    similarity_score: float
    temporal_similarity: float
    spatial_similarity: float
    matched_variables: Dict[str, float] = field(default_factory=dict)
    lag: Optional[timedelta] = None
    causal_strength: Optional[float] = None


@dataclass
class CausalRelation:
    cause_variable: str
    effect_variable: str
    strength: float
    p_value: float
    lag: int
    significant: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TFSpectrum:
    frequencies: np.ndarray
    times: np.ndarray
    power: np.ndarray
    method: str
    variable: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeFrequencyAnalyzer(ABC):
    @abstractmethod
    def analyze(self, data: np.ndarray, fs: float, **kwargs) -> TFSpectrum:
        pass


class WaveletAnalyzer(TimeFrequencyAnalyzer):
    def __init__(
        self,
        wavelet: str = "morlet",
        scales_per_octave: int = 12,
        min_period: float = 2.0,
        max_period: Optional[float] = None,
    ):
        self.wavelet = wavelet
        self.scales_per_octave = scales_per_octave
        self.min_period = min_period
        self.max_period = max_period

    def analyze(self, data: np.ndarray, fs: float, **kwargs) -> TFSpectrum:
        n = len(data)
        dt = 1.0 / fs

        if self.max_period is None:
            self.max_period = n * dt / 8

        min_scale = self.min_period / dt
        max_scale = self.max_period / dt
        n_scales = int(np.log2(max_scale / min_scale) * self.scales_per_octave)
        scales = min_scale * 2 ** (np.arange(n_scales) / self.scales_per_octave)

        data_detrend = data - np.nanmean(data)
        data_detrend = np.nan_to_num(data_detrend, nan=0.0)

        try:
            from scipy.signal import cwt, ricker, morlet2
            if self.wavelet == "morlet":
                widths = scales
                cwt_matrix = np.zeros((len(widths), n), dtype=complex)
                omega0 = 5
                for i, width in enumerate(widths):
                    w = width
                    s = w
                    t = np.arange(-s * 5, s * 5 + 1)
                    wavelet_data = np.exp(1j * omega0 * t / s) * np.exp(-t ** 2 / (2 * s ** 2))
                    cwt_matrix[i] = np.convolve(data_detrend, wavelet_data, mode='same')
                power = np.abs(cwt_matrix) ** 2
            else:
                widths = scales
                cwt_matrix = cwt(data_detrend, ricker, widths)
                power = np.abs(cwt_matrix) ** 2

            freqs = 1.0 / (scales * dt)
            times = np.arange(n) * dt

            return TFSpectrum(
                frequencies=freqs,
                times=times,
                power=power,
                method=f"wavelet_{self.wavelet}",
                variable=kwargs.get("variable", ""),
            )
        except Exception as e:
            logger.error(f"Wavelet analysis failed: {e}")
            return TFSpectrum(
                frequencies=np.array([]),
                times=np.array([]),
                power=np.array([]),
                method=f"wavelet_{self.wavelet}",
                variable=kwargs.get("variable", ""),
                metadata={"error": str(e)},
            )


class FFTAnalyzer(TimeFrequencyAnalyzer):
    def __init__(self, window: str = "hann", nperseg: Optional[int] = None, noverlap: Optional[int] = None):
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap

    def analyze(self, data: np.ndarray, fs: float, **kwargs) -> TFSpectrum:
        data_detrend = np.nan_to_num(data - np.nanmean(data), nan=0.0)
        nperseg = self.nperseg or min(256, len(data_detrend))
        noverlap = self.noverlap or nperseg // 2

        try:
            f, t, Zxx = signal.stft(data_detrend, fs=fs, window=self.window,
                                    nperseg=nperseg, noverlap=noverlap)
            power = np.abs(Zxx) ** 2

            return TFSpectrum(
                frequencies=f,
                times=t,
                power=power,
                method="fft_stft",
                variable=kwargs.get("variable", ""),
            )
        except Exception as e:
            logger.error(f"FFT analysis failed: {e}")
            return TFSpectrum(
                frequencies=np.array([]),
                times=np.array([]),
                power=np.array([]),
                method="fft_stft",
                variable=kwargs.get("variable", ""),
                metadata={"error": str(e)},
            )


class HilbertHuangTransform(TimeFrequencyAnalyzer):
    def __init__(self, num_imfs: int = 10, max_sifts: int = 100):
        self.num_imfs = num_imfs
        self.max_sifts = max_sifts

    def _emd(self, data: np.ndarray) -> List[np.ndarray]:
        imfs = []
        residual = data.copy()

        for _ in range(self.num_imfs):
            imf = self._extract_imf(residual)
            if imf is None:
                break
            imfs.append(imf)
            residual = residual - imf
            if np.std(residual) < 1e-10 * np.std(data):
                break

        imfs.append(residual)
        return imfs

    def _extract_imf(self, data: np.ndarray) -> Optional[np.ndarray]:
        x = data.copy()
        for _ in range(self.max_sifts):
            maxima = self._find_extrema(x, "max")
            minima = self._find_extrema(x, "min")

            if len(maxima) < 2 or len(minima) < 2:
                return None

            upper_env = self._interp_envelope(x, maxima)
            lower_env = self._interp_envelope(x, minima)
            mean_env = (upper_env + lower_env) / 2

            x_new = x - mean_env

            if np.all(np.abs(x_new - x) < 1e-10 * np.std(data)):
                return x_new
            x = x_new

        return x

    @staticmethod
    def _find_extrema(data: np.ndarray, extrema_type: str) -> np.ndarray:
        indices = []
        for i in range(1, len(data) - 1):
            if extrema_type == "max":
                if data[i] > data[i - 1] and data[i] > data[i + 1]:
                    indices.append(i)
            else:
                if data[i] < data[i - 1] and data[i] < data[i + 1]:
                    indices.append(i)
        return np.array(indices)

    @staticmethod
    def _interp_envelope(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if len(indices) < 2:
            return data.copy()
        x_full = np.arange(len(data))
        return np.interp(x_full, indices, data[indices])

    def analyze(self, data: np.ndarray, fs: float, **kwargs) -> TFSpectrum:
        data_detrend = np.nan_to_num(data - np.nanmean(data), nan=0.0)
        imfs = self._emd(data_detrend)

        n = len(data_detrend)
        dt = 1.0 / fs
        times = np.arange(n) * dt

        all_freqs = []
        all_powers = []

        for imf in imfs[:-1]:
            analytic = signal.hilbert(imf)
            amplitude = np.abs(analytic)
            phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
            inst_freq = np.concatenate([[inst_freq[0]], inst_freq])

            all_freqs.append(inst_freq)
            all_powers.append(amplitude ** 2)

        if all_freqs:
            n_freqs = 100
            freqs = np.linspace(0, fs / 2, n_freqs)
            power = np.zeros((n_freqs, n))

            for f_list, p_list in zip(all_freqs, all_powers):
                for i in range(n):
                    if 0 <= f_list[i] < fs / 2:
                        freq_idx = int(f_list[i] / (fs / 2) * (n_freqs - 1))
                        power[freq_idx, i] += p_list[i]

            return TFSpectrum(
                frequencies=freqs,
                times=times,
                power=power,
                method="hilbert_huang",
                variable=kwargs.get("variable", ""),
                metadata={"num_imfs": len(imfs)},
            )

        return TFSpectrum(
            frequencies=np.array([]),
            times=np.array([]),
            power=np.array([]),
            method="hilbert_huang",
            variable=kwargs.get("variable", ""),
        )


class CausalInference:
    def __init__(
        self,
        method: str = "pc_stable",
        significance_level: float = 0.05,
        max_lag: int = 10,
    ):
        self.method = method
        self.significance_level = significance_level
        self.max_lag = max_lag
        self._relations: List[CausalRelation] = []

    def granger_causality(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        max_lag: Optional[int] = None,
    ) -> CausalRelation:
        lags = max_lag or self.max_lag
        cause = np.nan_to_num(cause, nan=np.nanmean(cause))
        effect = np.nan_to_num(effect, nan=np.nanmean(effect))

        best_lag = 0
        best_f_stat = 0.0
        best_p_value = 1.0

        for lag in range(1, lags + 1):
            n = len(cause) - lag
            if n < lag * 2 + 5:
                continue

            y_restricted = effect[lag:]
            X_restricted = np.column_stack([effect[i:-lag + i] for i in range(lag)])

            y_full = effect[lag:]
            X_full = np.column_stack(
                [X_restricted] + [cause[i:-lag + i] for i in range(lag)]
            )

            try:
                beta_restricted, _, _, _ = np.linalg.lstsq(X_restricted, y_restricted, rcond=None)
                residuals_restricted = y_restricted - X_restricted @ beta_restricted
                rss_restricted = np.sum(residuals_restricted ** 2)

                beta_full, _, _, _ = np.linalg.lstsq(X_full, y_full, rcond=None)
                residuals_full = y_full - X_full @ beta_full
                rss_full = np.sum(residuals_full ** 2)

                df1 = lag
                df2 = n - X_full.shape[1]
                if df2 > 0 and rss_full > 0:
                    f_stat = ((rss_restricted - rss_full) / df1) / (rss_full / df2)
                    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

                    if p_value < best_p_value:
                        best_lag = lag
                        best_f_stat = f_stat
                        best_p_value = p_value
            except Exception:
                continue

        relation = CausalRelation(
            cause_variable="cause",
            effect_variable="effect",
            strength=best_f_stat if best_f_stat > 0 else 0.0,
            p_value=best_p_value,
            lag=best_lag,
            significant=best_p_value < self.significance_level,
            metadata={"method": "granger"},
        )
        self._relations.append(relation)
        return relation

    def pc_stable(
        self,
        variables: Dict[str, np.ndarray],
        names: Optional[List[str]] = None,
    ) -> List[CausalRelation]:
        var_names = names or list(variables.keys())
        var_list = [np.nan_to_num(v, nan=np.nanmean(v)) for v in variables.values()]
        n_vars = len(var_list)

        relations = []

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                cause = var_list[i]
                effect = var_list[j]

                result = self.granger_causality(cause, effect)
                result.cause_variable = var_names[i]
                result.effect_variable = var_names[j]

                if result.significant:
                    relations.append(result)

        self._relations.extend(relations)
        return relations

    def transfer_entropy(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        bins: int = 10,
        lag: int = 1,
    ) -> CausalRelation:
        cause = np.nan_to_num(cause, nan=np.nanmean(cause))
        effect = np.nan_to_num(effect, nan=np.nanmean(effect))
        n = min(len(cause), len(effect))

        cause_past = cause[:n - lag]
        effect_past = effect[:n - lag]
        effect_future = effect[lag:n]

        cause_bins = np.linspace(cause.min(), cause.max(), bins + 1)
        effect_bins = np.linspace(effect.min(), effect.max(), bins + 1)

        cause_digitized = np.digitize(cause_past, cause_bins) - 1
        effect_past_digitized = np.digitize(effect_past, effect_bins) - 1
        effect_future_digitized = np.digitize(effect_future, effect_bins) - 1

        def prob(x):
            unique, counts = np.unique(x, return_counts=True)
            return dict(zip(unique, counts / len(x)))

        p_yf = prob(effect_future_digitized)
        p_yp = prob(effect_past_digitized)
        p_yf_yp = prob(tuple(zip(effect_future_digitized, effect_past_digitized)))
        p_yf_yp_xp = prob(tuple(zip(effect_future_digitized, effect_past_digitized, cause_digitized)))
        p_yp_xp = prob(tuple(zip(effect_past_digitized, cause_digitized)))

        te = 0.0
        for yf in set(effect_future_digitized):
            for yp in set(effect_past_digitized):
                for xp in set(cause_digitized):
                    p_joint = p_yf_yp_xp.get((yf, yp, xp), 0)
                    p_cond1 = p_joint / p_yp_xp.get((yp, xp), 1e-10) if p_yp_xp.get((yp, xp), 0) > 0 else 0
                    p_cond2 = p_yf_yp.get((yf, yp), 0) / p_yp.get(yp, 1e-10) if p_yp.get(yp, 0) > 0 else 0
                    if p_joint > 0 and p_cond1 > 0 and p_cond2 > 0:
                        te += p_joint * np.log2(p_cond1 / p_cond2)

        relation = CausalRelation(
            cause_variable="cause",
            effect_variable="effect",
            strength=max(0, te),
            p_value=0.0 if te > 0.01 else 1.0,
            lag=lag,
            significant=te > 0.01,
            metadata={"method": "transfer_entropy", "bins": bins},
        )
        self._relations.append(relation)
        return relation

    @property
    def relations(self) -> List[CausalRelation]:
        return self._relations.copy()


class PatternMatcher:
    def __init__(
        self,
        similarity_metric: str = "cosine",
        top_k: int = 20,
        temporal_weight: float = 0.6,
        spatial_weight: float = 0.4,
    ):
        self.similarity_metric = similarity_metric
        self.top_k = top_k
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        self._database: List[ClimateEvent] = []

    def add_event(self, event: ClimateEvent):
        self._database.append(event)

    def add_events(self, events: List[ClimateEvent]):
        self._database.extend(events)

    def extract_fingerprint(self, dataset: xr.Dataset, variables: Optional[List[str]] = None) -> np.ndarray:
        vars_to_use = variables or list(dataset.data_vars)
        features = []

        for var in vars_to_use:
            if var in dataset.data_vars:
                data = dataset[var].values
                valid = np.isfinite(data)
                if np.any(valid):
                    flat = data[valid]
                    features.extend([
                        np.mean(flat),
                        np.std(flat),
                        np.min(flat),
                        np.max(flat),
                        np.percentile(flat, 25),
                        np.percentile(flat, 50),
                        np.percentile(flat, 75),
                        stats.skew(flat) if len(flat) > 2 else 0,
                        stats.kurtosis(flat) if len(flat) > 3 else 0,
                    ])

        return np.array(features)

    def _temporal_similarity(self, query: ClimateEvent, target: ClimateEvent) -> float:
        duration_sim = 1.0 - min(abs(query.duration.total_seconds() - target.duration.total_seconds()) /
                                 max(query.duration.total_seconds(), target.duration.total_seconds(), 1), 1.0)

        q_fp = query.fingerprint
        t_fp = target.fingerprint
        if len(q_fp) > 0 and len(t_fp) > 0 and len(q_fp) == len(t_fp):
            fp_sim = 1 - cosine(q_fp, t_fp) if np.any(q_fp) and np.any(t_fp) else 0.0
            fp_sim = max(0, min(1, fp_sim))
        else:
            fp_sim = 0.0

        return 0.5 * duration_sim + 0.5 * fp_sim

    def _spatial_similarity(self, query: ClimateEvent, target: ClimateEvent) -> float:
        common_vars = set(query.variables.keys()) & set(target.variables.keys())
        if not common_vars:
            return 0.0

        sims = []
        for var in common_vars:
            q_val = query.variables.get(var, 0)
            t_val = target.variables.get(var, 0)
            max_val = max(abs(q_val), abs(t_val), 1e-10)
            sims.append(1.0 - abs(q_val - t_val) / max_val)

        return np.mean(sims) if sims else 0.0

    def compute_similarity(self, query: ClimateEvent, target: ClimateEvent) -> SimilarityResult:
        temp_sim = self._temporal_similarity(query, target)
        spat_sim = self._spatial_similarity(query, target)
        overall = self.temporal_weight * temp_sim + self.spatial_weight * spat_sim

        matched_vars = {}
        for var in set(query.variables.keys()) & set(target.variables.keys()):
            matched_vars[var] = abs(query.variables[var] - target.variables[var])

        return SimilarityResult(
            query_event=query,
            matched_event=target,
            similarity_score=overall,
            temporal_similarity=temp_sim,
            spatial_similarity=spat_sim,
            matched_variables=matched_vars,
        )

    def find_similar(
        self,
        query: ClimateEvent,
        candidates: Optional[List[ClimateEvent]] = None,
    ) -> List[SimilarityResult]:
        search_space = candidates or self._database
        results = []

        for candidate in search_space:
            if candidate.event_id == query.event_id:
                continue
            results.append(self.compute_similarity(query, candidate))

        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:self.top_k]

    def find_by_pattern(
        self,
        dataset: xr.Dataset,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
    ) -> List[SimilarityResult]:
        fingerprint = self.extract_fingerprint(dataset, variables)
        var_stats = {}
        for var in (variables or list(dataset.data_vars)):
            if var in dataset.data_vars:
                data = dataset[var].values
                valid = np.isfinite(data)
                if np.any(valid):
                    var_stats[var] = float(np.mean(data[valid]))

        query = ClimateEvent(
            event_id="query",
            start_time=start_time or datetime.now(),
            end_time=end_time or datetime.now(),
            description="Query pattern",
            variables=var_stats,
            fingerprint=fingerprint,
        )
        return self.find_similar(query)


class ClimateFingerprintEngine:
    def __init__(
        self,
        tf_methods: Optional[List[str]] = None,
        causal_method: str = "pc_stable",
        similarity_metric: str = "cosine",
    ):
        self.tf_methods = tf_methods or ["wavelet", "fft", "hilbert_huang"]
        self._tf_analyzers: Dict[str, TimeFrequencyAnalyzer] = {
            "wavelet": WaveletAnalyzer(),
            "fft": FFTAnalyzer(),
            "hilbert_huang": HilbertHuangTransform(),
        }
        self.causal_inference = CausalInference(method=causal_method)
        self.pattern_matcher = PatternMatcher(similarity_metric=similarity_metric)

    def analyze_time_frequency(
        self,
        data: np.ndarray,
        fs: float,
        variable: str = "",
        method: Optional[str] = None,
    ) -> List[TFSpectrum]:
        methods_to_use = [method] if method else self.tf_methods
        results = []

        for m in methods_to_use:
            if m in self._tf_analyzers:
                result = self._tf_analyzers[m].analyze(data, fs, variable=variable)
                results.append(result)

        return results

    def analyze_dataset_tf(
        self,
        dataset: xr.Dataset,
        time_dim: str = "time",
        variables: Optional[List[str]] = None,
    ) -> Dict[str, List[TFSpectrum]]:
        results = {}
        vars_to_use = variables or list(dataset.data_vars)

        for var in vars_to_use:
            if var in dataset.data_vars and time_dim in dataset[var].dims:
                var_data = dataset[var]
                time_axis = var_data.dims.index(time_dim)

                spatial_dims = [i for i in range(var_data.ndim) if i != time_axis]
                if spatial_dims:
                    slices = [0] * var_data.ndim
                    for sd in spatial_dims:
                        slices[sd] = slice(None)
                    slices[time_axis] = slice(None)
                    time_series = var_data.values[tuple(slices)]
                    if time_series.ndim > 1:
                        time_series = np.nanmean(time_series, axis=tuple(range(1, time_series.ndim)))
                else:
                    time_series = var_data.values

                if "time" in dataset.coords:
                    times = dataset["time"].values
                    if len(times) > 1:
                        dt = float((times[1] - times[0]).astype("timedelta64[s]").astype(float))
                        fs = 1.0 / dt if dt > 0 else 1.0
                    else:
                        fs = 1.0
                else:
                    fs = 1.0

                results[var] = self.analyze_time_frequency(time_series, fs, variable=var)

        return results

    def infer_causality(
        self,
        variables: Dict[str, np.ndarray],
        names: Optional[List[str]] = None,
    ) -> List[CausalRelation]:
        return self.causal_inference.pc_stable(variables, names)

    def build_event_database(
        self,
        historical_data: Dict[str, xr.Dataset],
        event_descriptions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ClimateEvent]:
        events = []
        descriptions = event_descriptions or {}

        for event_id, ds in historical_data.items():
            meta = descriptions.get(event_id, {})
            fingerprint = self.pattern_matcher.extract_fingerprint(ds)
            var_stats = {}
            for var in ds.data_vars:
                data = ds[var].values
                valid = np.isfinite(data)
                if np.any(valid):
                    var_stats[var] = float(np.mean(data[valid]))

            event = ClimateEvent(
                event_id=event_id,
                start_time=meta.get("start_time", datetime(1900, 1, 1)),
                end_time=meta.get("end_time", datetime(1900, 12, 31)),
                description=meta.get("description", event_id),
                region=meta.get("region"),
                variables=var_stats,
                fingerprint=fingerprint,
                metadata=meta,
            )
            events.append(event)
            self.pattern_matcher.add_event(event)

        logger.info(f"Built climate event database with {len(events)} events")
        return events

    def search_similar_events(
        self,
        query_dataset: xr.Dataset,
        query_time_range: Optional[Tuple[datetime, datetime]] = None,
        top_k: Optional[int] = None,
    ) -> List[SimilarityResult]:
        if top_k is not None:
            self.pattern_matcher.top_k = top_k

        results = self.pattern_matcher.find_by_pattern(
            query_dataset,
            start_time=query_time_range[0] if query_time_range else None,
            end_time=query_time_range[1] if query_time_range else None,
        )
        return results

    def attribution_analysis(
        self,
        target_event: xr.Dataset,
        background_data: xr.Dataset,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        vars_to_use = variables or list(target_event.data_vars)
        attribution = {}

        for var in vars_to_use:
            if var in target_event.data_vars and var in background_data.data_vars:
                event_data = target_event[var].values
                bg_data = background_data[var].values

                event_valid = event_data[np.isfinite(event_data)]
                bg_valid = bg_data[np.isfinite(bg_data)]

                if len(event_valid) > 0 and len(bg_valid) > 0:
                    event_mean = np.mean(event_valid)
                    bg_mean = np.mean(bg_valid)
                    delta = event_mean - bg_mean

                    t_stat, p_value = stats.ttest_ind(event_valid, bg_valid, equal_var=False)

                    bg_std = np.std(bg_valid) if np.std(bg_valid) > 0 else 1.0
                    sigma_delta = delta / bg_std

                    attribution[var] = {
                        "event_mean": float(event_mean),
                        "background_mean": float(bg_mean),
                        "absolute_change": float(delta),
                        "relative_change_pct": float(delta / abs(bg_mean) * 100) if bg_mean != 0 else 0,
                        "sigma": float(sigma_delta),
                        "p_value": float(p_value),
                        "statistically_significant": bool(p_value < 0.05),
                    }

        return attribution
