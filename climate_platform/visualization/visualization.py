"""
Climate data visualization: 3D rendering, streamlines, isosurfaces, volume rendering, statistics.
"""

import logging
import json
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


class RenderBackend(Enum):
    PYVISTA = "pyvista"
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    MAYAVI = "mayavi"


class OutputFormat(Enum):
    HTML = "html"
    PNG = "png"
    MP4 = "mp4"
    GLB = "glb"
    VTK = "vtk"
    CSV = "csv"
    JSON = "json"


@dataclass
class VisualizationOutput:
    output_id: str
    format: OutputFormat
    data: bytes
    title: str = ""
    description: str = ""
    variables: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str, filename: Optional[str] = None) -> Path:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = filename or f"{self.output_id}.{self.format.value}"
        full_path = out_path / fname
        with open(full_path, "wb") as f:
            f.write(self.data)
        return full_path

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("utf-8")


@dataclass
class ColorMap:
    name: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    n_colors: int = 256
    reverse: bool = False

    @property
    def matplotlib_name(self) -> str:
        return self.name if not self.reverse else f"{self.name}_r"


class RendererBase(ABC):
    def __init__(self, backend: RenderBackend = RenderBackend.PLOTLY):
        self.backend = backend

    @abstractmethod
    def render(self, dataset: xr.Dataset, **kwargs) -> VisualizationOutput:
        pass

    def _get_lat_lon(self, dataset: xr.Dataset) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        lat = None
        lon = None
        for ln in ["lat", "latitude"]:
            if ln in dataset.coords:
                lat = dataset[ln].values
                break
        for ln in ["lon", "longitude"]:
            if ln in dataset.coords:
                lon = dataset[ln].values
                break
        return lat, lon


class WindStreamlineRenderer(RendererBase):
    def __init__(
        self,
        u_var: str = "u_wind",
        v_var: str = "v_wind",
        w_var: Optional[str] = None,
        density: float = 1.0,
        line_width: float = 1.0,
        max_length: int = 5000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.u_var = u_var
        self.v_var = v_var
        self.w_var = w_var
        self.density = density
        self.line_width = line_width
        self.max_length = max_length

    def render(
        self,
        dataset: xr.Dataset,
        variable: Optional[str] = None,
        colormap: Optional[ColorMap] = None,
        output_format: OutputFormat = OutputFormat.HTML,
        title: str = "Wind Streamlines",
        level_idx: Optional[int] = None,
        **kwargs,
    ) -> VisualizationOutput:
        lat, lon = self._get_lat_lon(dataset)
        if lat is None or lon is None:
            raise ValueError("Dataset missing lat/lon coordinates")

        u = dataset[self.u_var].values if self.u_var in dataset.data_vars else None
        v = dataset[self.v_var].values if self.v_var in dataset.data_vars else None

        if u is None or v is None:
            raise ValueError(f"Wind components {self.u_var}/{self.v_var} not found in dataset")

        cmap = colormap or ColorMap()

        u_2d = self._extract_2d(u, dataset[self.u_var].dims, level_idx)
        v_2d = self._extract_2d(v, dataset[self.v_var].dims, level_idx)

        if self.backend == RenderBackend.PLOTLY:
            return self._render_plotly(lat, lon, u_2d, v_2d, cmap, output_format, title, variable or "wind_speed", dataset)

        return self._render_simple(lat, lon, u_2d, v_2d, cmap, output_format, title)

    @staticmethod
    def _extract_2d(data: np.ndarray, dims: Tuple, level_idx: Optional[int]) -> np.ndarray:
        lat_dim = None
        lon_dim = None
        for i, d in enumerate(dims):
            if d in ["lat", "latitude", "y"]:
                lat_dim = i
            if d in ["lon", "longitude", "x"]:
                lon_dim = i

        if lat_dim is None or lon_dim is None:
            if data.ndim >= 2:
                return data.mean(axis=tuple(range(data.ndim - 2)))
            return data

        slicer = [0] * data.ndim
        slicer[lat_dim] = slice(None)
        slicer[lon_dim] = slice(None)

        for i in range(data.ndim):
            if slicer[i] == 0:
                if "level" in dims[i] or "height" in dims[i] or "depth" in dims[i]:
                    slicer[i] = level_idx if level_idx is not None else 0
                elif "time" in dims[i]:
                    slicer[i] = -1

        return data[tuple(slicer)]

    def _render_plotly(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        cmap: ColorMap,
        output_format: OutputFormat,
        title: str,
        speed_var: str,
        dataset: xr.Dataset,
    ) -> VisualizationOutput:
        try:
            import plotly.graph_objects as go

            lon_grid, lat_grid = np.meshgrid(lon, lat)
            speed = np.sqrt(u ** 2 + v ** 2)

            n_streamlines = int(20 * self.density)
            lat_steps = np.linspace(lat.min(), lat.max(), n_streamlines)
            lon_steps = np.linspace(lon.min(), lon.max(), n_streamlines)

            fig = go.Figure()

            if output_format in [OutputFormat.HTML, OutputFormat.JSON, OutputFormat.PNG]:
                fig.add_trace(go.Contour(
                    z=speed,
                    x=lon,
                    y=lat,
                    colorscale=cmap.matplotlib_name,
                    showscale=True,
                    colorbar=dict(title="Wind Speed (m/s)"),
                ))

                step = max(1, len(lat) // 20)
                fig.add_trace(go.Cone(
                    x=lon_grid[::step, ::step].flatten(),
                    y=lat_grid[::step, ::step].flatten(),
                    u=u[::step, ::step].flatten(),
                    v=v[::step, ::step].flatten(),
                    colorscale=cmap.matplotlib_name,
                    showscale=False,
                    sizemode="absolute",
                    sizeref=2,
                ))

                fig.update_layout(
                    title=title,
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=700,
                )

            if output_format == OutputFormat.HTML:
                html_content = fig.to_html(include_plotlyjs="cdn")
                return VisualizationOutput(
                    output_id=f"wind_streamlines_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    format=output_format,
                    data=html_content.encode("utf-8"),
                    title=title,
                    variables=[self.u_var, self.v_var, speed_var],
                )
            elif output_format == OutputFormat.JSON:
                json_content = fig.to_json()
                return VisualizationOutput(
                    output_id=f"wind_streamlines_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    format=output_format,
                    data=json_content.encode("utf-8"),
                    title=title,
                    variables=[self.u_var, self.v_var, speed_var],
                )
            elif output_format == OutputFormat.PNG:
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                return VisualizationOutput(
                    output_id=f"wind_streamlines_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    format=output_format,
                    data=img_bytes,
                    title=title,
                    variables=[self.u_var, self.v_var, speed_var],
                )
        except ImportError:
            pass

        return self._render_simple(lat, lon, u, v, cmap, output_format, title)

    def _render_simple(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        cmap: ColorMap,
        output_format: OutputFormat,
        title: str,
    ) -> VisualizationOutput:
        speed = np.sqrt(u ** 2 + v ** 2)
        data = {
            "title": title,
            "latitudes": lat.tolist(),
            "longitudes": lon.tolist(),
            "wind_u": u.tolist(),
            "wind_v": v.tolist(),
            "wind_speed": speed.tolist(),
            "colormap": cmap.__dict__,
            "stats": {
                "speed_max": float(np.nanmax(speed)),
                "speed_min": float(np.nanmin(speed)),
                "speed_mean": float(np.nanmean(speed)),
            }
        }

        return VisualizationOutput(
            output_id=f"wind_streamlines_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            format=OutputFormat.JSON if output_format != OutputFormat.JSON else output_format,
            data=json.dumps(data, indent=2, default=str).encode("utf-8"),
            title=title,
            variables=[self.u_var, self.v_var],
            metadata=data["stats"],
        )


class IsosurfaceRenderer(RendererBase):
    def __init__(
        self,
        levels: Optional[List[float]] = None,
        n_levels: int = 10,
        opacity: float = 0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.levels = levels
        self.n_levels = n_levels
        self.opacity = opacity

    def render(
        self,
        dataset: xr.Dataset,
        variable: str,
        colormap: Optional[ColorMap] = None,
        output_format: OutputFormat = OutputFormat.HTML,
        title: Optional[str] = None,
        **kwargs,
    ) -> VisualizationOutput:
        if variable not in dataset.data_vars:
            raise ValueError(f"Variable {variable} not found in dataset")

        data = dataset[variable].values
        lat, lon = self._get_lat_lon(dataset)
        cmap = colormap or ColorMap()
        plot_title = title or f"Isosurface: {variable}"

        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            raise ValueError(f"No valid data for variable {variable}")

        if self.levels is None:
            self.levels = np.linspace(np.nanmin(valid_data), np.nanmax(valid_data), self.n_levels).tolist()

        if self.backend == RenderBackend.PLOTLY:
            return self._render_plotly(data, lat, lon, variable, self.levels, cmap, output_format, plot_title)

        return self._render_simple(data, lat, lon, variable, self.levels, cmap, output_format, plot_title)

    def _render_plotly(
        self,
        data: np.ndarray,
        lat: Optional[np.ndarray],
        lon: Optional[np.ndarray],
        variable: str,
        levels: List[float],
        cmap: ColorMap,
        output_format: OutputFormat,
        title: str,
    ) -> VisualizationOutput:
        try:
            import plotly.graph_objects as go

            if data.ndim == 2 and lat is not None and lon is not None:
                lon_grid, lat_grid = np.meshgrid(lon, lat)
                fig = go.Figure()

                for i, level in enumerate(levels):
                    fig.add_trace(go.Contour(
                        z=data,
                        x=lon,
                        y=lat,
                        contours=dict(
                            type='constraint',
                            operation='=',
                            value=level,
                        ),
                        showscale=False,
                        line_width=2,
                    ))

                fig.add_trace(go.Heatmap(
                    z=data,
                    x=lon,
                    y=lat,
                    colorscale=cmap.matplotlib_name,
                    showscale=True,
                    opacity=0.7,
                ))

                fig.update_layout(title=title, height=700)

            elif data.ndim >= 3:
                data_3d = data
                if data.ndim > 3:
                    data_3d = data.reshape(data.shape[-3], data.shape[-2], data.shape[-1])

                fig = go.Figure(data=go.Volume(
                    x=np.linspace(0, 1, data_3d.shape[2]),
                    y=np.linspace(0, 1, data_3d.shape[1]),
                    z=np.linspace(0, 1, data_3d.shape[0]),
                    value=data_3d.flatten(),
                    isomin=levels[0],
                    isomax=levels[-1],
                    opacity=self.opacity,
                    surface_count=len(levels),
                    colorscale=cmap.matplotlib_name,
                ))
                fig.update_layout(title=title, height=700)
            else:
                fig = go.Figure()
                fig.update_layout(title="Unsupported data dimensions")

            if output_format == OutputFormat.HTML:
                html_content = fig.to_html(include_plotlyjs="cdn")
                return VisualizationOutput(
                    output_id=f"isosurface_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    format=output_format,
                    data=html_content.encode("utf-8"),
                    title=title,
                    variables=[variable],
                )
            elif output_format == OutputFormat.JSON:
                return VisualizationOutput(
                    output_id=f"isosurface_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    format=output_format,
                    data=fig.to_json().encode("utf-8"),
                    title=title,
                    variables=[variable],
                )
            elif output_format == OutputFormat.PNG:
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                return VisualizationOutput(
                    output_id=f"isosurface_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    format=output_format,
                    data=img_bytes,
                    title=title,
                    variables=[variable],
                )
        except ImportError:
            pass

        return self._render_simple(data, lat, lon, variable, levels, cmap, output_format, title)

    def _render_simple(
        self,
        data: np.ndarray,
        lat: Optional[np.ndarray],
        lon: Optional[np.ndarray],
        variable: str,
        levels: List[float],
        cmap: ColorMap,
        output_format: OutputFormat,
        title: str,
    ) -> VisualizationOutput:
        result = {
            "title": title,
            "variable": variable,
            "levels": levels,
            "colormap": cmap.__dict__,
            "shape": list(data.shape),
            "stats": {
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
            }
        }
        if lat is not None:
            result["latitudes"] = lat.tolist()
        if lon is not None:
            result["longitudes"] = lon.tolist()

        return VisualizationOutput(
            output_id=f"isosurface_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            format=OutputFormat.JSON,
            data=json.dumps(result, indent=2, default=str).encode("utf-8"),
            title=title,
            variables=[variable],
            metadata=result["stats"],
        )


class VolumeRenderer(RendererBase):
    def __init__(
        self,
        opacity: float = 0.3,
        sampling_rate: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.opacity = opacity
        self.sampling_rate = sampling_rate

    def render(
        self,
        dataset: xr.Dataset,
        variable: str,
        colormap: Optional[ColorMap] = None,
        output_format: OutputFormat = OutputFormat.HTML,
        title: Optional[str] = None,
        **kwargs,
    ) -> VisualizationOutput:
        if variable not in dataset.data_vars:
            raise ValueError(f"Variable {variable} not found in dataset")

        data = dataset[variable].values
        cmap = colormap or ColorMap()
        plot_title = title or f"Volume Rendering: {variable}"

        iso = IsosurfaceRenderer(backend=self.backend)
        return iso.render(dataset, variable, cmap, output_format, plot_title)


@dataclass
class StatisticsData:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    percentiles: Dict[int, np.ndarray] = field(default_factory=dict)
    trend: Optional[np.ndarray] = None
    climatology: Optional[np.ndarray] = None
    anomalies: Optional[np.ndarray] = None
    spatial_moments: Dict[str, np.ndarray] = field(default_factory=dict)


class StatisticalReport:
    def __init__(self):
        self._reports: Dict[str, StatisticsData] = {}

    def compute_statistics(
        self,
        dataset: xr.Dataset,
        variables: Optional[List[str]] = None,
        dims: Optional[List[str]] = None,
        percentiles: Optional[List[int]] = None,
    ) -> Dict[str, StatisticsData]:
        vars_to_process = variables or list(dataset.data_vars)
        pcts = percentiles or [10, 25, 50, 75, 90]

        for var in vars_to_process:
            if var not in dataset.data_vars:
                continue

            data = dataset[var]
            stats = StatisticsData()

            try:
                stats.mean = data.mean(dim=dims).values if dims else float(np.nanmean(data.values))
                stats.std = data.std(dim=dims).values if dims else float(np.nanstd(data.values))
                stats.min = data.min(dim=dims).values if dims else float(np.nanmin(data.values))
                stats.max = data.max(dim=dims).values if dims else float(np.nanmax(data.values))

                for p in pcts:
                    stats.percentiles[p] = (
                        data.quantile(p / 100.0, dim=dims).values if dims
                        else float(np.nanpercentile(data.values, p))
                    )

                if "time" in data.dims:
                    time_axis = data.dims.index("time")
                    stats.trend = self._compute_trend(data.values, time_axis)

                stats.spatial_moments = self._compute_spatial_moments(data)

            except Exception as e:
                logger.warning(f"Failed to compute stats for {var}: {e}")

            self._reports[var] = stats

        return self._reports

    @staticmethod
    def _compute_trend(data: np.ndarray, time_axis: int) -> np.ndarray:
        n = data.shape[time_axis]
        if n < 2:
            return np.zeros_like(np.take(data, 0, axis=time_axis))

        x = np.arange(n)
        other_axes = tuple(i for i in range(data.ndim) if i != time_axis)

        if not other_axes:
            from scipy import stats as scipy_stats
            slope, _, _, _, _ = scipy_stats.linregress(x, data)
            return np.array(slope)

        moved = np.moveaxis(data, time_axis, 0)
        flat = moved.reshape(n, -1)
        slopes = np.zeros(flat.shape[1])

        for i in range(flat.shape[1]):
            y = flat[:, i]
            valid = np.isfinite(y)
            if np.sum(valid) >= 2:
                from scipy import stats as scipy_stats
                try:
                    slopes[i], _, _, _, _ = scipy_stats.linregress(x[valid], y[valid])
                except Exception:
                    slopes[i] = 0

        out_shape = list(moved.shape[1:])
        return slopes.reshape(out_shape)

    @staticmethod
    def _compute_spatial_moments(data: xr.DataArray) -> Dict[str, np.ndarray]:
        result = {}
        lat = None
        lon = None
        for ln in ["lat", "latitude"]:
            if ln in data.coords:
                lat = data[ln].values
                break
        for ln in ["lon", "longitude"]:
            if ln in data.coords:
                lon = data[ln].values
                break

        if lat is not None and lon is not None and data.ndim >= 2:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            values = data.values
            if values.ndim > 2:
                values = np.nanmean(values, axis=tuple(range(values.ndim - 2)))

            valid = np.isfinite(values)
            if np.any(valid):
                total = np.sum(values[valid])
                if total > 0:
                    center_lon = np.sum(lon_grid[valid] * values[valid]) / total
                    center_lat = np.sum(lat_grid[valid] * values[valid]) / total
                    result["center_of_mass"] = np.array([center_lat, center_lon])

        return result

    def generate_report(
        self,
        dataset: xr.Dataset,
        variables: Optional[List[str]] = None,
        output_format: OutputFormat = OutputFormat.JSON,
        title: str = "Statistical Report",
    ) -> VisualizationOutput:
        stats = self.compute_statistics(dataset, variables)
        report = {
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "variables": {},
        }

        for var, data in stats.items():
            var_report = {}
            if data.mean is not None:
                var_report["mean"] = float(data.mean) if np.ndim(data.mean) == 0 else "spatial_map"
            if data.std is not None:
                var_report["std"] = float(data.std) if np.ndim(data.std) == 0 else "spatial_map"
            if data.min is not None:
                var_report["min"] = float(data.min) if np.ndim(data.min) == 0 else "spatial_map"
            if data.max is not None:
                var_report["max"] = float(data.max) if np.ndim(data.max) == 0 else "spatial_map"
            if data.percentiles:
                var_report["percentiles"] = {
                    str(p): float(v) if np.ndim(v) == 0 else "spatial_map"
                    for p, v in data.percentiles.items()
                }
            report["variables"][var] = var_report

        if output_format == OutputFormat.JSON:
            return VisualizationOutput(
                output_id=f"stats_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                format=OutputFormat.JSON,
                data=json.dumps(report, indent=2, default=str).encode("utf-8"),
                title=title,
                variables=list(stats.keys()),
            )

        if output_format == OutputFormat.CSV:
            lines = ["variable,statistic,value"]
            for var, data in stats.items():
                for stat_name, stat_val in [
                    ("mean", data.mean), ("std", data.std),
                    ("min", data.min), ("max", data.max),
                ]:
                    if stat_val is not None and np.ndim(stat_val) == 0:
                        lines.append(f"{var},{stat_name},{float(stat_val)}")
            return VisualizationOutput(
                output_id=f"stats_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                format=OutputFormat.CSV,
                data="\n".join(lines).encode("utf-8"),
                title=title,
                variables=list(stats.keys()),
            )

        return VisualizationOutput(
            output_id=f"stats_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            format=OutputFormat.JSON,
            data=json.dumps(report, indent=2, default=str).encode("utf-8"),
            title=title,
            variables=list(stats.keys()),
        )


class TemporalPlayer:
    def __init__(self, fps: int = 24, loop: bool = True):
        self.fps = fps
        self.loop = loop
        self._frames: List[VisualizationOutput] = []
        self._current_frame = 0

    def add_frame(self, frame: VisualizationOutput):
        self._frames.append(frame)

    def add_frames(self, frames: List[VisualizationOutput]):
        self._frames.extend(frames)

    def generate_animation(
        self,
        output_format: OutputFormat = OutputFormat.MP4,
        title: str = "Temporal Animation",
    ) -> VisualizationOutput:
        if not self._frames:
            raise ValueError("No frames to animate")

        frame_data = []
        for frame in self._frames:
            if frame.format == OutputFormat.JSON:
                try:
                    frame_data.append(json.loads(frame.data.decode("utf-8")))
                except Exception:
                    frame_data.append(frame.metadata)

        animation = {
            "title": title,
            "fps": self.fps,
            "loop": self.loop,
            "num_frames": len(self._frames),
            "duration_seconds": len(self._frames) / self.fps,
            "frames": frame_data,
            "timestamps": [f.timestamp.isoformat() for f in self._frames],
        }

        return VisualizationOutput(
            output_id=f"animation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            format=OutputFormat.JSON if output_format != OutputFormat.MP4 else output_format,
            data=json.dumps(animation, indent=2, default=str).encode("utf-8"),
            title=title,
            variables=list(set(v for f in self._frames for v in f.variables)),
            metadata={"num_frames": len(self._frames), "fps": self.fps},
        )

    def __len__(self) -> int:
        return len(self._frames)


class VisualizationEngine:
    def __init__(
        self,
        backend: RenderBackend = RenderBackend.PLOTLY,
        output_formats: Optional[List[OutputFormat]] = None,
        interactive: bool = True,
        render_quality: str = "high",
    ):
        self.backend = backend
        self.output_formats = output_formats or [
            OutputFormat.HTML, OutputFormat.PNG, OutputFormat.JSON
        ]
        self.interactive = interactive
        self.render_quality = render_quality

        self.wind_renderer = WindStreamlineRenderer(backend=backend)
        self.isosurface_renderer = IsosurfaceRenderer(backend=backend)
        self.volume_renderer = VolumeRenderer(backend=backend)
        self.statistical_report = StatisticalReport()
        self.temporal_player = TemporalPlayer()

    def render_wind_streamlines(
        self,
        dataset: xr.Dataset,
        output_format: OutputFormat = OutputFormat.HTML,
        **kwargs,
    ) -> VisualizationOutput:
        return self.wind_renderer.render(dataset, output_format=output_format, **kwargs)

    def render_isosurface(
        self,
        dataset: xr.Dataset,
        variable: str,
        output_format: OutputFormat = OutputFormat.HTML,
        **kwargs,
    ) -> VisualizationOutput:
        return self.isosurface_renderer.render(dataset, variable, output_format=output_format, **kwargs)

    def render_volume(
        self,
        dataset: xr.Dataset,
        variable: str,
        output_format: OutputFormat = OutputFormat.HTML,
        **kwargs,
    ) -> VisualizationOutput:
        return self.volume_renderer.render(dataset, variable, output_format=output_format, **kwargs)

    def render_statistics(
        self,
        dataset: xr.Dataset,
        variables: Optional[List[str]] = None,
        output_format: OutputFormat = OutputFormat.JSON,
        **kwargs,
    ) -> VisualizationOutput:
        return self.statistical_report.generate_report(
            dataset, variables, output_format, **kwargs
        )

    def render_all(
        self,
        dataset: xr.Dataset,
        variables: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, VisualizationOutput]:
        results = {}
        vars_to_use = variables or list(dataset.data_vars)

        try:
            if "u_wind" in dataset.data_vars and "v_wind" in dataset.data_vars:
                results["wind_streamlines"] = self.render_wind_streamlines(dataset)
        except Exception as e:
            logger.warning(f"Failed to render wind streamlines: {e}")

        for var in vars_to_use[:3]:
            try:
                results[f"isosurface_{var}"] = self.render_isosurface(dataset, var)
            except Exception as e:
                logger.warning(f"Failed to render isosurface for {var}: {e}")

        try:
            results["statistics"] = self.render_statistics(dataset, variables)
        except Exception as e:
            logger.warning(f"Failed to generate statistics: {e}")

        if output_dir:
            for name, output in results.items():
                try:
                    output.save(output_dir, f"{name}.{output.format.value}")
                except Exception as e:
                    logger.warning(f"Failed to save {name}: {e}")

        return results

    def create_temporal_animation(
        self,
        dataset: xr.Dataset,
        variable: str,
        time_dim: str = "time",
        renderer: str = "isosurface",
        output_format: OutputFormat = OutputFormat.JSON,
    ) -> VisualizationOutput:
        if variable not in dataset.data_vars or time_dim not in dataset.dims:
            raise ValueError(f"Variable {variable} or dimension {time_dim} not found")

        time_values = dataset[time_dim].values
        player = TemporalPlayer()

        for i, t in enumerate(time_values):
            frame_ds = dataset.isel({time_dim: i})
            ts = pd.to_datetime(t).to_pydatetime() if not isinstance(t, datetime) else t

            if renderer == "isosurface":
                frame = self.isosurface_renderer.render(
                    frame_ds, variable, output_format=OutputFormat.JSON,
                    title=f"{variable} at {ts}"
                )
            else:
                frame = self.volume_renderer.render(
                    frame_ds, variable, output_format=OutputFormat.JSON,
                    title=f"{variable} at {ts}"
                )

            frame.timestamp = ts
            player.add_frame(frame)

        return player.generate_animation(output_format=output_format, title=f"{variable} Temporal Evolution")
