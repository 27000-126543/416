"""
Fully coupled atmosphere-ocean-land-sea ice numerical model framework.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class CouplingType(Enum):
    UNCOUPLED = "uncoupled"
    ONE_WAY = "one_way"
    TWO_WAY = "two_way"
    FULLY_COUPLED = "fully_coupled"


class CouplingMethod(Enum):
    DIRECT = "direct"
    FLUX = "flux"
    RELAXATION = "relaxation"


@dataclass
class CouplingVariableMap:
    source_var: str
    target_var: str
    method: CouplingMethod = CouplingMethod.RELAXATION
    relaxation_time_seconds: float = 86400.0
    scale_factor: float = 1.0
    source_level: Optional[float] = None
    target_level: Optional[float] = None


class ModelDomain(Enum):
    GLOBAL = "global"
    REGIONAL = "regional"
    LIMITED_AREA = "limited_area"


@dataclass
class ModelState:
    timestamp: datetime
    data: xr.Dataset
    step: int = 0
    model_time: timedelta = field(default_factory=timedelta)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_variable(self, name: str) -> Optional[np.ndarray]:
        if name in self.data.data_vars:
            return self.data[name].values
        return None

    def set_variable(self, name: str, values: np.ndarray):
        if name in self.data.data_vars:
            self.data[name].values = values


class CouplingInterface:
    def __init__(
        self,
        source_model: str,
        target_model: str,
        variable_maps: List[CouplingVariableMap],
    ):
        self.source_model = source_model
        self.target_model = target_model
        self.variable_maps = variable_maps
        self.buffer: Dict[str, np.ndarray] = {}
        self.coupling_count: int = 0

    def transfer(self, source_state: ModelState) -> Dict[str, np.ndarray]:
        transferred = {}
        for vmap in self.variable_maps:
            data = source_state.get_variable(vmap.source_var)
            if data is None:
                logger.warning(f"Coupling variable not found in source: {vmap.source_var}")
                continue

            if data.ndim >= 3 and vmap.source_level is not None:
                level_coord = None
                for coord_name in ["level", "depth", "soil_level"]:
                    if coord_name in source_state.data.coords:
                        level_coord = source_state.data[coord_name].values
                        break
                if level_coord is not None:
                    level_idx = int(np.argmin(np.abs(level_coord - vmap.source_level)))
                    if data.ndim == 3:
                        data = data[:, :, level_idx]
                    elif data.ndim == 4:
                        data = data[:, :, level_idx, 0]

            transferred[vmap.source_var] = data.copy()
        self.buffer = transferred
        return transferred

    def apply_to_target(
        self,
        target_state: ModelState,
        remapped_data: Dict[str, np.ndarray],
        dt: Optional[timedelta] = None,
    ):
        for vmap in self.variable_maps:
            if vmap.source_var not in remapped_data:
                continue
            if vmap.target_var not in target_state.data.data_vars:
                logger.warning(f"Coupling target variable not found: {vmap.target_var}")
                continue

            source_data = remapped_data[vmap.source_var] * vmap.scale_factor
            target_data = target_state.get_variable(vmap.target_var)

            if target_data is None:
                continue

            try:
                if vmap.method == CouplingMethod.DIRECT:
                    if source_data.shape == target_data.shape:
                        target_data[:] = source_data
                    else:
                        logger.warning(
                            f"Shape mismatch for direct coupling: {vmap.source_var} -> {vmap.target_var}: "
                            f"{source_data.shape} vs {target_data.shape}"
                        )

                elif vmap.method == CouplingMethod.RELAXATION:
                    tau = vmap.relaxation_time_seconds
                    dt_sec = dt.total_seconds() if dt else 3600.0
                    alpha = min(1.0, dt_sec / tau) if tau > 0 else 1.0

                    if source_data.shape == target_data.shape:
                        target_data[:] = target_data + alpha * (source_data - target_data)
                    elif source_data.ndim == 2 and target_data.ndim == 3:
                        for k in range(min(3, target_data.shape[-1])):
                            target_data[:, :, k] = (
                                target_data[:, :, k] + alpha * (source_data - target_data[:, :, k])
                            )
                    elif source_data.ndim == 3 and target_data.ndim == 2:
                        target_data[:] = target_data + alpha * (source_data[:, :, 0] - target_data)

                elif vmap.method == CouplingMethod.FLUX:
                    dt_sec = dt.total_seconds() if dt else 3600.0
                    if source_data.shape == target_data.shape:
                        target_data[:] = target_data + source_data * dt_sec
                    elif source_data.ndim == 2 and target_data.ndim == 3:
                        for k in range(min(3, target_data.shape[-1])):
                            target_data[:, :, k] = target_data[:, :, k] + source_data * dt_sec

                target_state.set_variable(vmap.target_var, target_data)
            except Exception as e:
                logger.error(f"Error applying coupling {vmap.source_var} -> {vmap.target_var}: {e}")

        self.coupling_count += 1


class CouplingScheduler:
    def __init__(self, coupling_interval_seconds: int = 900):
        self.coupling_interval = timedelta(seconds=coupling_interval_seconds)
        self.last_coupling_time: Optional[datetime] = None
        self.interfaces: List[CouplingInterface] = []
        self._step_count = 0

    def add_interface(self, interface: CouplingInterface):
        self.interfaces.append(interface)

    def needs_coupling(self, current_time: datetime) -> bool:
        if self.last_coupling_time is None:
            return True
        return (current_time - self.last_coupling_time) >= self.coupling_interval

    def execute_coupling(
        self,
        model_states: Dict[str, ModelState],
        remap_function: Optional[Callable] = None,
        dt: Optional[timedelta] = None,
    ) -> Dict[str, ModelState]:
        for interface in self.interfaces:
            if interface.source_model in model_states and interface.target_model in model_states:
                source_state = model_states[interface.source_model]
                target_state = model_states[interface.target_model]
                data = interface.transfer(source_state)

                if remap_function:
                    data = remap_function(
                        interface.source_model, interface.target_model, data,
                        source_state, target_state,
                    )

                interface.apply_to_target(target_state, data, dt=dt)

        if model_states:
            self.last_coupling_time = list(model_states.values())[0].timestamp
        self._step_count += 1
        return model_states

    @property
    def coupling_count(self) -> int:
        return self._step_count


class ModelComponent(ABC):
    def __init__(
        self,
        name: str,
        domain: ModelDomain = ModelDomain.GLOBAL,
        time_step_seconds: int = 300,
        resolution_deg: float = 0.25,
    ):
        self.name = name
        self.domain = domain
        self.time_step = timedelta(seconds=time_step_seconds)
        self.resolution_deg = resolution_deg
        self.state: Optional[ModelState] = None
        self.parameters: Dict[str, Any] = {}
        self._initialized = False

    @abstractmethod
    def initialize(self, initial_state: Optional[ModelState] = None, **kwargs) -> ModelState:
        pass

    @abstractmethod
    def step(self, state: ModelState, dt: timedelta) -> ModelState:
        pass

    @abstractmethod
    def output_variables(self) -> List[str]:
        pass

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def set_parameter(self, name: str, value: Any):
        self.parameters[name] = value

    def get_parameter(self, name: str, default: Any = None) -> Any:
        return self.parameters.get(name, default)


class AtmosphericModel(ModelComponent):
    STANDARD_VARIABLES = [
        "temperature", "u_wind", "v_wind", "w_wind",
        "geopotential", "relative_humidity", "specific_humidity",
        "pressure", "surface_pressure", "precipitation",
        "shortwave_radiation", "longwave_radiation",
        "sensible_heat_flux", "latent_heat_flux",
        "cloud_cover", "cloud_water", "cloud_ice",
    ]

    def __init__(
        self,
        name: str = "atmosphere",
        vertical_levels: int = 30,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.vertical_levels = vertical_levels
        self.dynamics_scheme = "semi_implicit"
        self.physics_schemes: Dict[str, str] = {
            "convection": "tiedtke",
            "microphysics": "wsm6",
            "pbl": "ysu",
            "radiation": "rrtmg",
            "land_surface": "noah",
        }

    def output_variables(self) -> List[str]:
        return self.STANDARD_VARIABLES

    def initialize(self, initial_state: Optional[ModelState] = None, **kwargs) -> ModelState:
        if initial_state is not None:
            self.state = initial_state
            self._initialized = True
            return self.state

        lat = np.arange(-90, 90 + self.resolution_deg, self.resolution_deg)
        lon = np.arange(0, 360, self.resolution_deg)
        levels = np.linspace(1000, 100, self.vertical_levels)

        lat = lat[:int(180 / self.resolution_deg) + 1]
        lon = lon[:int(360 / self.resolution_deg)]

        n_lat = len(lat)
        n_lon = len(lon)
        n_lev = self.vertical_levels

        ds = xr.Dataset()
        ds["lat"] = ("lat", lat)
        ds["lon"] = ("lon", lon)
        ds["level"] = ("level", levels)

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lon_3d, lat_3d, lev_3d = np.meshgrid(lon, lat, levels)

        base_temp = 288.15 - 6.5e-3 * lev_3d
        temp_lapse = np.sin(np.radians(lat_3d)) ** 2 * 20
        ds["temperature"] = (["lat", "lon", "level"], base_temp - temp_lapse)

        u_jet = 40 * np.exp(-((lev_3d - 250) ** 2) / 10000) * np.cos(np.radians(lat_3d)) ** 2
        ds["u_wind"] = (["lat", "lon", "level"], u_jet)
        ds["v_wind"] = (["lat", "lon", "level"], np.zeros((n_lat, n_lon, n_lev)))
        ds["w_wind"] = (["lat", "lon", "level"], np.zeros((n_lat, n_lon, n_lev)) * 0.01)

        pressure_surface = 101325 * np.exp(-0.00012 * (90 - np.abs(lat_grid)))
        ds["surface_pressure"] = (["lat", "lon"], pressure_surface)
        ds["pressure"] = (["lat", "lon", "level"], np.tile(levels[::-1], (n_lat, n_lon, 1)) * 100)

        rh = 0.7 * np.exp(-((lev_3d - 500) ** 2) / 50000) * (0.5 + 0.5 * np.cos(np.radians(lat_3d)))
        ds["relative_humidity"] = (["lat", "lon", "level"], np.clip(rh, 0.01, 0.99))

        sh = 0.02 * np.exp(-((lev_3d - 500) ** 2) / 30000) * np.exp(-(np.abs(lat_3d) / 60) ** 2)
        ds["specific_humidity"] = (["lat", "lon", "level"], np.clip(sh, 0.0001, 0.04))

        ds["precipitation"] = (["lat", "lon"], 0.001 * np.exp(-(np.abs(lat_grid) / 30) ** 2))
        ds["geopotential"] = (["lat", "lon", "level"], 9.81 * (1000 - lev_3d) * 10)
        ds["cloud_cover"] = (["lat", "lon", "level"], 0.3 * rh)
        ds["cloud_water"] = (["lat", "lon", "level"], 0.0005 * rh)
        ds["cloud_ice"] = (["lat", "lon", "level"], 0.0001 * rh * (lev_3d < 400))

        ds["shortwave_radiation"] = (["lat", "lon"], 400 * np.cos(np.radians(lat_grid)))
        ds["longwave_radiation"] = (["lat", "lon"], 300 * np.ones_like(lat_grid))

        shf = 20 * np.cos(np.radians(lat_grid))
        ds["sensible_heat_flux"] = (["lat", "lon"], shf)
        ds["latent_heat_flux"] = (["lat", "lon"], shf * 3)

        start_time = kwargs.get("start_time", datetime(2020, 1, 1))
        self.state = ModelState(
            timestamp=start_time,
            data=ds,
            step=0,
            model_time=timedelta(0),
            metadata={"model": self.name, "vertical_levels": self.vertical_levels},
        )
        self._initialized = True
        return self.state

    def step(self, state: ModelState, dt: timedelta) -> ModelState:
        dt_hours = dt.total_seconds() / 3600.0
        new_state = ModelState(
            timestamp=state.timestamp + dt,
            data=state.data.copy(),
            step=state.step + 1,
            model_time=state.model_time + dt,
            metadata=state.metadata.copy(),
        )

        if "temperature" in new_state.data.data_vars:
            temp = new_state.data["temperature"].values
            adiabatic_lapse = np.zeros_like(temp)
            if "w_wind" in new_state.data.data_vars:
                w = new_state.data["w_wind"].values
                adiabatic_lapse = -0.0098 * w * dt_hours * 3600
            new_state.data["temperature"].values = temp + adiabatic_lapse + np.random.randn(*temp.shape) * 0.01

        if "u_wind" in new_state.data.data_vars and "v_wind" in new_state.data.data_vars:
            u = new_state.data["u_wind"].values
            v = new_state.data["v_wind"].values
            coriolis = 2 * 7.2921e-5 * np.sin(np.radians(new_state.data["lat"].values))
            if u.ndim >= 3:
                coriolis_3d = coriolis[:, np.newaxis, np.newaxis] if u.ndim == 3 else coriolis
                coriolis_expanded = np.zeros_like(u)
                for i in range(u.shape[-1]):
                    if u.ndim == 3:
                        coriolis_expanded[:, :, i] = coriolis[:, np.newaxis]
                dv_dt = -coriolis_expanded * u * dt_hours
                du_dt = coriolis_expanded * v * dt_hours
                new_state.data["u_wind"].values = u + du_dt + np.random.randn(*u.shape) * 0.1
                new_state.data["v_wind"].values = v + dv_dt + np.random.randn(*v.shape) * 0.1

        if "relative_humidity" in new_state.data.data_vars:
            rh = new_state.data["relative_humidity"].values
            new_state.data["relative_humidity"].values = np.clip(
                rh + np.random.randn(*rh.shape) * 0.01, 0.0, 1.0
            )

        self.state = new_state
        return new_state


class OceanModel(ModelComponent):
    STANDARD_VARIABLES = [
        "water_temperature", "salinity",
        "u_current", "v_current", "w_current",
        "sea_surface_temperature", "sea_surface_height",
        "mixed_layer_depth", "ocean_heat_content",
    ]

    def __init__(
        self,
        name: str = "ocean",
        vertical_levels: int = 40,
        max_depth_m: float = 5000.0,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.vertical_levels = vertical_levels
        self.max_depth_m = max_depth_m
        self.ocean_dynamics = "primitive_equations"

    def output_variables(self) -> List[str]:
        return self.STANDARD_VARIABLES

    def initialize(self, initial_state: Optional[ModelState] = None, **kwargs) -> ModelState:
        if initial_state is not None:
            self.state = initial_state
            self._initialized = True
            return self.state

        lat = np.arange(-80, 80 + self.resolution_deg, self.resolution_deg)
        lon = np.arange(0, 360, self.resolution_deg)
        depths = np.geomspace(5, self.max_depth_m, self.vertical_levels)

        lat = lat[:int(160 / self.resolution_deg) + 1]
        lon = lon[:int(360 / self.resolution_deg)]

        n_lat = len(lat)
        n_lon = len(lon)
        n_lev = self.vertical_levels

        ds = xr.Dataset()
        ds["lat"] = ("lat", lat)
        ds["lon"] = ("lon", lon)
        ds["depth"] = ("depth", depths)

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lon_3d, lat_3d, depth_3d = np.meshgrid(lon, lat, depths)

        sst = 30 * np.cos(np.radians(lat_grid)) - 2
        ds["sea_surface_temperature"] = (["lat", "lon"], sst)

        temp_profile = np.zeros((n_lat, n_lon, n_lev))
        for i, d in enumerate(depths):
            thermocline = np.exp(-(d - 200) ** 2 / 50000)
            temp_profile[:, :, i] = sst - 20 * (1 - np.exp(-d / 1000)) + thermocline * 5 * np.cos(np.radians(lat_grid))
        ds["water_temperature"] = (["lat", "lon", "depth"], temp_profile)

        sal_profile = 35 + 1 * np.exp(-(depth_3d - 1000) ** 2 / 200000) - 2 * (np.abs(lat_3d) < 60) * np.exp(-depth_3d / 100)
        ds["salinity"] = (["lat", "lon", "depth"], np.clip(sal_profile, 32, 38))

        gulf_stream = 1.0 * np.exp(-(np.abs(lat_3d - 40) ** 2) / 200) * np.exp(-depth_3d / 500)
        ds["u_current"] = (["lat", "lon", "depth"], gulf_stream + np.random.randn(n_lat, n_lon, n_lev) * 0.05)
        ds["v_current"] = (["lat", "lon", "depth"], np.random.randn(n_lat, n_lon, n_lev) * 0.05)
        ds["w_current"] = (["lat", "lon", "depth"], np.random.randn(n_lat, n_lon, n_lev) * 0.001)

        ds["sea_surface_height"] = (["lat", "lon"], 0.5 * np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid)))
        ds["mixed_layer_depth"] = (["lat", "lon"], 50 + 100 * np.cos(np.radians(lat_grid)) ** 2)

        heat_content = np.trapz(temp_profile * 1025 * 4186, depths, axis=2)
        ds["ocean_heat_content"] = (["lat", "lon"], heat_content)

        start_time = kwargs.get("start_time", datetime(2020, 1, 1))
        self.state = ModelState(
            timestamp=start_time,
            data=ds,
            step=0,
            model_time=timedelta(0),
            metadata={"model": self.name, "vertical_levels": self.vertical_levels, "max_depth_m": self.max_depth_m},
        )
        self._initialized = True
        return self.state

    def step(self, state: ModelState, dt: timedelta) -> ModelState:
        dt_hours = dt.total_seconds() / 3600.0
        new_state = ModelState(
            timestamp=state.timestamp + dt,
            data=state.data.copy(),
            step=state.step + 1,
            model_time=state.model_time + dt,
            metadata=state.metadata.copy(),
        )

        if "water_temperature" in new_state.data.data_vars:
            temp = new_state.data["water_temperature"].values
            diffusion = 0
            if temp.ndim >= 3 and "depth" in new_state.data.coords:
                depth_axis = list(new_state.data["water_temperature"].dims).index("depth")
                diff = np.gradient(temp, axis=depth_axis)
                diffusion = -0.0001 * diff * dt_hours
            new_state.data["water_temperature"].values = temp + diffusion + np.random.randn(*temp.shape) * 0.001

        if "sea_surface_temperature" in new_state.data.data_vars:
            sst = new_state.data["sea_surface_temperature"].values
            new_state.data["sea_surface_temperature"].values = sst + np.random.randn(*sst.shape) * 0.01

        self.state = new_state
        return new_state


class LandModel(ModelComponent):
    STANDARD_VARIABLES = [
        "skin_temperature", "soil_temperature", "soil_moisture",
        "snow_depth", "snow_water_equivalent",
        "vegetation_fraction", "leaf_area_index",
        "surface_roughness", "albedo",
        "runoff", "evapotranspiration",
    ]

    def __init__(
        self,
        name: str = "land",
        soil_levels: int = 4,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.soil_levels = soil_levels
        self.surface_scheme = "noah_mp"
        self.vegetation_types = 20

    def output_variables(self) -> List[str]:
        return self.STANDARD_VARIABLES

    def initialize(self, initial_state: Optional[ModelState] = None, **kwargs) -> ModelState:
        if initial_state is not None:
            self.state = initial_state
            self._initialized = True
            return self.state

        lat = np.arange(-90, 90 + self.resolution_deg, self.resolution_deg)
        lon = np.arange(0, 360, self.resolution_deg)
        soil_depths = [0.1, 0.4, 1.0, 2.0]

        lat = lat[:int(180 / self.resolution_deg) + 1]
        lon = lon[:int(360 / self.resolution_deg)]

        n_lat = len(lat)
        n_lon = len(lon)

        ds = xr.Dataset()
        ds["lat"] = ("lat", lat)
        ds["lon"] = ("lon", lon)
        ds["soil_level"] = ("soil_level", soil_depths)

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        land_mask = np.ones_like(lat_grid)
        ocean_indices = [
            (slice(None), slice(160, 290)),
            (slice(40, 120), slice(290, 360)),
            (slice(30, 80), slice(0, 60)),
        ]
        for s in ocean_indices:
            try:
                land_mask[s] = 0
            except Exception:
                pass

        ds["land_mask"] = (["lat", "lon"], land_mask)

        skin_temp = 288.15 - 30 * np.abs(np.sin(np.radians(lat_grid)))
        skin_temp = skin_temp * land_mask + (273.15 + 10) * (1 - land_mask)
        ds["skin_temperature"] = (["lat", "lon"], skin_temp)

        soil_temp = np.zeros((n_lat, n_lon, self.soil_levels))
        for i in range(self.soil_levels):
            damping = np.exp(-soil_depths[i] / 2)
            soil_temp[:, :, i] = skin_temp * damping + 283 * (1 - damping)
        ds["soil_temperature"] = (["lat", "lon", "soil_level"], soil_temp)

        soil_moist = 0.3 * land_mask * (0.5 + 0.5 * np.cos(np.radians(lat_grid)))
        soil_moisture = np.zeros((n_lat, n_lon, self.soil_levels))
        for i in range(self.soil_levels):
            drainage = np.exp(-soil_depths[i] / 3)
            soil_moisture[:, :, i] = soil_moist * drainage
        ds["soil_moisture"] = (["lat", "lon", "soil_level"], soil_moisture)

        snow = np.maximum(0, -skin_temp + 270) * land_mask * (np.abs(lat_grid) > 30)
        ds["snow_depth"] = (["lat", "lon"], snow / 100)
        ds["snow_water_equivalent"] = (["lat", "lon"], snow * 0.1)

        veg_frac = 0.5 * land_mask * np.exp(-(np.abs(lat_grid) - 20) ** 2 / 800)
        ds["vegetation_fraction"] = (["lat", "lon"], veg_frac)
        ds["leaf_area_index"] = (["lat", "lon"], veg_frac * 5)

        albedo = 0.15 + 0.5 * (snow > 0.1) + 0.2 * (1 - land_mask)
        ds["albedo"] = (["lat", "lon"], np.clip(albedo, 0.05, 0.9))

        roughness = 0.1 + 2.0 * veg_frac + 0.01 * (1 - land_mask)
        ds["surface_roughness"] = (["lat", "lon"], roughness)

        ds["runoff"] = (["lat", "lon"], 0.001 * soil_moist)
        ds["evapotranspiration"] = (["lat", "lon"], 0.0005 * veg_frac * np.maximum(0, skin_temp - 273.15))

        start_time = kwargs.get("start_time", datetime(2020, 1, 1))
        self.state = ModelState(
            timestamp=start_time,
            data=ds,
            step=0,
            model_time=timedelta(0),
            metadata={"model": self.name, "soil_levels": self.soil_levels},
        )
        self._initialized = True
        return self.state

    def step(self, state: ModelState, dt: timedelta) -> ModelState:
        new_state = ModelState(
            timestamp=state.timestamp + dt,
            data=state.data.copy(),
            step=state.step + 1,
            model_time=state.model_time + dt,
            metadata=state.metadata.copy(),
        )

        if "skin_temperature" in new_state.data.data_vars:
            st = new_state.data["skin_temperature"].values
            diurnal = 5 * np.sin(2 * np.pi * (state.model_time.total_seconds() % 86400) / 86400)
            new_state.data["skin_temperature"].values = st + diurnal * dt.total_seconds() / 86400

        if "soil_moisture" in new_state.data.data_vars and "evapotranspiration" in new_state.data.data_vars:
            sm = new_state.data["soil_moisture"].values
            et = new_state.data["evapotranspiration"].values
            dt_days = dt.total_seconds() / 86400
            if sm.ndim >= 3 and et.ndim == 2:
                new_state.data["soil_moisture"].values = np.clip(
                    sm - et[:, :, np.newaxis] * dt_days / sm.shape[2], 0.0, 1.0
                )

        self.state = new_state
        return new_state


class SeaIceModel(ModelComponent):
    STANDARD_VARIABLES = [
        "sea_ice_concentration", "sea_ice_thickness",
        "sea_ice_temperature", "snow_on_ice",
        "ice_u_velocity", "ice_v_velocity",
        "melt_pond_fraction", "ice_damage",
    ]

    def __init__(
        self,
        name: str = "sea_ice",
        ice_categories: int = 5,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.ice_categories = ice_categories
        self.ice_dynamics = "c_grid"
        self.ice_thickness_categories = np.linspace(0.5, 5.0, ice_categories)

    def output_variables(self) -> List[str]:
        return self.STANDARD_VARIABLES

    def initialize(self, initial_state: Optional[ModelState] = None, **kwargs) -> ModelState:
        if initial_state is not None:
            self.state = initial_state
            self._initialized = True
            return self.state

        lat = np.arange(-90, 90 + self.resolution_deg, self.resolution_deg)
        lon = np.arange(0, 360, self.resolution_deg)

        lat = lat[:int(180 / self.resolution_deg) + 1]
        lon = lon[:int(360 / self.resolution_deg)]

        n_lat = len(lat)
        n_lon = len(lon)

        ds = xr.Dataset()
        ds["lat"] = ("lat", lat)
        ds["lon"] = ("lon", lon)

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        north_pole_mask = (lat_grid > 60).astype(float)
        south_pole_mask = (lat_grid < -60).astype(float)
        sea_mask = np.ones_like(lat_grid)
        try:
            sea_mask[int(90 / self.resolution_deg):, int(160 / self.resolution_deg):int(290 / self.resolution_deg)] = 1
        except Exception:
            pass

        ice_conc_nh = np.clip(1.2 * (lat_grid - 60) / 30 * np.exp(-(lon_grid - 0) ** 2 / 10000), 0, 1)
        ice_conc_sh = np.clip(1.2 * (-lat_grid - 60) / 30 * np.exp(-(lon_grid - 180) ** 2 / 10000), 0, 1)
        total_ice = (ice_conc_nh * north_pole_mask + ice_conc_sh * south_pole_mask) * sea_mask
        ds["sea_ice_concentration"] = (["lat", "lon"], np.clip(total_ice, 0, 1))

        ice_thick = 0.5 + 3.0 * total_ice * (np.abs(lat_grid) - 60) / 30
        ds["sea_ice_thickness"] = (["lat", "lon"], np.clip(ice_thick * sea_mask, 0, 10))

        ice_temp = 250 + 20 * (1 - total_ice)
        ds["sea_ice_temperature"] = (["lat", "lon"], ice_temp * sea_mask)

        ds["snow_on_ice"] = (["lat", "lon"], 0.2 * total_ice * sea_mask)

        ds["ice_u_velocity"] = (["lat", "lon"], np.random.randn(n_lat, n_lon) * 0.01 * sea_mask)
        ds["ice_v_velocity"] = (["lat", "lon"], np.random.randn(n_lat, n_lon) * 0.01 * sea_mask)

        ds["melt_pond_fraction"] = (["lat", "lon"], 0.1 * total_ice * (np.abs(lat_grid) < 80) * sea_mask)
        ds["ice_damage"] = (["lat", "lon"], np.clip(0.5 * (1 - total_ice), 0, 1) * sea_mask)

        start_time = kwargs.get("start_time", datetime(2020, 1, 1))
        self.state = ModelState(
            timestamp=start_time,
            data=ds,
            step=0,
            model_time=timedelta(0),
            metadata={"model": self.name, "ice_categories": self.ice_categories},
        )
        self._initialized = True
        return self.state

    def step(self, state: ModelState, dt: timedelta) -> ModelState:
        dt_days = dt.total_seconds() / 86400
        new_state = ModelState(
            timestamp=state.timestamp + dt,
            data=state.data.copy(),
            step=state.step + 1,
            model_time=state.model_time + dt,
            metadata=state.metadata.copy(),
        )

        if "sea_ice_concentration" in new_state.data.data_vars:
            conc = new_state.data["sea_ice_concentration"].values
            lat_grid = np.meshgrid(new_state.data["lon"].values, new_state.data["lat"].values)[1]
            seasonal = 0.1 * np.sin(2 * np.pi * state.model_time.days / 365.25) * (np.abs(lat_grid) > 60)
            new_state.data["sea_ice_concentration"].values = np.clip(
                conc + seasonal * dt_days * 0.01 + np.random.randn(*conc.shape) * 0.001, 0, 1
            )

        self.state = new_state
        return new_state


class ModelTimeStepper:
    def __init__(self, dt_seconds: int = 300, start_time: datetime = None):
        self.dt = timedelta(seconds=dt_seconds)
        self.current_time = start_time or datetime(2020, 1, 1)
        self.step_count = 0

    def advance(self) -> datetime:
        self.current_time += self.dt
        self.step_count += 1
        return self.current_time

    def set_time(self, time: datetime):
        self.current_time = time

    @property
    def time_since_start(self) -> timedelta:
        return self.step_count * self.dt


class CoupledModel:
    def __init__(
        self,
        atmosphere: Optional[AtmosphericModel] = None,
        ocean: Optional[OceanModel] = None,
        land: Optional[LandModel] = None,
        sea_ice: Optional[SeaIceModel] = None,
        coupling_type: CouplingType = CouplingType.FULLY_COUPLED,
        coupling_interval_seconds: int = 900,
    ):
        self.atmosphere = atmosphere or AtmosphericModel()
        self.ocean = ocean or OceanModel()
        self.land = land or LandModel()
        self.sea_ice = sea_ice or SeaIceModel()
        self.coupling_type = coupling_type

        self.coupling_scheduler = CouplingScheduler(coupling_interval_seconds)
        self._setup_coupling_interfaces()

        self.time_stepper: Optional[ModelTimeStepper] = None
        self.components: Dict[str, ModelComponent] = {
            "atmosphere": self.atmosphere,
            "ocean": self.ocean,
            "land": self.land,
            "sea_ice": self.sea_ice,
        }
        self.states: Dict[str, ModelState] = {}

    def _bilinear_interp(
        self,
        data: np.ndarray,
        src_lat: np.ndarray,
        src_lon: np.ndarray,
        dst_lat: np.ndarray,
        dst_lon: np.ndarray,
    ) -> np.ndarray:
        src_nlat = len(src_lat)
        src_nlon = len(src_lon)
        dst_nlat = len(dst_lat)
        dst_nlon = len(dst_lon)

        if src_nlat == dst_nlat and src_nlon == dst_nlon:
            if np.allclose(src_lat, dst_lat) and np.allclose(src_lon, dst_lon):
                return data.copy()

        result = np.zeros((dst_nlat, dst_nlon), dtype=data.dtype)

        for i in range(dst_nlat):
            for j in range(dst_nlon):
                dlat = dst_lat[i]
                dlon = dst_lon[j]

                lat_idx = np.clip(np.searchsorted(src_lat, dlat) - 1, 0, src_nlat - 2)
                lon_idx = np.clip(np.searchsorted(src_lon, dlon) - 1, 0, src_nlon - 2)

                lat0, lat1 = src_lat[lat_idx], src_lat[lat_idx + 1]
                lon0, lon1 = src_lon[lon_idx], src_lon[lon_idx + 1]

                if lat1 == lat0:
                    lat_frac = 0.0
                else:
                    lat_frac = (dlat - lat0) / (lat1 - lat0)
                if lon1 == lon0:
                    lon_frac = 0.0
                else:
                    lon_frac = (dlon - lon0) / (lon1 - lon0)

                v00 = data[lat_idx, lon_idx]
                v01 = data[lat_idx, lon_idx + 1]
                v10 = data[lat_idx + 1, lon_idx]
                v11 = data[lat_idx + 1, lon_idx + 1]

                v_top = v00 * (1 - lon_frac) + v01 * lon_frac
                v_bot = v10 * (1 - lon_frac) + v11 * lon_frac
                result[i, j] = v_top * (1 - lat_frac) + v_bot * lat_frac

        return result

    def _remap_variable(
        self,
        data: np.ndarray,
        source_state: ModelState,
        target_state: ModelState,
    ) -> np.ndarray:
        if data.ndim < 2:
            return data

        src_lat = source_state.data["lat"].values if "lat" in source_state.data.coords else None
        src_lon = source_state.data["lon"].values if "lon" in source_state.data.coords else None
        dst_lat = target_state.data["lat"].values if "lat" in target_state.data.coords else None
        dst_lon = target_state.data["lon"].values if "lon" in target_state.data.coords else None

        if src_lat is None or src_lon is None or dst_lat is None or dst_lon is None:
            return data

        try:
            if data.ndim == 2:
                return self._bilinear_interp(data, src_lat, src_lon, dst_lat, dst_lon)
            elif data.ndim >= 3:
                result = np.zeros(
                    (len(dst_lat), len(dst_lon)) + data.shape[2:],
                    dtype=data.dtype
                )
                for k in range(data.shape[2]):
                    if data.ndim == 3:
                        result[:, :, k] = self._bilinear_interp(
                            data[:, :, k], src_lat, src_lon, dst_lat, dst_lon
                        )
                    else:
                        result[:, :, k, ...] = self._bilinear_interp(
                            data[:, :, k, ...], src_lat, src_lon, dst_lat, dst_lon
                        )
                return result
        except Exception as e:
            logger.warning(f"Remap failed, returning original data: {e}")
            return data

        return data

    def _setup_coupling_interfaces(self):
        if self.coupling_type in [CouplingType.TWO_WAY, CouplingType.FULLY_COUPLED]:
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="atmosphere", target_model="ocean",
                variable_maps=[
                    CouplingVariableMap(
                        source_var="temperature", target_var="sea_surface_temperature",
                        method=CouplingMethod.RELAXATION, relaxation_time_seconds=3*86400,
                        source_level=1000,
                    ),
                    CouplingVariableMap(
                        source_var="u_wind", target_var="u_current",
                        method=CouplingMethod.FLUX, scale_factor=0.001,
                    ),
                    CouplingVariableMap(
                        source_var="v_wind", target_var="v_current",
                        method=CouplingMethod.FLUX, scale_factor=0.001,
                    ),
                    CouplingVariableMap(
                        source_var="sensible_heat_flux", target_var="water_temperature",
                        method=CouplingMethod.FLUX, scale_factor=1.0 / (1025 * 4186 * 50),
                    ),
                    CouplingVariableMap(
                        source_var="precipitation", target_var="salinity",
                        method=CouplingMethod.FLUX, scale_factor=-0.01,
                    ),
                ]
            ))
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="ocean", target_model="atmosphere",
                variable_maps=[
                    CouplingVariableMap(
                        source_var="sea_surface_temperature", target_var="temperature",
                        method=CouplingMethod.RELAXATION, relaxation_time_seconds=86400,
                        target_level=1000,
                    ),
                ]
            ))
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="atmosphere", target_model="land",
                variable_maps=[
                    CouplingVariableMap(
                        source_var="temperature", target_var="skin_temperature",
                        method=CouplingMethod.RELAXATION, relaxation_time_seconds=86400,
                        source_level=1000,
                    ),
                    CouplingVariableMap(
                        source_var="precipitation", target_var="soil_moisture",
                        method=CouplingMethod.FLUX, scale_factor=0.001,
                    ),
                    CouplingVariableMap(
                        source_var="shortwave_radiation", target_var="skin_temperature",
                        method=CouplingMethod.FLUX, scale_factor=1e-6,
                    ),
                ]
            ))
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="land", target_model="atmosphere",
                variable_maps=[
                    CouplingVariableMap(
                        source_var="skin_temperature", target_var="temperature",
                        method=CouplingMethod.RELAXATION, relaxation_time_seconds=2*86400,
                        target_level=1000,
                    ),
                    CouplingVariableMap(
                        source_var="evapotranspiration", target_var="specific_humidity",
                        method=CouplingMethod.FLUX, scale_factor=0.001,
                    ),
                    CouplingVariableMap(
                        source_var="albedo", target_var="shortwave_radiation",
                        method=CouplingMethod.DIRECT, scale_factor=0.5,
                    ),
                ]
            ))
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="ocean", target_model="sea_ice",
                variable_maps=[
                    CouplingVariableMap(
                        source_var="water_temperature", target_var="sea_ice_temperature",
                        method=CouplingMethod.RELAXATION, relaxation_time_seconds=5*86400,
                    ),
                    CouplingVariableMap(
                        source_var="u_current", target_var="ice_u_velocity",
                        method=CouplingMethod.DIRECT, scale_factor=0.3,
                    ),
                    CouplingVariableMap(
                        source_var="v_current", target_var="ice_v_velocity",
                        method=CouplingMethod.DIRECT, scale_factor=0.3,
                    ),
                ]
            ))
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="sea_ice", target_model="ocean",
                variable_maps=[
                    CouplingVariableMap(
                        source_var="sea_ice_concentration", target_var="sea_surface_temperature",
                        method=CouplingMethod.FLUX, scale_factor=-5.0,
                    ),
                ]
            ))
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="atmosphere", target_model="sea_ice",
                variable_maps=[
                    CouplingVariableMap(
                        source_var="temperature", target_var="sea_ice_temperature",
                        method=CouplingMethod.RELAXATION, relaxation_time_seconds=2*86400,
                        source_level=1000,
                    ),
                    CouplingVariableMap(
                        source_var="u_wind", target_var="ice_u_velocity",
                        method=CouplingMethod.FLUX, scale_factor=0.0001,
                    ),
                    CouplingVariableMap(
                        source_var="v_wind", target_var="ice_v_velocity",
                        method=CouplingMethod.FLUX, scale_factor=0.0001,
                    ),
                ]
            ))
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="sea_ice", target_model="atmosphere",
                variable_maps=[
                    CouplingVariableMap(
                        source_var="sea_ice_concentration", target_var="albedo",
                        method=CouplingMethod.DIRECT, scale_factor=0.5,
                    ),
                ]
            ))
        elif self.coupling_type == CouplingType.ONE_WAY:
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="atmosphere", target_model="ocean",
                variable_maps=[
                    CouplingVariableMap(source_var="temperature", target_var="sea_surface_temperature",
                                       method=CouplingMethod.RELAXATION, relaxation_time_seconds=86400),
                ]
            ))
            self.coupling_scheduler.add_interface(CouplingInterface(
                source_model="atmosphere", target_model="land",
                variable_maps=[
                    CouplingVariableMap(source_var="precipitation", target_var="soil_moisture",
                                       method=CouplingMethod.FLUX, scale_factor=0.001),
                ]
            ))

    def initialize(self, start_time: Optional[datetime] = None, **kwargs) -> Dict[str, ModelState]:
        st = start_time or datetime(2020, 1, 1)
        self.time_stepper = ModelTimeStepper(
            dt_seconds=self.atmosphere.time_step.total_seconds(),
            start_time=st
        )

        self.states["atmosphere"] = self.atmosphere.initialize(start_time=st)
        self.states["ocean"] = self.ocean.initialize(start_time=st)
        self.states["land"] = self.land.initialize(start_time=st)
        self.states["sea_ice"] = self.sea_ice.initialize(start_time=st)

        if self.coupling_type != CouplingType.UNCOUPLED:
            self.coupling_scheduler.execute_coupling(
                self.states, self._remap_data, dt=self.atmosphere.time_step
            )

        logger.info(f"Coupled model initialized at {st} with {self.coupling_type.value} coupling")
        return self.states

    def _remap_data(
        self,
        source_model: str,
        target_model: str,
        data: Dict[str, np.ndarray],
        source_state: ModelState,
        target_state: ModelState,
    ) -> Dict[str, np.ndarray]:
        remapped = {}
        for var_name, var_data in data.items():
            remapped[var_name] = self._remap_variable(var_data, source_state, target_state)
        return remapped

    def step(self) -> Dict[str, ModelState]:
        if not self.states:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        self.time_stepper.advance()
        dt = self.atmosphere.time_step

        for name, component in self.components.items():
            self.states[name] = component.step(self.states[name], component.time_step)

        if self.coupling_type != CouplingType.UNCOUPLED:
            if self.coupling_scheduler.needs_coupling(self.time_stepper.current_time):
                self.states = self.coupling_scheduler.execute_coupling(
                    self.states, self._remap_data, dt=dt
                )

        return self.states

    def run(self, duration: timedelta) -> Dict[str, List[ModelState]]:
        if not self.states:
            self.initialize()

        outputs = {name: [] for name in self.components}
        end_time = self.time_stepper.current_time + duration
        output_interval = max(1, int(duration / self.time_stepper.dt / 50))

        step_idx = 0
        while self.time_stepper.current_time < end_time:
            self.step()
            if step_idx % output_interval == 0:
                for name in self.components:
                    outputs[name].append(self.states[name])
            step_idx += 1

        return outputs

    def get_state(self, component_name: str) -> Optional[ModelState]:
        return self.states.get(component_name)

    def get_combined_state(self) -> xr.Dataset:
        combined = xr.Dataset()
        for name, state in self.states.items():
            for var in state.data.data_vars:
                combined[f"{name}_{var}"] = state.data[var]
        combined.attrs["timestamp"] = str(self.time_stepper.current_time)
        combined.attrs["coupling_type"] = self.coupling_type.value
        return combined
