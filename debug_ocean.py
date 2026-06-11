"""Debug ocean init."""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from climate_platform.model import OceanModel

ocean = OceanModel(name="ocean", vertical_levels=5, resolution_deg=5.0, time_step_seconds=10800)
state = ocean.initialize()
print(f"Variables: {list(state.data.data_vars)}")
sst = state.data["sea_surface_temperature"].values
print(f"SST shape: {sst.shape}")
print(f"SST min: {sst.min()}")
print(f"SST max: {sst.max()}")
print(f"SST mean: {sst.mean()}")

# Let's also check the lat range
print(f"Lat range: {state.data.lat.values.min()} to {state.data.lat.values.max()}")
print(f"Lon range: {state.data.lon.values.min()} to {state.data.lon.values.max()}")

# Manual calculation
lat = np.arange(-80, 80 + 5.0, 5.0)
lat = lat[:int(160 / 5.0) + 1]
lon = np.arange(0, 360, 5.0)
lon = lon[:int(360 / 5.0)]
print(f"\nManual lat: {len(lat)} points, {lat.min()} to {lat.max()}")
print(f"Manual lon: {len(lon)} points")

lon_grid, lat_grid = np.meshgrid(lon, lat)
sst_manual = 30 * np.cos(np.radians(lat_grid)) - 2 + 273.15
print(f"Manual SST mean: {sst_manual.mean()}")
print(f"Manual SST min: {sst_manual.min()}")
print(f"Manual SST max: {sst_manual.max()}")
