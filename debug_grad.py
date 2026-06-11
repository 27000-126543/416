"""Debug gradient check."""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import xarray as xr
from climate_platform.data_stream.cleaning import DataCleaner

lat2 = np.linspace(-90, 90, 37)
lon2 = np.linspace(0, 359, 72)
lon_grid2, lat_grid2 = np.meshgrid(lon2, lat2)
temp2 = 288 - 30 * np.abs(np.sin(np.radians(lat_grid2)))
temp2[10, 20] = 500.0
temp2[20, 40] = 150.0
ds2 = xr.Dataset(
    {"temperature": (["lat", "lon"], temp2)},
    coords={"lat": lat2, "lon": lon2}
)

cleaner = DataCleaner()

print(f"Dataset dims: {ds2.dims}")
print(f"Lat in coords: {'lat' in ds2.coords}")
print(f"Lon in coords: {'lon' in ds2.coords}")
print(f"Lat axis: {ds2.dims.index('lat') if 'lat' in ds2.dims else None}")
print(f"Lon axis: {ds2.dims.index('lon') if 'lon' in ds2.dims else None}")

data = temp2
lat_axis = 0
lon_axis = 1
grad_lat = np.abs(np.gradient(data, axis=lat_axis))
grad_lon = np.abs(np.gradient(data, axis=lon_axis))
total_grad = np.sqrt(grad_lat ** 2 + grad_lon ** 2)

print(f"\nGradient stats:")
print(f"  grad_lat max: {grad_lat.max()}")
print(f"  grad_lon max: {grad_lon.max()}")
print(f"  total_grad max: {total_grad.max()}")
print(f"  max_gradient for temp: 15.0")
print(f"  Points above 15: {int(np.sum(total_grad > 15))}")

# Check the spike point
print(f"\nAt spike point (10,20):")
print(f"  value: {temp2[10, 20]}")
print(f"  grad_lat: {grad_lat[10, 20]}")
print(f"  grad_lon: {grad_lon[10, 20]}")
print(f"  total_grad: {total_grad[10, 20]}")

print(f"\nAt spike point (20,40):")
print(f"  value: {temp2[20, 40]}")
print(f"  grad_lat: {grad_lat[20, 40]}")
print(f"  grad_lon: {grad_lon[20, 40]}")
print(f"  total_grad: {total_grad[20, 40]}")
