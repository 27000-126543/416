"""Debug QC test."""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import xarray as xr
from climate_platform.data_stream import DataCleaner

cleaner = DataCleaner(quality_threshold=0.85, auto_clean=True)

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

print(f"temperature variable exists in ds: {'temperature' in ds2.data_vars}")
print(f"temp min: {temp2.min()}, max: {temp2.max()}")
print(f"PHYSICAL_RANGES for temperature: {cleaner.PHYSICAL_RANGES.get('temperature')}")
print(f"enable_range_check: {cleaner.enable_range_check}")
print(f"enable_gradient_check: {cleaner.enable_gradient_check}")

result = cleaner.run_qc(ds2)
print(f"\nFailures count: {len(result.failures)}")
for f in result.failures:
    print(f"  - {f.variable}: {f.check_name}: {f.num_failures} pts")
print(f"Cleaned: {result.cleaned}")
print(f"Overall quality: {result.overall_quality}")

cleaned_temp = result.dataset["temperature"].values
print(f"Cleaned temp min: {cleaned_temp.min()}, max: {cleaned_temp.max()}")
print(f"NaN count: {int(np.sum(np.isnan(cleaned_temp)))}")
