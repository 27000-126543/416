"""Debug QC test - detailed."""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import xarray as xr
from climate_platform.data_stream.cleaning import DataCleaner

cleaner = DataCleaner(quality_threshold=0.85, auto_clean=False)

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

print("=== Before QC (auto_clean=False) ===")
result = cleaner.run_qc(ds2)
print(f"Failures count: {len(result.failures)}")
for f in result.failures:
    print(f"  - {f.variable}: {f.check_name}: {f.num_failures} pts")
print(f"Cleaned: {result.cleaned}")
print(f"Overall quality: {result.overall_quality}")
print(f"Variable quality: {result.variable_quality}")

print("\n=== Calling clean() ===")
cleaned_result = cleaner.clean(result)
print(f"Failures count after clean: {len(cleaned_result.failures)}")
for f in cleaned_result.failures:
    print(f"  - {f.variable}: {f.check_name}: {f.num_failures} pts")
print(f"Cleaned: {cleaned_result.cleaned}")

cleaned_temp = cleaned_result.dataset["temperature"].values
print(f"Original temp min: {temp2.min()}, max: {temp2.max()}")
print(f"Cleaned temp min: {cleaned_temp.min()}, max: {cleaned_temp.max()}")
print(f"NaN count: {int(np.sum(np.isnan(cleaned_temp)))}")
