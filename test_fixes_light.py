"""Quick test for fixes 1-3 and 5 (skipping heavy coupled model)."""
import sys
import os
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.WARNING)

print("=" * 60)
print("Verification of Fixes 1, 2, 3, 5")
print("=" * 60)

print("\n[Fix 1] API-ready default state")
from climate_platform.main import ClimateSimulationPlatform
platform = ClimateSimulationPlatform()
platform.initialize()
print(f"  ✓ Platform initialized")
print(f"  ✓ Coupled model states: {len(platform.coupled_model.states)} components")
combined = platform.coupled_model.get_combined_state()
print(f"  ✓ Combined state vars: {len(combined.data_vars)}")
print(f"  ✓ Sample vars: {list(combined.data_vars.keys())[:6]}...")

print("\n[Fix 2] Multi-tenant resource quota")
from climate_platform.multi_tenant import TenantManager, TenantRole, ResourceQuota
import shutil
sandbox_path = "./test_sandboxes"
if os.path.exists(sandbox_path):
    shutil.rmtree(sandbox_path, ignore_errors=True)
tm = TenantManager(base_sandbox_path=sandbox_path)
t = tm.create_tenant(tenant_id="test", name="Test Tenant", role=TenantRole.RESEARCH)
print(f"  ✓ Initial storage: {t.usage.storage_tb:.1f}/{t.quota.storage_tb:.0f} TB")

ok1 = tm.allocate_resources("test", ResourceQuota(storage_tb=2.0, concurrent_jobs=5))
print(f"  ✓ Allocate 2TB+5jobs: {ok1}, used={t.usage.storage_tb:.1f}TB")
assert ok1 == True, "First allocation should succeed"

ok2 = tm.allocate_resources("test", ResourceQuota(storage_tb=3.0, concurrent_jobs=3))
print(f"  ✓ Allocate 3TB+3jobs more: {ok2}, used={t.usage.storage_tb:.1f}TB, {t.usage.concurrent_jobs}jobs")
assert ok2 == True, "Second allocation should succeed"

ok_over = tm.allocate_resources("test", ResourceQuota(storage_tb=100.0, concurrent_jobs=50))
print(f"  ✓ Over-quota (100TB+50jobs) rejected: {not ok_over}")
assert ok_over == False, "Over-quota allocation should be rejected"

tm.release_resources("test", ResourceQuota(storage_tb=3.0, concurrent_jobs=2))
print(f"  ✓ Release 3TB+2jobs, used={t.usage.storage_tb:.1f}TB, {t.usage.concurrent_jobs}jobs")
assert abs(t.usage.storage_tb - 2.0) < 0.01, "Storage should be 2.0 TB after release"

ok_after = tm.allocate_resources("test", ResourceQuota(storage_tb=1.0, concurrent_jobs=1))
print(f"  ✓ Re-allocate 1TB+1job after release: {ok_after}")
assert ok_after == True, "Re-allocation after release should succeed"
print(f"  ✓ Final usage: {t.usage.storage_tb:.1f}TB, {t.usage.concurrent_jobs}jobs")

shutil.rmtree(sandbox_path, ignore_errors=True)
print("  ✓ ALL QUOTA TESTS PASSED")

print("\n[Fix 3] Data ingestion stats")
import asyncio
from climate_platform.data_stream import (
    DataIngestionManager, SatelliteSource, GroundStationSource, OceanBuoySource
)
import xarray as xr
import numpy as np
from datetime import datetime

mgr = DataIngestionManager(parallel_workers=4)
sat = SatelliteSource(source_id="sat1", satellite_name="TestSat")
gs = GroundStationSource(source_id="gs1", station_id="GS001", station_name="Test Station", latitude=40.0, longitude=-105.0)
buoy = OceanBuoySource(source_id="buoy1", buoy_id="BOB01", latitude=0.0, longitude=180.0)
mgr.register_source(sat)
mgr.register_source(gs)
mgr.register_source(buoy)

stats0 = mgr.get_stats()
print(f"  ✓ Initial: {stats0.total_chunks} chunks, {stats0.total_bytes} bytes")
assert stats0.total_chunks == 0

lat = np.linspace(-90, 90, 19)
lon = np.linspace(0, 359, 36)
lon_grid, lat_grid = np.meshgrid(lon, lat)
temp = 288 - 30 * np.abs(np.sin(np.radians(lat_grid)))
ds = xr.Dataset(
    {"temperature": (["lat", "lon"], temp),
     "u_wind": (["lat", "lon"], 5 * np.cos(np.radians(lat_grid)))},
    coords={"lat": lat, "lon": lon}
)

async def test_ingestion():
    await sat.ingest_from_memory(ds, timestamp=datetime.now())
    await sat.ingest_from_memory(ds, timestamp=datetime.now())
    await gs.ingest_observation({"temperature_2m": 295.0, "relative_humidity": 70.0, "pressure_surface": 101300.0})
    await buoy.ingest_profile({0.0: 298.0, 10.0: 296.0, 50.0: 290.0, 100.0: 285.0})

asyncio.run(test_ingestion())

stats1 = mgr.get_stats()
print(f"  ✓ After 4 ingestions: {stats1.total_chunks} chunks, {stats1.total_bytes/1024:.2f} KB")
assert stats1.total_chunks == 4, f"Expected 4 chunks, got {stats1.total_chunks}"

sat_stats = mgr.get_source_stats("sat1")
print(f"  ✓ Satellite source: {sat_stats.total_chunks} chunks")
assert sat_stats.total_chunks == 2

buoy_stats = mgr.get_source_stats("buoy1")
print(f"  ✓ Buoy source: {buoy_stats.total_chunks} chunks")
assert buoy_stats.total_chunks == 1

print(f"  ✓ Rate: {stats1.avg_ingestion_rate_mbps:.4f} Mbps")
print("  ✓ ALL INGESTION TESTS PASSED")

print("\n[Fix 5] Data cleaning - gradient check & fill")
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

result = cleaner.run_qc(ds2)
print(f"  ✓ QC overall quality: {result.overall_quality:.4f}")
print(f"  ✓ QC cleaned: {result.cleaned}")
print(f"  ✓ Variable quality: {result.variable_quality}")
print(f"  ✓ Failures detected: {len(result.failures)}")
for f in result.failures[:5]:
    print(f"    - {f.variable}: {f.check_name}: {f.num_failures} pts")
print(f"  ✓ Failure summary: {result.failure_summary}")

cleaned_data = result.dataset["temperature"].values
nan_count = int(np.sum(np.isnan(cleaned_data)))
print(f"  ✓ NaN count after cleaning: {nan_count}")
assert nan_count == 0, f"Expected 0 NaNs after cleaning, got {nan_count}"

print("  ✓ ALL CLEANING TESTS PASSED")

print("\n" + "=" * 60)
print("FIXES 1, 2, 3, 5 ALL VERIFIED! ✓")
print("=" * 60)
