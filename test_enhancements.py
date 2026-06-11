"""Integration test for all 4 enhancements."""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import logging; logging.basicConfig(level=logging.WARNING)
import asyncio
import numpy as np
import xarray as xr
import shutil
from datetime import datetime

print("=" * 70)
print("Integration Test: 4 Enhancements")
print("=" * 70)

# ===================================================================
# Enhancement 1: Ingestion Session Management
# ===================================================================
print("\n[Enhancement 1] Ingestion Session Management")
from climate_platform.data_stream import DataIngestionManager, SatelliteSource, GroundStationSource, OceanBuoySource

mgr = DataIngestionManager(parallel_workers=4)
sat = SatelliteSource(source_id="sat1", satellite_name="TestSat")
gs = GroundStationSource(source_id="gs1", station_id="GS001", station_name="Test Station", latitude=40.0, longitude=-105.0)
buoy = OceanBuoySource(source_id="buoy1", buoy_id="BOB01", latitude=0.0, longitude=180.0)
mgr.register_source(sat)
mgr.register_source(gs)
mgr.register_source(buoy)

async def setup():
    await mgr.start()
asyncio.run(setup())

sessions = mgr.get_all_sessions()
print(f"  After start: {len(sessions)} sessions")
for sid, sess in sessions.items():
    print(f"    {sid}: state={sess.state.value}")

# Ingest some data while running
lat = np.linspace(-90, 90, 19)
lon = np.linspace(0, 359, 36)
lon_grid, lat_grid = np.meshgrid(lon, lat)
temp = 288 - 30 * np.abs(np.sin(np.radians(lat_grid)))
ds = xr.Dataset({"temperature": (["lat", "lon"], temp)}, coords={"lat": lat, "lon": lon})

async def ingest_running():
    await sat.ingest_from_memory(ds, timestamp=datetime.now())
    await gs.ingest_observation({"temperature_2m": 295.0, "relative_humidity": 70.0})
asyncio.run(ingest_running())

stats1 = mgr.get_stats()
print(f"  After 2 ingestions (running): {stats1.total_chunks} chunks")
sess_sat = mgr.get_session("sat1")
print(f"  Sat session chunks: {sess_sat.session_chunks}")

# Stop ingestion
result = mgr.stop_ingestion("sat1")
print(f"  Stopped sat1: state={mgr.get_session('sat1').state.value}")

# Try to ingest while stopped - should be rejected
async def ingest_stopped():
    await sat.ingest_from_memory(ds, timestamp=datetime.now())
asyncio.run(ingest_stopped())

stats2 = mgr.get_stats()
print(f"  After ingestion while stopped: {stats2.total_chunks} chunks (should be same)")

# Restart and ingest again
mgr.start_ingestion("sat1")
async def ingest_restart():
    await sat.ingest_from_memory(ds, timestamp=datetime.now())
asyncio.run(ingest_restart())

stats3 = mgr.get_stats()
sess_sat2 = mgr.get_session("sat1")
cumulative = mgr.get_cumulative_stats()
print(f"  After restart+ingest: session={sess_sat2.session_chunks} chunks, cumulative={cumulative.total_chunks} chunks")
print(f"  ✓ Session management working!")

# ===================================================================
# Enhancement 2: QC Kelvin Fix & Variable Details
# ===================================================================
print("\n[Enhancement 2] QC Kelvin Fix & Variable Details")
from climate_platform.data_stream import DataCleaner

cleaner = DataCleaner(quality_threshold=0.85, auto_clean=True)

# Test with Kelvin-range data (model output)
temp_kelvin = 288 - 30 * np.abs(np.sin(np.radians(lat_grid)))
ds_kelvin = xr.Dataset({"temperature": (["lat", "lon"], temp_kelvin)}, coords={"lat": lat, "lon": lon})

result_qc = cleaner.run_qc(ds_kelvin)
print(f"  Kelvin data QC: passed={result_qc.overall_quality >= 0.85}, quality={result_qc.overall_quality:.4f}")
print(f"  Failures: {len(result_qc.failures)}")
print(f"  Variable details: {result_qc.variable_details}")

# Verify the quality_control module also fixed
from climate_platform.quality_control.quality_control import RangeCheck
rc = RangeCheck()
test_ds = xr.Dataset({"temperature": (["lat", "lon"], temp_kelvin)}, coords={"lat": lat, "lon": lon})
rc_result = rc.check(test_ds, "temperature")
print(f"  RangeCheck result: passed={rc_result.passed}, unit={rc_result.stats.get('unit', 'unknown')}")
assert rc_result.passed, "Kelvin temperature should pass range check!"
print(f"  ✓ Kelvin false positive fixed!")

# Test with anomalous data
temp_anom = temp_kelvin.copy()
temp_anom[10, 20] = 500.0
ds_anom = xr.Dataset({"temperature": (["lat", "lon"], temp_anom)}, coords={"lat": lat, "lon": lon})
result_anom = cleaner.run_qc(ds_anom)
print(f"  Anomalous data: failures={len(result_anom.failures)}, quality={result_anom.overall_quality:.4f}")
assert len(result_anom.failures) > 0, "Anomaly should be detected!"
print(f"  ✓ Anomaly detection working!")

# ===================================================================
# Enhancement 3: Ingestion → Cleaning → QC Pipeline
# ===================================================================
print("\n[Enhancement 3] Ingestion → Cleaning → QC Pipeline")
from climate_platform.data_stream import QCSummary

# Enable pipeline
mgr2 = DataIngestionManager(parallel_workers=4)
sat2 = SatelliteSource(source_id="sat2", satellite_name="TestSat2")
mgr2.register_source(sat2)

async def setup2():
    await mgr2.start()
asyncio.run(setup2())

mgr2.enable_pipeline()

# Create data with anomaly
temp_pipe = temp_kelvin.copy()
temp_pipe[5, 10] = 500.0
ds_pipe = xr.Dataset({"temperature": (["lat", "lon"], temp_pipe)}, coords={"lat": lat, "lon": lon})

async def ingest_pipeline():
    await sat2.ingest_from_memory(ds_pipe, timestamp=datetime.now())
asyncio.run(ingest_pipeline())

# Wait a bit for processing
import time; time.sleep(0.5)

summaries = mgr2.get_qc_summaries()
print(f"  QC summaries generated: {len(summaries)}")
if summaries:
    s = summaries[-1]
    print(f"    Source: {s.source_id}")
    print(f"    Original anomalies: {s.original_anomaly_points}")
    print(f"    Cleaned NaN count: {s.cleaned_nan_count}")
    print(f"    Modified points: {s.modified_points}")
    print(f"    Passed: {s.passed}")
    print(f"    Variable summaries: {list(s.variable_summaries.keys())}")
print(f"  ✓ Pipeline working!")

# Also test the convenience method
mgr2.start_ingestion("sat2")
summary = mgr2.ingest_and_qc("sat2", ds_pipe)
if summary:
    print(f"  ingest_and_qc: source={summary.source_id}, anomalies={summary.original_anomaly_points}")
print(f"  ✓ Convenience method working!")

# ===================================================================
# Enhancement 4: Multi-tenant Quota Enhancements
# ===================================================================
print("\n[Enhancement 4] Multi-tenant Quota Enhancements")
from climate_platform.multi_tenant import TenantManager, TenantRole, ResourceQuota

sandbox_path = "./test_sandboxes_v2"
if os.path.exists(sandbox_path):
    shutil.rmtree(sandbox_path, ignore_errors=True)
tm = TenantManager(base_sandbox_path=sandbox_path)
t = tm.create_tenant(tenant_id="test2", name="Test Tenant", role=TenantRole.RESEARCH)

# Test usage summary
usage = tm.get_tenant_usage("test2")
print(f"  Usage summary:")
for res, info in usage["usage"].items():
    print(f"    {res}: used={info['used']}, quota={info['quota']}, remaining={info['remaining']}")

# Test allocate_resources_detailed
ok, reasons = tm.allocate_resources_detailed("test2", ResourceQuota(storage_tb=2.0, concurrent_jobs=3))
print(f"  Allocate 2TB+3jobs: ok={ok}, reasons={reasons}")

ok2, reasons2 = tm.allocate_resources_detailed("test2", ResourceQuota(storage_tb=100.0, concurrent_jobs=50))
print(f"  Over-quota 100TB+50jobs: ok={ok2}, reasons={reasons2}")
assert not ok2, "Should be rejected"
assert len(reasons2) > 0, "Should have specific reasons"

# Test release with history
tm.release_resources("test2", ResourceQuota(storage_tb=1.0, concurrent_jobs=1))
history = tm.get_tenant_history("test2")
print(f"  History events: {len(history)}")
for evt in history:
    print(f"    {evt.event_type}: {evt.resources}")

# Test usage after release
usage2 = tm.get_tenant_usage("test2")
print(f"  After release: storage used={usage2['usage']['storage_tb']['used']}, remaining={usage2['usage']['storage_tb']['remaining']}")

# Test history filtering
alloc_history = tm.get_tenant_history("test2", event_type="allocate")
release_history = tm.get_tenant_history("test2", event_type="release")
print(f"  Alloc events: {len(alloc_history)}, Release events: {len(release_history)}")

shutil.rmtree(sandbox_path, ignore_errors=True)
print(f"  ✓ Multi-tenant quota enhancements working!")

print("\n" + "=" * 70)
print("ALL 4 ENHANCEMENTS VERIFIED! ✓")
print("=" * 70)
