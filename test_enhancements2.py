"""Integration test for all 4 enhancements (round 2)."""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import logging; logging.basicConfig(level=logging.WARNING)
import asyncio
import shutil
import numpy as np
import xarray as xr
from datetime import datetime

print("=" * 70)
print("Integration Test: 4 Enhancements (Round 2)")
print("=" * 70)

lat = np.linspace(-90, 90, 19)
lon = np.linspace(0, 359, 36)
lon_grid, lat_grid = np.meshgrid(lon, lat)
temp = 288 - 30 * np.abs(np.sin(np.radians(lat_grid)))

# ===================================================================
# Enhancement 1: Ingestion Dashboard + Rejection Tracking
# ===================================================================
print("\n[Enhancement 1] Ingestion Dashboard & Rejection Tracking")
from climate_platform.data_stream import DataIngestionManager, SatelliteSource

mgr = DataIngestionManager(parallel_workers=4)
sat = SatelliteSource(source_id="sat_dash", satellite_name="DashSat")
mgr.register_source(sat)

async def setup1():
    await mgr.start()
asyncio.run(setup1())

ds = xr.Dataset({"temperature": (["lat", "lon"], temp)}, coords={"lat": lat, "lon": lon})

async def ingest1():
    await sat.ingest_from_memory(ds, timestamp=datetime.now())
asyncio.run(ingest1())

dashboard = mgr.get_source_dashboard("sat_dash")
assert dashboard is not None, "Dashboard should exist"
print(f"  Dashboard state: {dashboard['state']}")
print(f"  Session chunks: {dashboard['current_session']['session_chunks']}")
print(f"  Cumulative chunks: {dashboard['cumulative']['chunks']}")
print(f"  Last data time: {dashboard['current_session']['last_data_time']}")

mgr.stop_ingestion("sat_dash", reason="maintenance")
dashboard2 = mgr.get_source_dashboard("sat_dash")
print(f"  After stop: state={dashboard2['state']}, reason={dashboard2['status']['stop_reason']}")

async def ingest_stopped():
    await sat.ingest_from_memory(ds, timestamp=datetime.now())
asyncio.run(ingest_stopped())

dashboard3 = mgr.get_source_dashboard("sat_dash")
print(f"  Rejected chunks: {dashboard3['rejection']['rejected_chunks']}")
print(f"  Last rejected time: {dashboard3['rejection']['last_rejected_time']}")
assert dashboard3['rejection']['rejected_chunks'] == 1, "Should have 1 rejection"

mgr.start_ingestion("sat_dash")
async def ingest_restart():
    await sat.ingest_from_memory(ds, timestamp=datetime.now())
asyncio.run(ingest_restart())

dashboard4 = mgr.get_source_dashboard("sat_dash")
print(f"  After restart: session_chunks={dashboard4['current_session']['session_chunks']}, cumulative={dashboard4['cumulative']['chunks']}")
assert dashboard4['current_session']['session_chunks'] == 1, "New session should have 1 chunk"
assert dashboard4['cumulative']['chunks'] == 2, "Cumulative should be 2"

all_dash = mgr.get_all_dashboards()
print(f"  All dashboards: {list(all_dash.keys())}")
print("  ✓ Dashboard & rejection tracking working!")

# ===================================================================
# Enhancement 2: QC Validation History & Variable Details
# ===================================================================
print("\n[Enhancement 2] QC Validation History & Variable Details")
from climate_platform.quality_control.quality_control import QualityControlEngine

qce = QualityControlEngine()

ds_kelvin = xr.Dataset({"temperature": (["lat", "lon"], temp)}, coords={"lat": lat, "lon": lon})
passed1, report1 = qce.validate(ds_kelvin, data_source="default_model_state")
print(f"  Validate (default): passed={passed1}, rate={report1['overall_pass_rate']:.4f}")
print(f"  Record ID: {report1.get('record_id', 'N/A')}")
print(f"  Variable details: {report1.get('variable_details', {})}")

vd = report1.get('variable_details', {})
if 'temperature' in vd:
    t_vd = vd['temperature']
    assert t_vd.get('unit') != 'unknown', f"Temperature unit should not be unknown, got: {t_vd.get('unit')}"
    assert t_vd.get('range_min') is not None, "range_min should not be None"
    print(f"  Temperature: unit={t_vd.get('unit')}, range=[{t_vd.get('range_min')}, {t_vd.get('range_max')}]")

passed2, report2 = qce.validate(ds_kelvin, data_source="simulation_output")
print(f"  Validate (simulation): passed={passed2}")

history = qce.query_history(data_source="default_model_state")
print(f"  History for default_model_state: {len(history)} records")
assert len(history) >= 1, "Should have at least 1 record"

all_history = qce.get_validation_history(limit=10)
print(f"  All history (limit 10): {len(all_history)} records")
print("  ✓ QC validation history working!")

# ===================================================================
# Enhancement 3: Ingestion→Cleaning→QC Pipeline with API
# ===================================================================
print("\n[Enhancement 3] Pipeline Summary with Enhanced Fields")
from climate_platform.data_stream import DataCleaner, QCSummary

cleaner = DataCleaner(quality_threshold=0.85, auto_clean=True)

temp_anom = temp.copy()
temp_anom[10, 20] = 500.0
temp_anom[5, 5] = np.nan
ds_anom = xr.Dataset({"temperature": (["lat", "lon"], temp_anom)}, coords={"lat": lat, "lon": lon})

result, summary = cleaner.run_qc_with_summary(ds_anom, source_id="sat_dash", source_type="satellite_remote_sensing")
print(f"  Summary ID: {summary.summary_id}")
print(f"  Source: {summary.source_id}")
print(f"  Original anomalies: {summary.original_anomaly_points}")
print(f"  Original NaN: {summary.original_nan_count}")
print(f"  Original total points: {summary.original_total_points}")
print(f"  Cleaning interpolated: {summary.cleaning_interpolated_points}")
print(f"  Cleaned NaN: {summary.cleaned_nan_count}")
print(f"  Final pass rate: {summary.final_pass_rate:.4f}")
print(f"  Passed: {summary.passed}")
print(f"  QC failures detail: {summary.qc_failures_detail}")

assert summary.original_nan_count > 0, "Should detect original NaN"
assert summary.cleaned_nan_count == 0, "Should have 0 NaN after cleaning"
assert summary.cleaning_interpolated_points > 0, "Should have interpolated points"
assert summary.original_anomaly_points > 0, "Should detect anomalies"
print("  ✓ Pipeline summary with enhanced fields working!")

# ===================================================================
# Enhancement 4: Multi-tenant Quota API (Direct Test)
# ===================================================================
print("\n[Enhancement 4] Multi-tenant Quota API")
from climate_platform.multi_tenant import TenantManager, TenantRole, ResourceQuota

sandbox_path = "./test_sandboxes_v3"
if os.path.exists(sandbox_path):
    shutil.rmtree(sandbox_path, ignore_errors=True)
tm = TenantManager(base_sandbox_path=sandbox_path)
t = tm.create_tenant(tenant_id="api_test", name="API Test", role=TenantRole.RESEARCH)

# Test usage with all fields
usage = tm.get_tenant_usage("api_test")
for res_name, info in usage["usage"].items():
    has_fields = "used" in info and "quota" in info and "remaining" in info
    print(f"  {res_name}: used={info['used']}, quota={info['quota']}, remaining={info['remaining']} {'✓' if has_fields else '✗'}")
    assert has_fields, f"{res_name} missing fields"

# Test allocate_detailed with failure reasons
ok, reasons = tm.allocate_resources_detailed("api_test", ResourceQuota(storage_tb=2.0, concurrent_jobs=3))
print(f"  Allocate 2TB+3jobs: ok={ok}")
assert ok, "Should succeed"

ok2, reasons2 = tm.allocate_resources_detailed("api_test", ResourceQuota(storage_tb=100.0, concurrent_jobs=50))
print(f"  Over-quota 100TB+50jobs: ok={ok2}, reasons={reasons2}")
assert not ok2, "Should fail"
assert len(reasons2) > 0, "Should have specific reasons"

# Test release with history
tm.release_resources("api_test", ResourceQuota(storage_tb=1.0, concurrent_jobs=1))
history = tm.get_tenant_history("api_test")
print(f"  History events: {len(history)}")
alloc_events = [e for e in history if e.event_type == "allocate"]
release_events = [e for e in history if e.event_type == "release"]
print(f"  Alloc events: {len(alloc_events)}, Release events: {len(release_events)}")
assert len(release_events) == 1, "Should have 1 release event"

# Check usage after release
usage2 = tm.get_tenant_usage("api_test")
print(f"  After release: storage used={usage2['usage']['storage_tb']['used']}, remaining={usage2['usage']['storage_tb']['remaining']}")

# Filter by event_type
alloc_only = tm.get_tenant_history("api_test", event_type="allocate")
print(f"  Alloc-only history: {len(alloc_only)}")

shutil.rmtree(sandbox_path, ignore_errors=True)
print("  ✓ Multi-tenant quota API working!")

print("\n" + "=" * 70)
print("ALL 4 ENHANCEMENTS VERIFIED! ✓")
print("=" * 70)
