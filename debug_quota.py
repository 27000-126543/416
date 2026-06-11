"""Debug quota test."""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import logging; logging.basicConfig(level=logging.WARNING)

from climate_platform.multi_tenant import TenantManager, TenantRole, ResourceQuota
import shutil
sandbox_path = "./test_sandboxes_debug"
if os.path.exists(sandbox_path):
    shutil.rmtree(sandbox_path, ignore_errors=True)
tm = TenantManager(base_sandbox_path=sandbox_path)
t = tm.create_tenant(tenant_id="test", name="Test Tenant", role=TenantRole.RESEARCH)

print(f"Quota: {t.quota}")
print(f"Usage: {t.usage}")

print("\n--- Allocation 1: 2TB + 5 jobs ---")
req1 = ResourceQuota(storage_tb=2.0, concurrent_jobs=5)
ok1, reasons1 = tm.check_resource_available("test", req1)
print(f"Available: {ok1}, reasons: {reasons1}")
result1 = tm.allocate_resources("test", req1)
print(f"Allocated: {result1}")
print(f"Usage: {t.usage}")

print("\n--- Allocation 2: 3TB + 3 jobs ---")
req2 = ResourceQuota(storage_tb=3.0, concurrent_jobs=3)
ok2, reasons2 = tm.check_resource_available("test", req2)
print(f"Available: {ok2}, reasons: {reasons2}")
result2 = tm.allocate_resources("test", req2)
print(f"Allocated: {result2}")
print(f"Usage: {t.usage}")

shutil.rmtree(sandbox_path, ignore_errors=True)
