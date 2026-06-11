import sys
sys.path.insert(0, '.')
from datetime import datetime
from climate_platform.model import CoupledModel, CouplingType

model = CoupledModel(coupling_type=CouplingType.FULLY_COUPLED, coupling_interval_seconds=3600)
states = model.initialize(start_time=datetime(2020, 1, 1))
atmos = states["atmosphere"].data
print("Atmosphere variables:")
for v in atmos.data_vars:
    arr = atmos[v].values
    print(f"  {v}: shape={arr.shape}, size={arr.size}, dtype={arr.dtype}")
total_bytes = sum(atmos[v].values.nbytes for v in atmos.data_vars)
print(f"\nTotal: {total_bytes/1024/1024:.2f} MB")
