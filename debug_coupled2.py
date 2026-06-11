"""Detailed coupled model debug."""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import logging; logging.basicConfig(level=logging.DEBUG)
import numpy as np
from datetime import datetime, timedelta

from climate_platform.model import (
    CoupledModel, CouplingType,
    AtmosphericModel, OceanModel, LandModel, SeaIceModel,
)

print("Creating models...")
atmos = AtmosphericModel(name="atmosphere", vertical_levels=5, resolution_deg=5.0, time_step_seconds=3600)
ocean = OceanModel(name="ocean", vertical_levels=5, resolution_deg=5.0, time_step_seconds=3600)
land = LandModel(name="land", resolution_deg=5.0, time_step_seconds=3600)
seaice = SeaIceModel(name="sea_ice", resolution_deg=5.0, time_step_seconds=3600)

# Initialize individually
print("\nInitializing individually...")
atmos_state = atmos.initialize()
ocean_state = ocean.initialize()
land_state = land.initialize()
seaice_state = seaice.initialize()

print(f"Atmos T(level=0) mean: {atmos_state.data['temperature'].isel(level=0).mean().item():.2f} K")
print(f"Ocean SST mean: {ocean_state.data['sea_surface_temperature'].mean().item():.2f} K")
print(f"Land skin T mean: {land_state.data['skin_temperature'].mean().item():.2f} K")
print(f"SeaIce T mean: {seaice_state.data['sea_ice_temperature'].mean().item():.2f} K")

# Now test one coupling step manually
print("\nTesting coupling manually...")
model = CoupledModel(coupling_type=CouplingType.FULLY_COUPLED, coupling_interval_seconds=86400)
model.atmosphere = atmos
model.ocean = ocean
model.land = land
model.sea_ice = seaice
model._setup_coupling_interfaces()
model.states = {
    "atmosphere": atmos_state,
    "ocean": ocean_state,
    "land": land_state,
    "sea_ice": seaice_state,
}

print(f"Number of interfaces: {len(model.coupling_scheduler.interfaces)}")
for iface in model.coupling_scheduler.interfaces:
    print(f"  {iface.source_model} -> {iface.target_model}: {len(iface.variable_maps)} vars")

# Execute one coupling step
dt = timedelta(hours=1)
print(f"\nExecuting coupling with dt={dt}...")
model.coupling_scheduler.execute_coupling(model.states, model._remap_data, dt=dt)

print(f"\nAfter 1 coupling step:")
print(f"Atmos T(level=0) mean: {atmos_state.data['temperature'].isel(level=0).mean().item():.2f} K")
print(f"Ocean SST mean: {ocean_state.data['sea_surface_temperature'].mean().item():.2f} K")
print(f"Land skin T mean: {land_state.data['skin_temperature'].mean().item():.2f} K")
print(f"SeaIce T mean: {seaice_state.data['sea_ice_temperature'].mean().item():.2f} K")
