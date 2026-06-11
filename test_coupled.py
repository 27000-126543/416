"""Quick coupled model test with low resolution."""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np

print("=" * 60)
print("Coupled Model Test (Low Resolution: 5 deg)")
print("=" * 60)

from climate_platform.model import (
    CoupledModel, CouplingType,
    AtmosphericModel, OceanModel, LandModel, SeaIceModel,
)

print("\nCreating models with 5-degree resolution...")
atmos = AtmosphericModel(name="atmosphere", vertical_levels=5, resolution_deg=5.0, time_step_seconds=10800)
ocean = OceanModel(name="ocean", vertical_levels=5, resolution_deg=5.0, time_step_seconds=10800)
land = LandModel(name="land", resolution_deg=5.0, time_step_seconds=10800)
seaice = SeaIceModel(name="sea_ice", resolution_deg=5.0, time_step_seconds=10800)

model = CoupledModel(
    coupling_type=CouplingType.FULLY_COUPLED,
    coupling_interval_seconds=10800,
)
model.atmosphere = atmos
model.ocean = ocean
model.land = land
model.sea_ice = seaice

print("Initializing...")
states = model.initialize()
print(f"  ✓ Components: {list(states.keys())}")
for name, state in states.items():
    print(f"    {name}: {len(state.data.data_vars)} vars")

# Check key coupling variables
sst_before = states["ocean"].data["sea_surface_temperature"].mean().item()
atmos_t_surf_before = states["atmosphere"].data["temperature"].isel(level=0).mean().item()
skin_t_before = states["land"].data["skin_temperature"].mean().item()
sic_before = states["sea_ice"].data["sea_ice_concentration"].mean().item()

print(f"\nBefore coupling steps:")
print(f"  Ocean SST mean: {sst_before:.2f} K")
print(f"  Atmos surface T mean: {atmos_t_surf_before:.2f} K")
print(f"  Land skin T mean: {skin_t_before:.2f} K")
print(f"  Sea ice concentration mean: {sic_before:.4f}")

print(f"\nRunning 5 coupling steps...")
for i in range(5):
    model.step()
    if (i + 1) % 1 == 0:
        print(f"  Step {i+1} done")

sst_after = states["ocean"].data["sea_surface_temperature"].mean().item()
atmos_t_surf_after = states["atmosphere"].data["temperature"].isel(level=0).mean().item()
skin_t_after = states["land"].data["skin_temperature"].mean().item()
sic_after = states["sea_ice"].data["sea_ice_concentration"].mean().item()

print(f"\nAfter 5 steps:")
print(f"  Ocean SST mean: {sst_after:.2f} K  (change: {sst_after - sst_before:+.4f})")
print(f"  Atmos surface T mean: {atmos_t_surf_after:.2f} K  (change: {atmos_t_surf_after - atmos_t_surf_before:+.4f})")
print(f"  Land skin T mean: {skin_t_after:.2f} K  (change: {skin_t_after - skin_t_before:+.4f})")
print(f"  Sea ice concentration mean: {sic_after:.4f}  (change: {sic_after - sic_before:+.6f})")

# Verify that coupling actually made a change
sst_change = abs(sst_after - sst_before)
atmos_change = abs(atmos_t_surf_after - atmos_t_surf_before)
land_change = abs(skin_t_after - skin_t_before)

print(f"\nCoupling verification:")
print(f"  SST changed: {sst_change > 1e-6} ({sst_change:.6f})")
print(f"  Atmos surface T changed: {atmos_change > 1e-6} ({atmos_change:.6f})")
print(f"  Land skin T changed: {land_change > 1e-6} ({land_change:.6f})")

total_change = sst_change + atmos_change + land_change
assert total_change > 1e-6, "No coupling change detected!"
print(f"\n  ✓ Coupling is active! Total change: {total_change:.6f} K")

print("\n" + "=" * 60)
print("COUPLED MODEL TEST PASSED! ✓")
print("=" * 60)
