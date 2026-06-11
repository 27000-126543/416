import numpy as np
import xarray as xr
from climate_platform.quality_control.quality_control import QualityControlEngine, ValidationRecord

engine = QualityControlEngine()

lat = np.linspace(-90, 90, 10)
lon = np.linspace(0, 360, 20)
lon_grid, lat_grid = np.meshgrid(lon, lat)
temp_data = 288 - 30 * np.abs(np.sin(np.radians(lat_grid)))
wind_data = 10 * np.ones_like(temp_data)

ds = xr.Dataset(
    {
        'temperature': (['lat', 'lon'], temp_data),
        'wind_speed': (['lat', 'lon'], wind_data),
    },
    coords={'lat': lat, 'lon': lon},
)

passed, report = engine.validate(ds, data_source='test_source')
print('validate() passed:', passed)
print('record_id in report:', 'record_id' in report)
print('timestamp in report:', 'timestamp' in report)
print('data_source in report:', report.get('data_source'))
print('variable_details keys:', list(report.get('variable_details', {}).keys()))

history = engine.get_validation_history()
print('validation history length:', len(history))
print('first record type:', type(history[0]))
print('first record data_source:', history[0].data_source)

query_result = engine.query_history(data_source='test_source')
print('query by data_source result count:', len(query_result))

query_result2 = engine.query_history(variable='temperature')
print('query by variable result count:', len(query_result2))

print('All tests passed!')
