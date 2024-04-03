import pandas as pd
import xarray as xr
import numpy as np

datetime = pd.date_range(start='01-01-2023', end='01-01-2024')
x = np.arange(0, 64)
values = np.random.rand(datetime.size, x.size)

ds = xr.Dataset(
        {'s': (('datetime', 'x'), values)},
        coords = {
            'datetime': datetime,
            'x': x})

print(ds)
