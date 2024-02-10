
#%%
%matplotlib inline
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt



#%%
ds = xr.tutorial.open_dataset("rasm").load()
ds
# %%
month_length = ds.time.dt.days_in_month
month_length.values


# %%
# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby("time.season") / month_length.groupby("time.season").sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

# Calculate the weighted average
ds_weighted = (ds * weights).groupby("time.season").sum(dim="time")
ds_weighted
# %%
weights
# %%
month_length.groupby("time.season")
# %%
ds
# %%
ds.attrs
# %%
