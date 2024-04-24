#%%
'''
Imports
'''
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns



#%%
ds = xr.tutorial.load_dataset("air_temperature")
ds


# %%
type(ds.air)
# %%
with xr.set_options(display_style="html"):
    display(ds)
# %%
ds.air.coords
# %%
