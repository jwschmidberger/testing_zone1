
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
month_length.groupby("time.season")
# %%
# only used for comparisons
ds_unweighted = ds.groupby("time.season").mean("time")
ds_diff = ds_weighted - ds_unweighted
# %%
ds_diff
# %%
# Quick plot to show the results
notnull = pd.notnull(ds_unweighted["Tair"][0])

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 12))
for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):
    ds_weighted["Tair"].sel(season=season).where(notnull).plot.pcolormesh(
        ax=axes[i, 0],
        vmin=-30,
        vmax=30,
        cmap="Spectral_r",
        add_colorbar=True,
        extend="both",
    )

    ds_unweighted["Tair"].sel(season=season).where(notnull).plot.pcolormesh(
        ax=axes[i, 1],
        vmin=-30,
        vmax=30,
        cmap="Spectral_r",
        add_colorbar=True,
        extend="both",
    )

    ds_diff["Tair"].sel(season=season).where(notnull).plot.pcolormesh(
        ax=axes[i, 2],
        vmin=-0.1,
        vmax=0.1,
        cmap="RdBu_r",
        add_colorbar=True,
        extend="both",
    )

    axes[i, 0].set_ylabel(season)
    axes[i, 1].set_ylabel("")
    axes[i, 2].set_ylabel("")

for ax in axes.flat:
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis("tight")
    ax.set_xlabel("")

axes[0, 0].set_title("Weighted by DPM")
axes[0, 1].set_title("Equal Weighting")
axes[0, 2].set_title("Difference")

plt.tight_layout()

fig.suptitle("Seasonal Surface Air Temperature", fontsize=16, y=1.02)
# %%
# Wrap it into a simple function
def season_mean(ds, calendar="standard"):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby("time.season").sum(dim="time")
# %%
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 4))
ds.xc.plot(ax=ax1)
ds.yc.plot(ax=ax2)
# %%
'''Next one'''
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

%matplotlib inline
%config InlineBackend.figure_format='retina'

#%%
ds = xr.tutorial.load_dataset("air_temperature")
ds

#%%
# pull out "air" dataarray with dictionary syntax
ds["air"]



# %%
