# Packages

# for navigating into directories
import os
from pathlib import Path
#for dealing with .nc data
import cfgrib
import cftime
import xarray as xr
#for dealing with panda arrays
import numpy as np
import pandas as pd
#for math
import xesmf as xe                             # for regridding
from glob import glob
from scipy import integrate
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import dask
#for ignoring stuff
import warnings
warnings.simplefilter("ignore")
# for skiping running a certain cell
from IPython.core.magic import register_cell_magic

@register_cell_magic
def comment(line, cell):
    return
    
# for saving lists
import pickle 

# for fancy plotting
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

# Functions

def to_cftime(time):
    return np.array([
        t if isinstance(t, cftime.DatetimeNoLeap) or isinstance(t, cftime.DatetimeGregorian)
        else cftime.DatetimeGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second)
        for t in time
    ])

def yearly_avg(ds):
    """ Calculates timeseries over yearly averages from timeseries of monthly means
    The weighted average considers that each month has a different number of days.
    
    Parameters
    ----------
    ds : xarray.Dataset
    
    Returns
    -------
    ds_weighted : xarray.DataArray with yearly averaged values
    
    """
    month_length = ds.time.dt.days_in_month
    weights = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    
    # Test that the sum of the weights for each year is 1.0
    np.testing.assert_allclose(weights.groupby("time.year").sum().values, np.ones(len(np.unique(ds.time.dt.year))))
    
    # Calculate the weighted average:
    ds_weighted = (ds * weights).groupby("time.year").sum(dim="time")
    
    if "long_name" in ds.attrs:
        ds_weighted.attrs["long_name"] = "Annual mean " + ds.long_name
    if "units" in ds.attrs:
        ds_weighted.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_weighted.attrs["standard_name"] = ds.standard_name
    
    return ds_weighted

def areaavg(ds, var):
    _da = ds[var]
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    weighted = _da.weighted(weights)
    _daglob = weighted.mean(("lon", "lat"))
    return _daglob

def plot_map_anomalies_zoom_contour(
    model,
    data_piControl,
    data_periods,
    extent,
    vmin_pi,
    vmax_pi,
    vmin_anom,
    vmax_anom,
    title,
    colorbar_label_pi,
    colorbar_label_anom
):
    """
    Plot contour maps for piControl and anomalies at a selected pressure level.
    
    Parameters
    ----------
    data_piControl : xr.DataArray (plev, lat, lon)
    data_periods   : list of xr.DataArray, same structure
    plev_value     : float or None
    Pressure level (Pa) to plot. If None, uses first level.    
    levels         : int
    Number of contour levels.
    """
    levels = np.linspace(vmin_pi, vmax_pi, 19)
    levels_anom = np.linspace(vmin_anom, vmax_anom, 19)

    lat = data_piControl.lat
    lon = data_piControl.lon
   
    titles = ["piControl y0-y30", "y120-y150", "y470-y500", "y870-y900", "y970-y999"]

    if model == 'NorESM2-LM':
        nrows = 2
    elif model == 'IPSL-CM6A-LR':
        nrows = 3
    elif model == 'CESM2':
        nrows = 4

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(nrows=nrows, ncols=2, width_ratios=[1, 1.2])
    
    ax_pi = fig.add_subplot(gs[:, 0], projection=ccrs.Robinson())
    
    # --- piControl plot ---
    im_pi = ax_pi.contourf(
            lon,
            lat,
            data_piControl,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin_pi,
            vmax=vmax_pi,
            extend="both"
        )

    ax_pi.coastlines()
    ax_pi.set_extent(extent, crs=ccrs.PlateCarree())
    ax_pi.set_title(f"{titles[0]}", fontsize=12, fontweight="bold")

    gl = ax_pi.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Colorbar for piControl
    cbar_ax_pi = fig.add_axes([0.05, 0.15, 0.02, 0.7])
    cbar_pi = fig.colorbar(im_pi, cax=cbar_ax_pi, orientation="vertical")
    cbar_pi.set_label(colorbar_label_pi, fontweight='bold', labelpad=15)
    cbar_pi.ax.yaxis.set_label_position('left')  # move label to left side
    cbar_pi.ax.yaxis.tick_left()                 # move ticks to left    
    
    for tick in cbar_pi.ax.yaxis.get_ticklabels():
        tick.set_fontweight('bold')

    # --- anomaly plots ---
    for i in range(nrows):
        ax = fig.add_subplot(gs[i, 1], projection=ccrs.Robinson())
        if i < len(data_periods):
            data_period = data_periods[i]
            data_anom = data_period - data_piControl
        else:
            continue

        im_anom = ax.contourf(
                lon,
                lat,
                data_anom,
                levels=levels_anom,
                transform=ccrs.PlateCarree(),
                cmap="coolwarm",                
                vmin=vmin_anom,
                vmax=vmax_anom,
                extend="both"
            )

        ax.coastlines()
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        if i == 0:
            gl.bottom_labels = False
        else:
            gl.bottom_labels = True
        
        ax.set_title(f"{titles[i+1]}", fontsize=12, fontweight='bold')

    # --- colorbar for anomalies ---
    cbar_anom_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar_anom = fig.colorbar(im_anom, cax=cbar_anom_ax, orientation="vertical")
    cbar_anom.set_label(colorbar_label_anom, fontweight='bold')
    for tick in cbar_anom.ax.yaxis.get_ticklabels():
        tick.set_fontweight('bold')

    fig.suptitle(title, fontsize=16, weight='bold', y=1.02)
    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.25, wspace=0.05)
    plt.show()

