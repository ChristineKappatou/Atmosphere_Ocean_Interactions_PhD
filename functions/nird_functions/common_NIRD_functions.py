# Packages
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
# for saving lists
import pickle 
#for plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec

# Functions

# for IPSL time
def to_cftime(time):
    return np.array([
        t if isinstance(t, cftime.DatetimeNoLeap) or isinstance(t, cftime.DatetimeGregorian)
        else cftime.DatetimeGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second)
        for t in time
    ])

# for yearly average
def yearly_avg(ds):
    """Calculate yearly weighted average from monthly means."""
    
    month_length = ds.time.dt.days_in_month
    
    def weighted_mean_year(x):
        # Select weights for that specific year's time subset
        w = month_length.sel(time=x.time)
        return x.weighted(w).mean(dim="time")
    
    ds_weighted = ds.groupby("time.year").apply(weighted_mean_year)
    
    # Copy metadata
    if "long_name" in ds.attrs:
        ds_weighted.attrs["long_name"] = "Annual mean " + ds.long_name
    if "units" in ds.attrs:
        ds_weighted.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_weighted.attrs["standard_name"] = ds.standard_name
    
    return ds_weighted

def yearly_avg_nan_proof(ds): 
    """ The same yearly average function this time excluding the nans when taking the mean
    
    Parameters
    ----------
    ds : xarray.Dataset
    
    Returns
    -------
    ds_weighted : xarray.DataArray with yearly averaged values
    
    """
    month_length = ds.time.dt.days_in_month
    weights = month_length.groupby("time.year") / month_length.groupby("time.year").sum(skipna = True)
    
    # Test that the sum of the weights for each year is 1.0
    np.testing.assert_allclose(weights.groupby("time.year").sum(skipna = True).values, np.ones(len(np.unique(ds.time.dt.year))))
    
    # Calculate the weighted average:
    ds_weighted = (ds * weights).groupby("time.year").sum(dim="time", skipna = True)
    
    if "long_name" in ds.attrs:
        ds_weighted.attrs["long_name"] = "Annual mean " + ds.long_name
    if "units" in ds.attrs:
        ds_weighted.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_weighted.attrs["standard_name"] = ds.standard_name
    
    return ds_weighted

# for global averaging
def areaavg(ds, var):
    _da = ds[var]
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    weighted = _da.weighted(weights)
    _daglob = weighted.mean(("lon", "lat"))
    return _daglob

# for plotting contours of decided values
def nearest_levels(data, target_levels):
    data_vals = np.unique(data[~np.isnan(data)])
    return np.array([
        data_vals[np.argmin(np.abs(data_vals - lvl))]
        for lvl in target_levels
    ])
    
# two functions for rolling average
def rolling_average_ts(data, window_size):

    time = (data.shape)[0]
    output_length = time - window_size + 1
    result = np.zeros(output_length)

    for i in range(output_length):
        window = data[i:i+window_size]  # shape: (#models, window_size, #latitudes)
        result[i] = np.sum(window, axis=0) / window_size

    return result


def convolve_f(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="valid")

