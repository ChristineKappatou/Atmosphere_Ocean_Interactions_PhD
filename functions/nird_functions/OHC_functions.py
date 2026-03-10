# We're using a different yearly averaging function because the one we usually use makes the dask graph layers explode and the calculations become slow. 
# So now we're avoiding .groupby
import xarray as xr
import numpy as np

import os
import cftime

def yearly_avg_manual(model, ds, time_dim="time"):
    """
    Dask-safe yearly average from monthly data
    without using groupby.
    """

    nt = ds.sizes[time_dim]
    if nt % 12 != 0:
        raise ValueError("Time dimension length must be a multiple of 12")

    nyears = nt // 12

    # --- month lengths (cheap, 1D) ---
    month_length = ds[time_dim].dt.days_in_month
    month_length = month_length.data.reshape(nyears, 12)

    weights = month_length / month_length.sum(axis=1, keepdims=True)

    # --- build (year, month) index ---
    year = np.repeat(np.arange(nyears), 12)
    month = np.tile(np.arange(1, 13), nyears)

    ds2 = ds.assign_coords(
        year=(time_dim, year),
        month=(time_dim, month),
    )

    # --- unstack time → (year, month) ---
    ds2 = ds2.set_index({time_dim: ("year", "month")}).unstack(time_dim)

    # --- weighted average over month ---
    weights_da = xr.DataArray(
        weights,
        dims=("year", "month"),
        coords={"year": ds2.year, "month": ds2.month},
    )

    yearly = (ds2 * weights_da).sum(dim="month")
    # --- rename year → time ---
    #yearly = yearly.rename({"year": "time"})

    # optional: restore a proper time coordinate (year midpoints)
    yearly = yearly.assign_coords(year=yearly.year.values)
    
    if model == 'NorESM2-LM':
        yearly = yearly.transpose("year", "bnds", "lev", "j", "i", "vertices")
        
    elif model == "IPSL-CM6A-LR":
        # Put vertical first, then y, x
        all_dims = list(yearly.dims)
        ordered_dims = ["year"]
        if "olevel" in all_dims:
            ordered_dims.append("olevel")
        if "y" in all_dims:
            ordered_dims.append("y")
        if "x" in all_dims:
            ordered_dims.append("x")
        # Add any remaining dims at the end
        remaining_dims = [d for d in all_dims if d not in ordered_dims]
        ordered_dims += remaining_dims

        yearly = yearly.transpose(*ordered_dims)   

    elif model == 'CESM2':

        yearly = yearly.transpose("year", "lev", "nlat", "nlon", "d2", "vertices")        
    # --- attributes ---
    for attr in ("long_name", "units", "standard_name"):
        if attr in ds.attrs:
            yearly.attrs[attr] = ds.attrs[attr]

    return yearly

def ohc_funct_dV(model, weighted_temperature):
    
    rho = 1035 #kg/m^3   , values taken from: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2020GL091439 +++
    c_p = 4000 #J/(kg*C)
        
    if model == 'NorESM2-LM':
        ohc = weighted_temperature.sum(dim=("lev", "j", "i"))*rho*c_p
    elif model == 'IPSL-CM6A-LR':
        ohc = weighted_temperature.sum(dim=("olevel", 'y', 'x'))*rho*c_p
    elif model == 'CESM2':
        ohc = weighted_temperature.sum(dim=("lev", "nlat", "nlon"))*rho*c_p

    else:
        print('not a model here')
    return ohc

