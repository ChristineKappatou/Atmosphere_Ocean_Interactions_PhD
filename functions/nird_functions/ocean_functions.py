# Functions
from __future__ import annotations

# load useful packages
import xarray as xr
import numpy as np
xr.set_options(display_style='html')
import xesmf as xe
from matplotlib.gridspec import GridSpec
from scipy import integrate
import matplotlib.pyplot as plt
#import general_util_funcs as guf
#%matplotlib inline

import os
import cftime

def consistent_naming(_ds):
    """
    The naming convention for coordinates and dimensions are not the same 
    for noresm raw output and cmorized variables. This function rewrites the 
    coords and dims names to be consistent and the functions thus work on all
    Choose the cmor naming convention.

    Parameters
    ----------
   _ds : xarray.Dataset 

    Returns
    -------
   _ds : xarray.Dataset

    """
    #print(_ds)

#    if isinstance(obj, xr.DataArray):
#        obj = obj.to_dataset(name=obj.name or "var")
#        obj = consistent_naming(obj)
#        return obj[list(obj.data_vars)[0]]  # return as DataArray
#    elif isinstance(obj, xr.Dataset):
#        _ds = obj.copy()
#        # ... keep your current renaming logic ...
#        return _ds
#    else:
#        raise TypeError("Input must be xarray.DataArray or xarray.Dataset")

    if isinstance(_ds, xr.DataArray):
        _ds = _ds.to_dataset(name=_ds.name or "var")
        _ds = consistent_naming(_ds)
        return _ds[list(_ds.data_vars)[0]]  # return as DataArray
    elif isinstance(_ds, xr.Dataset):
        _ds = _ds.copy()
    else:
        raise TypeError("Input must be xarray.DataArray or xarray.Dataset")


    if "latitude" in _ds.coords and "lat" not in _ds.coords:
       _ds =_ds.rename({"latitude": "lat"})
    if "longitude" in _ds.coords and "lon" not in _ds.coords:
       _ds =_ds.rename({"longitude": "lon"})
    if "x" in _ds.dims:
       _ds =_ds.swap_dims({"x": "i"})
    if "y" in _ds.dims:
       _ds =_ds.swap_dims({"y": "j"})
    if "ni" in _ds.dims:
       _ds =_ds.swap_dims({"ni": "i"})
    if "nj" in _ds.dims:
       _ds =_ds.swap_dims({"nj": "j"})
    if "nlat" in _ds.dims:
       _ds =_ds.swap_dims({"nlat": "i"})
    if "nlon" in _ds.dims:
       _ds =_ds.swap_dims({"nlon": "j"})
    if "nv" in _ds.dims:
       _ds =_ds.swap_dims({"nv": "vertices"})
    if "nvertex" in _ds.dims:
       _ds =_ds.swap_dims({"nvertex": "vertices"})
    if "TLAT" in _ds.coords:
       _ds =_ds.rename({"TLAT": "lat"})
    if "TLONG" in _ds.coords:
       _ds =_ds.rename({"TLONG": "lon"})
    if "depth" in _ds.dims:
       _ds =_ds.swap_dims({"depth": "lev"})
    if "depth_bnds" in _ds.dims:
       _ds =_ds.swap_dims({"depth_bnds": "lev_bnds"})
    if "nbnd" in _ds.dims:
       _ds =_ds.swap_dims({"nbnd": "bnds"})
    if "hist_interval" in _ds.dims:
       _ds =_ds.swap_dims({"hist_interval": "bnds"})
    if "nbounds" in _ds.dims:
       _ds =_ds.swap_dims({"nbounds": "bnds"})
    if "bounds" in _ds.dims:
       _ds =_ds.swap_dims({"bounds": "bnds"})
    if "type" in _ds.coords:
       _ds =_ds.drop_vars("type")
    if 'bounds_nav_lat' in _ds.variables:
       _ds =_ds.rename({'bounds_nav_lat':'vertices_latitude', 'bounds_nav_lon':'vertices_longitude'})
    if 'latitude_bnds' in _ds.variables:
       _ds =_ds.rename({'latitude_bnds':'vertices_latitude', 'longitude_bnds':'vertices_longitude'})
    if 'nav_lat' in _ds.coords:
       _ds =_ds.rename({'nav_lon':'lon','nav_lat':'lat'})
    if 'lat_bnds' in _ds.variables:
       _ds = _ds.rename({'lat_bnds':'vertices_latitude','lon_bnds':'vertices_longitude'})
    return _ds


def make_bounds_ocean(_ds):
    """
    This function calculates latitude and longitude values and boundaries used for regridding
    from curvilinear grids to regular lat/lon grid (rectilinear grid)
    The Dataset generated is used as_ds_in in the regridder function

    The ocean/sea-ice grid of NorESM2 is a tripolar grid with 360 and 384 unique grid cells in i- and j-direction, respectively.
    Due to the way variables are staggered in the ocean model, an additional j-row is required explaining the 385 grid cells in the j-direction for the ocean grid.
    The row with j=385 is a duplicate of the row with j=384, but with reverse i-index.
    Ocean variables are on i=360, j=385 grid
    Sea-ice variables are on i=360, j=384 grid

    Parameters
    ----------
   _ds : xarray.DataSet, with model grid information for the data which need to be regridded
                         on (i,j,vertices) format where the 4 vertices give grid corner information

    Returns
    -------
   _ds_in :  xarray.DataSet, with 2D model grid information for the data which need to be regridded
    """
 
    ny, nx =_ds.lat.shape  # ny will be 384 for sea-ice variables
    
    if "lat" in _ds.lon.coords:
        lat_model =_ds.lat.isel(j=slice(0, ny)).drop_vars("lon").drop_vars("lat")
        lon_model =_ds.lon.isel(j=slice(0, ny)).drop_vars("lon").drop_vars("lat")
    else:
        lat_model =_ds.lat.isel(j=slice(0, ny))
        lon_model =_ds.lon.isel(j=slice(0, ny))
    lon_b_model = xr.concat(
        [_ds.vertices_longitude.isel(vertices=0),_ds.vertices_longitude.isel(vertices=1, i=-1)], dim="i"
    )
    
    lon_b_model = xr.concat(
        [
            lon_b_model,
            xr.concat(
                [_ds.vertices_longitude.isel(vertices=3, j=-1),_ds.vertices_longitude.isel(vertices=2, j=-1, i=-1)],
                dim="i",
            ),
        ],
        dim="j",
    )
   
    lat_b_model = xr.concat(
        [_ds.vertices_latitude.isel(vertices=0),_ds.vertices_latitude.isel(vertices=1, i=-1)], dim="i"
    )
    lat_b_model = xr.concat(
        [
            lat_b_model,
            xr.concat(
                [_ds.vertices_latitude.isel(vertices=3, j=-1),_ds.vertices_latitude.isel(vertices=2, j=-1, i=-1)],
                dim="i",
            ),
        ],
        dim="j",
    )
   
    if "lat" in lon_b_model.coords:
        lon_b_model = (
            lon_b_model.isel(j=slice(0, ny + 1))
            .rename("lon_b")
            .swap_dims({"j": "y_b", "i": "x_b"})
            .drop_vars("lon")
            .drop_vars("lat")
        )
        lat_b_model = (
            lat_b_model.isel(j=slice(0, ny + 1))
            .rename("lat_b")
            .swap_dims({"j": "y_b", "i": "x_b"})
            .drop_vars("lon")
            .drop_vars("lat")
        )
    else:
        lon_b_model = lon_b_model.isel(j=slice(0, ny + 1)).rename("lon_b").swap_dims({"j": "y_b", "i": "x_b"})
        lat_b_model = lat_b_model.isel(j=slice(0, ny + 1)).rename("lat_b").swap_dims({"j": "y_b", "i": "x_b"})
    lat_b_model = lat_b_model.where(lat_b_model < 90.0, 90.0)
    lat_b_model = lat_b_model.where(lat_b_model > -90.0, -90.0)
   
    if "time" in lat_b_model.dims:
        lat_b_model = lat_b_model.isel(time=0).drop_vars("time")
        lon_b_model = lon_b_model.isel(time=0).drop_vars("time")
   
    if "month" in lat_b_model.dims:
        lat_b_model = lat_b_model.isel(month=0).drop_vars("month")
        lon_b_model = lon_b_model.isel(month=0).drop_vars("month")
 
    ds_out = xr.Dataset()
    ds_out= ds_out.assign_coords(lat=lat_model)
    ds_out = ds_out.assign_coords(lon=lon_model)  
    ds_out= ds_out.assign_coords(lat_b=lat_b_model)
    ds_out = ds_out.assign_coords(lon_b=lon_b_model)
    keep_coords = ["lat", "lon", "lat_b", "lon_b"]
    for coord in list(ds_out.coords):
        if coord not in keep_coords:
            ds_out = ds_out.drop_vars(coord)
    if "x" in ds_out.coords:    
        ds_out = ds_out.drop('x')
    if "x_b" in ds_out.coords:    
        ds_out = ds_out.drop('x_b')
    if "y" in ds_out.coords:    
        ds_out = ds_out.drop('y')
    if "y_b" in ds_out.coords:    
        ds_out = ds_out.drop('y_b')
    ds_out = ds_out.swap_dims({"j": "y", "i": "x"})
    return ds_out


def make_regridder(
   _ds, outgrid, regrid_mode="conservative", curvilinear=True, periodic = True
):
    """The first step of the regridding routine!
    There is an important reason why the regridding is broken into two steps
    (making the regridder and perform regridding). For high-resolution grids,
    making the regridder (i.e. “computing regridding weights”) is quite computationally expensive,
    but performing regridding on data (“applying regridding weights”) is still pretty fast.

    Parameters
    ----------
   _ds :                 xarray.DataSet, with model grid information for the data which need to be regridded
    outgrid :            xarray.DataSet, with output grid information which the data will be regridded to
    regrid_mode : str,   'bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s'
    curvilinear :         bool, True for ocean and se-ice variables. False for atmosphere

    A comment about the ignore_degenerate = True option: if the grid cells have very close corners, the ESMF thinks that
    it is a triangle instead of a quadrilateral (i.e. "degenerated") and throws out an error which can be skipped by 
    setting ignore_degenerate=True. Alternatively, you can manually remove such degenerated cells.
    
    Returns
    -------
    regridder : xarray.DataSet with regridder weight file information
    """
    ds_in = make_bounds_ocean(_ds)
    regridder = xe.Regridder(ds_in, outgrid, method=regrid_mode, periodic=periodic, ignore_degenerate=True)
    return regridder


def regrid_file(
   _ds, var, outgrid,  grid_weight_path=None, regrid_mode="conservative", curvilinear=True, periodic = True
):
    """Second step of the regridding routine!

    Parameters
    ----------
   _ds :                 xarray.DataSet, with model grid information for the data which need to be regridded
    var :                str, name of varible
    outgrid :            xarray.DataSet, with output grid information which tif 'plev'in _ds[var].dims:
               
    Returns
    -------
    dr : xarray.Dataset with regridded variable data and lon from 0-360
    """
    print("In regrid_file")
    regridder= make_regridder(_ds, outgrid, regrid_mode, curvilinear, periodic)
    print("Regridding maker made")
    # Fix for numpy 2 while waiting for xESMF 0.8.7
    regridder.shape_in = tuple(map(int, regridder.shape_in))
    regridder.shape_out = tuple(map(int, regridder.shape_out))
    dr = regridder(_ds[var])  # needs DataArray
    lon = dr.lon.isel(y=0).drop_vars('lat').drop_vars('lon')
    lat = dr.lat.isel(x=0).drop_vars('lat').drop_vars('lon')
    dr = dr.drop_vars('lat').drop_vars('lon')
    dr = dr.assign_coords(lat=lat)
    dr = dr.assign_coords(lon=lon)
    dr = dr.swap_dims({"x": "lon", "y": "lat"})
    dr = dr.to_dataset(name=var)
    if "long_name" in _ds[var].attrs:
        dr[var].attrs["long_name"] =_ds[var].long_name
    if "units" in _ds[var].attrs:
        dr[var].attrs["units"] =_ds[var].units
    if "standard_name"in _ds[var].attrs:
        dr[var].attrs["standard_name"] =_ds[var].standard_name
    print("Regridding completed")
    return dr

def regrid_ocean(_ds,  var:str, outdir: str):#, expid: str, first_year:int, last_year:int,  tempstorage: str = "/scratch/adagj/noresm_raw/temp/" ):
   
    ## THIS PART YOU DON*T NEED _ YOU CAN ALSO JUST DEFINE A 1 x 1 GRID with 
    outgrid = xe.util.grid_global(1, 1)
    
    grid_weight_path = outdir
    regrid_mode = "nearest_s2d"

    dr = regrid_file(
                _ds,
                var=var,
                outgrid=outgrid,
                grid_weight_path=grid_weight_path,
                regrid_mode=regrid_mode,
                curvilinear=True,
                periodic=True,
            )
    return dr

# Function to process data
def basin_separation_I(amoc_data, dataset_name, model, model_name):
    atlantic_data = []
    global_data = []

    # Extract basin names 
    if dataset_name == "piControl":
        if model_name =='NorESM2-LM':
            basin_labels = ['atlantic_arctic_ocean', 'atlantic_arctic_extended_ocean', 'indian_pacific_ocean', 'global_ocean']
        else:
            basin_labels = ['atlantic_arctic_ocean', 'indian_pacific_ocean', 'global_ocean']
        basin_indexes = range(len(basin_labels))  # Create a default index range
    elif 'sector' in model.coords and 'basin' in model.dims:
        basin_indexes = model['basin'].values
        basin_labels = model['sector'].values
        basin_labels = [label.decode('utf-8').strip() for label in basin_labels]
    elif 'basin' in model.dims and 'requested' in model['basin'].attrs:
        basin_indexes = model['basin'].values
        requested_attr = model['basin'].attrs['requested']
        basin_labels = [k.split('=')[0].strip() for k in requested_attr.split(', ')]
    else:
        print(f"{dataset_name} Model {'model'}: No valid basin names found.")
        #continue

    print(basin_labels)

    # Create a mapping of indexes to standardized names
    basin_mapping = {}
    for idx, label in zip(basin_indexes, basin_labels):
        standardized_name = name_mapping.get(label.lower(), f"unknown_{label}")
        basin_mapping[idx] = standardized_name

    # Categorize and store data
    for idx, name in basin_mapping.items():
        basin_data = model.isel(basin=idx)  # Extract data for the basin
        if name == "atlantic_arctic_ocean":
            atlantic_data.append(basin_data)
        elif name == "global_ocean":
            global_data.append(basin_data)

    print(f"{dataset_name} Model {'model'}: Basin mapping -> {basin_mapping}")
    print()

    return atlantic_data, global_data