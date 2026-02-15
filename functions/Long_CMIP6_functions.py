import os
import cftime
import intake
import dask
import numpy as np
import xarray as xr
from scipy import integrate
import matplotlib.pyplot as plt
from IPython.display import HTML
from scipy.interpolate import interp1d

#import dask
# Function for renaming the time variable in IPSL

def to_cftime(time):
    return np.array([
        t if isinstance(t, cftime.DatetimeNoLeap) or isinstance(t, cftime.DatetimeGregorian)
        else cftime.DatetimeGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second)
        for t in time
    ])

# Function for time-average, considering that each month contains a different number of days. By Ada Gjermundsen. 

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


# Function for calculating the atmospheric fluxes

def atm_fluxes_fnc(model, start_y, end_y, experiments, components_atm):    

    atm_fluxes_exp = []
    
    for experiment in experiments:
        atm_fluxes = []
                
        for component in components_atm:
        
            folder_path = f'/mn/vann/chrikap/CMIP6/{model}/{experiment}/ATM/{component}'
            file_names = sorted(os.listdir(folder_path))
        
            # Using dask to read datasets
            datasets = [xr.open_dataset(os.path.join(folder_path, file_name), chunks={'time': 100}) for file_name in file_names if file_name.endswith('.nc')]
            
            if len(datasets) == 1:   # case where all datafiles are in the same .nc file
                ds = datasets[0]
                ds_y = ds.isel(time=slice(start_y, end_y))
            else:                    # if not, merge them together along the time dimension
                
                merged_datasets = xr.concat(datasets, dim='time')
                ds_y = merged_datasets.isel(time=slice(start_y, end_y))

            # IPSL has a diffenent naming of the time variables

            if model == 'IPSL-CM6A-LR':
                ds_y['time'] = to_cftime(ds_y['time'].values)

            if model == 'CESM2':
                ds_yearly = yearly_avg(ds_y[component][0, 0, :, :, :])
            else:        
                ds_yearly = yearly_avg(ds_y[component])
            atm_fluxes.append(ds_yearly) 
    
        atm_fluxes_exp.append(atm_fluxes)
    
    lon = ds_y['lon']  
    lat = ds_y['lat']

    return atm_fluxes_exp, lon, lat


# Function for applyin the area weights to the fluxes
def areaavg(ds, var):
    _da = ds[var]
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    weighted = _da.weighted(weights)
    _daglob = weighted.mean(("lon", "lat"))
    return _daglob

# Function for calculating the global and hemisphgeric imbalance

def imbalance(model, rsdt, rlut, rsut):
    toa_rad = rsdt-rlut-rsut

    if model == 'IPSL-CM6A-LR':
#        print('IPSL')
#        sh = toa_rad.sel(lat=slice(None, 0))
#        nh = toa_rad.sel(lat=rsdt.lat > 0)

        # Identify the equator index
        eq_idx = int((toa_rad.lat == 0).argmax())
        
        # Southern Hemisphere: all lat < 0 + half of equator
        sh = toa_rad.sel(lat=slice(None, 0))
        # Northern Hemisphere: all lat > 0 + half of equator
        nh = toa_rad.sel(lat=slice(0, None))
        
        # Split the equator value
        eq_val = toa_rad.sel(lat=0)
        sh.loc[dict(lat=0)] = eq_val / 2
        nh.loc[dict(lat=0)] = eq_val / 2

    else:
    
        sh = toa_rad.sel(lat=slice(None,0)) 
        #print(sh)
        nh = toa_rad.sel(lat=slice(0,None))
        #print(nh)
    
    sh_imb = areaavg(sh.to_dataset(name = "restom"), "restom")
    nh_imb = areaavg(nh.to_dataset(name = "restom"), "restom")
    
    glb_imb = areaavg(toa_rad.to_dataset(name = "restom"), "restom")

    return glb_imb, nh_imb, sh_imb 

# Function for calculating the heat transport long time

def heat_transport_vect(model, radiation, lat):
    #Radius of the Earth in [m]:
    R=6371000 

    integrad_1 = radiation 

    integrad_2 = (-0.5 * np.trapz(integrad_1 * np.cos(np.deg2rad(lat)), x=np.deg2rad(lat), dx = np.deg2rad(2)))[:, np.newaxis] # newaxis so I can perform the addition below

    integrad = (2*np.pi*R**2)*np.cos(np.deg2rad(lat))*(integrad_1+integrad_2)


    #integrad = (2*np.pi*R**2)*np.cos(np.deg2rad(lat))*(radiation) 
    #print(integrad.shape)

    #Integration in rads:

    if model in ['NorESM2-LM', 'IPSL-CM6A-LR']:
    
        heat_tr = integrate.cumulative_trapezoid(integrad, x=np.deg2rad(lat), dx = np.deg2rad(2), axis=0, initial=0) # axis =0 is correct. Although lat is in axis=1 
                                                                                                                     #in "radiation", it flips after the integrad
    elif model == 'CESM2':
                                                                                                                 
        heat_tr = integrate.cumulative_trapezoid(integrad, x=np.deg2rad(lat), dx = np.deg2rad(2), axis=1, initial=0)
    #[W] --> [PW]
    heat_tr = heat_tr*(10**-15)   
    
    return heat_tr

# Regridding for the atmospheric fluxes

def regridding (model, heat_tr, old_lat, target_lat):
    
    ht_interp_time = []

    if model in ['NorESM2-LM', 'IPSL-CM6A-LR']:

        heat_tr = heat_tr.T
    
    for ht in heat_tr:        
        
        interp_func = interp1d(old_lat, ht, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        ht_interp = interp_func(target_lat)
        
        ht_interp_time.append(ht_interp)            

    ht_interp_time = np.array(ht_interp_time)
    
    return ht_interp_time

# Function to process data and extract the global, atlantic and indopacific basin heat transports ########################################

name_mapping = {
    "Global": "global_ocean",
    "global_ocean": "global_ocean",
    "global": "global_ocean",
    "g": "global_ocean",  

    "Atlantic": "atlantic_arctic_ocean",    
    "atlantic_ocean": "atlantic_arctic_ocean",
    "atlantic_arctic_ocean": "atlantic_arctic_ocean",
    "atlantic": "atlantic_arctic_ocean",
    "a": "atlantic_arctic_ocean",    

    "Indo-Pacific": "indian_pacific_ocean",
    "indian_pacific_ocean": "indian_pacific_ocean",
    "indian-pacific": "indian_pacific_ocean",
    "i": "indian_pacific_ocean"       
}

def basin_separation_I(data, model_name):
    global_data = []
    atlantic_data = []
    indian_pacific_data = []

    # Some exceptions for some of the models...
    
    if model_name == 'IPSL-CM6A-LR':
        
        data = data.rename({'3basin': 'basin'})
        basin_labels = ['Global', 'Atlantic', 'indian_pacific_ocean']
        basin_indexes = range(len(basin_labels))      

    elif model_name == 'IPSL-CM6A-MR1':
        
        basin_labels = ['Global', 'Atlantic', 'indian_pacific_ocean']
        basin_indexes = range(len(basin_labels))  

    elif model_name == 'NorESM2-LM':
        basin_labels = ['atlantic_arctic_ocean', 'atlantic_arctic_extended_ocean', 'indian_pacific_ocean', 'global_ocean']
        basin_indexes = range(len(basin_labels))   

    #For the cases where the basin names are encoded and need to be extracted        
    elif 'sector' in data.coords and 'basin' in data.dims:
        
        basin_labels = data['sector'].values
        #basin_labels = [label.decode('utf-8').strip() for label in basin_labels]
        basin_labels = [label.decode('utf-8').strip() if isinstance(label, bytes) else str(label).strip() for label in basin_labels]
        basin_indexes = data['basin'].values
        
    elif 'basin' in data.dims and 'requested' in data['basin'].attrs:
        
        basin_labels = [k.split('=')[0].strip() for k in requested_attr.split(', ')]
        basin_indexes = data['basin'].values
        requested_attr = data['basin'].attrs['requested']        
        
    else:
        print(f"{model_name} : No valid basin names found.")
        return []

    # Create a mapping of indexes to standardized names
    basin_mapping = {}
    
    for idx, label in zip(basin_indexes, basin_labels):
        
        cleaned_label = label.strip()  # strip removes all potential white spaces #.lower() #turns all letters to to lower-case
        standardized_name = name_mapping.get(cleaned_label, f"unknown_{label}")
        print(f"Original: '{label}', Cleaned: '{cleaned_label}', Mapped: '{standardized_name}'")  # Debugging
        basin_mapping[idx] = standardized_name

    print("Basin Mapping:", basin_mapping)

    # Extract data for the global_ocean basin
    for idx, name in basin_mapping.items():

        basin_data = data.isel(basin=idx)  # Extract data for the basin
        
        if name == "global_ocean":
            global_data.append(basin_data)

        elif name == "atlantic_arctic_ocean":
            atlantic_data.append(basin_data)

        elif name == "indian_pacific_ocean":
            indian_pacific_data.append(basin_data)
            
    return global_data, atlantic_data, indian_pacific_data
    

# Calculating the OHT

def ocean_ht (model, experiments, start_y, end_y):

    hfbasins_gl_exp = []         
    hfbasins_atl_exp = []         
    hfbasins_ip_exp = []         
    
    for experiment in experiments:
    
        folder_path = f'/mn/vann/chrikap/CMIP6/{model}/{experiment}/OCEAN/hfbasin'
        file_names = sorted(os.listdir(folder_path))
        
        # Using dask to read datasets
        datasets = [xr.open_dataset(os.path.join(folder_path, file_name), chunks={'time': 100}) for file_name in file_names if file_name.endswith('.nc')]    
        if not datasets:
            print(f"Skipping {model} due to missing datasets")
            continue  # This model gets skipped
        
        if len(datasets) == 1:
            ds = datasets[0]
            model_50 = ds.isel(time=slice(start_y, end_y))
        else:
            # Merge all the datasets in the list along the same dimension
            merged_datasets = xr.concat(datasets, dim='time')
            model_50 = merged_datasets.isel(time=slice(start_y, end_y))

        if model == 'IPSL-CM6A-LR':

            lat = model_50['nav_lat'][:, 0].values

        else:
        
            lat =  model_50['lat'].values
        dask.compute()
                    
        component_values = model_50['hfbasin']

        if model == 'IPSL-CM6A-LR':
            component_values['time'] = to_cftime(component_values['time'].values)
            
        global_data, atlantic_data, indian_pacific_data = basin_separation_I(component_values, model)
        
        #print(global_data)
                    
        for global_basin_data, atlantic_basin_data, ip_basin_data in zip(global_data, atlantic_data, indian_pacific_data):

            #print(global_basin_data)
            
            # turning W->PW 
            global_basin_data = global_basin_data* 10**-15
            atlantic_basin_data = atlantic_basin_data* 10**-15
            ip_basin_data = ip_basin_data* 10**-15
    
            # turning monthly -> yearly averages
            global_basin_data = yearly_avg(global_basin_data)        
            atlantic_basin_data = yearly_avg(atlantic_basin_data)        
            ip_basin_data = yearly_avg(ip_basin_data)        
                        
            hfbasins_gl_exp.append(global_basin_data) 
            hfbasins_atl_exp.append(atlantic_basin_data) 
            hfbasins_ip_exp.append(ip_basin_data) 

    hfbasins_gl_exp = np.array(hfbasins_gl_exp)
    hfbasins_atl_exp = np.array(hfbasins_atl_exp)
    hfbasins_ip_exp = np.array(hfbasins_ip_exp)

    return hfbasins_gl_exp, hfbasins_atl_exp, hfbasins_ip_exp, lat

def gl_ocean_ht (model, experiments, start_y, end_y):

    hfbasins_gl_exp = []         
    
    for experiment in experiments:
    
        folder_path = f'/mn/vann/chrikap/CMIP6/{model}/{experiment}/OCEAN/hfbasin'
        file_names = sorted(os.listdir(folder_path))
        
        # Using dask to read datasets
        datasets = [xr.open_dataset(os.path.join(folder_path, file_name), chunks={'time': 100}) for file_name in file_names if file_name.endswith('.nc')]    
        
        if not datasets:
            print(f"Skipping {model} due to missing datasets")
            continue  # This model gets skipped
        
        if len(datasets) == 1:
            ds = datasets[0]
            model_50 = ds.isel(time=slice(start_y, end_y))
        else:
            # Merge all the datasets in the list along the same dimension
            merged_datasets = xr.concat(datasets, dim='time')
            model_50 = merged_datasets.isel(time=slice(start_y, end_y))

        if model == 'IPSL-CM6A-LR':
            lat = model_50['nav_lat'][:, 0].values
        else:        
            lat =  model_50['lat'].values
        dask.compute()
                    
        component_values = model_50['hfbasin']

        if model == 'IPSL-CM6A-LR':
            component_values['time'] = to_cftime(component_values['time'].values)
                
        global_data, _, _ = basin_separation_I(component_values, model)

        #print(global_data)
                    
        for global_basin_data in global_data:

            #print(global_basin_data)
            # turning W->PW 
            global_basin_data = global_basin_data* 10**-15
    
            # turning monthly -> yearly averages
            global_basin_data = yearly_avg(global_basin_data)                  
                       
            hfbasins_gl_exp.append(global_basin_data) 

    hfbasins_gl_exp = np.array(hfbasins_gl_exp)

    return hfbasins_gl_exp, lat


# Regridding for the ocean

def ocean_regr_vec(hfbasins_gl_exp, old_lat, target_lat):
    # Create interpolation function once per experiment
    # axis=-1 indicates interpolation along the lat axis
    oht_gl_interp_func = interp1d(
        old_lat,
        hfbasins_gl_exp,
        kind='linear',
        axis=-1,
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    # Evaluate interpolation on target_lat (broadcasts over experiments and time)
    oht_gl_int_exp = oht_gl_interp_func(target_lat)
    
    return oht_gl_int_exp

# Function for plotting the AHT, OT, AHT+OHT Anomalies
def ht_plots (model, num_y, aht, oht, toa_sum, target_lat, color_ind):

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=False)

    # Define your time windows
    time_windows = [(0, 30), (120, 150), (270, 300), (470, 500), (570, 600), (870, 900), (970, 999)]
    
    # Generate shades of blue (lighter → darker)
    #colors = plt.cm.Blues(np.linspace(0.2, 1, len(time_windows)))  # adjust range for contrast
    colors = plt.cm.rainbow(np.linspace(0, 1, len(time_windows)))
    
    for (start, end), color in zip(time_windows[0:color_ind], colors):
    
        # Calculate the anomalies
        # Atmosphere
        an_atm = np.mean(aht[1, start:end, :], axis=0) - np.mean(aht[0, 0:30, :], axis=0)
        
        # Ocean
        an_ocean = np.mean(oht[1, start:end, :], axis=0) - np.mean(oht[0, 0:30, :], axis=0)
        
        # TOA = ATM + OCEAN
        an_toa_sum = np.mean(toa_sum[1, start:end, :], axis=0) - np.mean(toa_sum[0, 0:30, :], axis=0)
    
        label = f'Years {start}-{end}'
        
        # --- Ocean ---
        axes[0].plot(target_lat, an_ocean, color = color, label=label)
        
        # --- Atmosphere ---
        axes[1].plot(target_lat, an_atm, color = color, label=label)
        
        # --- TOA ---
        axes[2].plot(target_lat, an_toa_sum, color = color, label=label)
    
    # ======= Formatting =======
    for ax, title in zip(axes, ['a) OCEAN', 'b) ATM', 'c) OCEAN+ATM']):
        ax.grid(alpha=0.5)
        ax.set_xlim(-90, 90)
        ax.set_ylim(-0.8, 1.5)
        ax.set_xlabel('Latitude [deg]', fontsize=15)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13)
    
    axes[0].set_ylabel('Heat Transport Anomalies [PW]', fontsize=16)
    
    # Add legend to the last panel
    axes[2].legend(
        loc='upper left', 
        bbox_to_anchor=(1, 1),  # moves legend outside to the right
        fontsize=13, 
        title="Time period"
    )
    fig.suptitle(f'Heat Transport Anomalies for {model}', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout(w_pad=4.0)
    plt.show()