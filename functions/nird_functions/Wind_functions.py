from common_NIRD_functions import*

def grab_NIRD_data(model, model_path, variable_path, variable, time_sl):
    
    experiments = ['piControl', 'abrupt-4xCO2']

    for experiment in experiments:

        folder_path = f'{model_path}/{experiment}/{variable_path}'
                    
        file_names = sorted(os.listdir(folder_path))

        if (experiment == 'piControl' and model == 'IPSL-CM6A-LR'):
        
            datasets = []
            
            for file_name in file_names:
                if not file_name.endswith(".nc"):
                    continue
        
                # extract start year from filename
                date_range = file_name.split("_")[-1].replace(".nc", "")
                start_year = int(date_range.split("-")[0][:4])
        
                if start_year >= 3050:
                    datasets.append(xr.open_dataset(os.path.join(folder_path, file_name), chunks={'time': 12}))
        else:
        
            datasets = [xr.open_dataset(os.path.join(folder_path, file_name), chunks={'time': 12}) for file_name in file_names if file_name.endswith('.nc')]

        # Adding the missing data for CESM2
        if (model =='CESM2' and experiment == 'abrupt-4xCO2' and variable == 'ua'):
            extra_file = '/nird/home/chrikap/supplementary_data/ua_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc'
            extra_ds = xr.open_dataset(extra_file, chunks={'time': 12})
            # Prepend it
            datasets = [extra_ds] + datasets

        if (model =='CESM2' and experiment == 'abrupt-4xCO2' and variable == 'va'):
            extra_file = '/nird/home/chrikap/supplementary_data/va_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc'
            extra_ds = xr.open_dataset(extra_file, chunks={'time': 12})
            # Prepend it
            datasets = [extra_ds] + datasets
        
        # Merge all the datasets in the list along the same dimension
        merged_datasets = xr.concat(datasets, dim='time')
        ds = merged_datasets.isel(time=slice(0, time_sl*12))

        if model == 'IPSL-CM6A-LR':
            ds['time'] = to_cftime(ds['time'].values)
    
        yearly_ds = yearly_avg(ds[variable])
    
        if experiment == 'piControl':
            piControl_data = yearly_ds
        elif experiment == 'abrupt-4xCO2':
            x4CO2_data = yearly_ds
            
    return piControl_data, x4CO2_data


# Globall averaging the winds to create a timeseries
# Function for calculating the global and hemisphgeric imbalance

def wind_gl_ave(model, wind):

    if model == 'IPSL-CM6A-LR':

        # Identify the equator index
        eq_idx = int((wind.lat == 0).argmax())
        
        # Southern Hemisphere: all lat < 0 + half of equator
        sh = wind.sel(lat=slice(None, 0))
        # Northern Hemisphere: all lat > 0 + half of equator
        nh = wind.sel(lat=slice(0, None))
        
        # Split the equator value
        eq_val = wind.sel(lat=0)
        sh.loc[dict(lat=0)] = eq_val / 2
        nh.loc[dict(lat=0)] = eq_val / 2

    else:
    
        sh = wind.sel(lat=slice(None,0)) 
        #print(sh)
        nh = wind.sel(lat=slice(0,None))
        #print(nh)
    
    sh_w = areaavg(sh.to_dataset(name = "restom"), "restom")
    nh_w = areaavg(nh.to_dataset(name = "restom"), "restom")
    
    glb_w = areaavg(wind.to_dataset(name = "restom"), "restom")

    return glb_w, nh_w, sh_w 