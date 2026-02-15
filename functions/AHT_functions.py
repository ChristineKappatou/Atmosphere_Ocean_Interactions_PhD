######################################################## Importing necessary packages ####################################################

# for changing directory
import os

# for opening and working with the .grib/.nv files
import cfgrib
import cftime
import intake
import warnings
import xarray as xr
xr.set_options(display_style='html')
warnings.simplefilter("ignore")

# for calculations
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate

# for nice displaying
import pandas as pd               
from IPython.display import HTML

# for nice plots
import matplotlib.pyplot as plt

############################################################ Functions ###################################################################

# Function to extract model names ########################################################################################################
def extract_model_name(string_name):
    return string_name.split('.')[2]

#Radius of the Earth in [m]:
R=6371000 

# Function for calculating the heat transport ############################################################################################
def heat_transport(radiation, lat):
    
    integrad = (2*np.pi*R**2)*np.cos(np.deg2rad(lat))*(radiation) 

    #Integration in rads:
    heat_tr = integrate.cumulative_trapezoid(integrad, x=np.deg2rad(lat), dx = np.deg2rad(2), initial=0)

    #[W] --> [PW]
    heat_tr = heat_tr*(10**-15)   
    
    return heat_tr

# The Heat transport function for time-dependent values ##################################################################################
# We will only work with time-dependent values of the heat transports. Then we can average over time if we choose for it.                                                                                                                
                                                                                                               
def time_depend_heat_trs (heat_fluxes, lon, lat, num_y):

    toa_rad = heat_fluxes[1] - heat_fluxes[0] - heat_fluxes[2]
    atm_rad = toa_rad + heat_fluxes[3] + heat_fluxes[4] - heat_fluxes[5] + heat_fluxes[6] - heat_fluxes[7] + heat_fluxes[8]

    toa_zonal_ave = np.mean(toa_rad, axis = 2)#(np.trapz(toa_rad, x=np.deg2rad(lon), axis=2)) / (2 * np.pi)
    atm_zonal_ave = np.mean(atm_rad, axis = 2)#(np.trapz(atm_rad, x=np.deg2rad(lon), axis=2)) / (2 * np.pi) 

    # Calculating the monthly means so we have yearly data

    #toa_zonal_ave_reshaped = toa_zonal_ave.reshape(num_y, 12, len(lat))
    #toa_zonal_ave = np.mean(toa_zonal_ave_reshaped, axis=1)

    # Get number of days in each month
    #days_in_month = toa_zonal_ave['time'].dt.days_in_month
    
    # Normalize the weights within each year
    #weights = days_in_month / days_in_month.groupby('time.year').sum()
    
    # Apply weights and sum by year
    #toa_zonal_ave_weighted = (toa_zonal_ave * weights).groupby('time.year').sum()
    
    #atm_zonal_ave_reshaped = atm_zonal_ave.reshape(num_y, 12, len(lat))
    #atm_zonal_ave = np.mean(atm_zonal_ave_reshaped, axis=1)
    
    # Same process we followed for toa
    #(atm_zonal_ave * weights).groupby('time.year').sum()
        
    toa_ht_t = []
    atm_ht_t = []

    for i in range(num_y):
            
        toa_ht_t.append(heat_transport(toa_zonal_ave[i, :], lat))
        atm_ht_t.append(heat_transport(atm_zonal_ave[i, :], lat))

    toa_ht_t = np.array(toa_ht_t)
    atm_ht_t = np.array(atm_ht_t)

    return toa_ht_t, atm_ht_t