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

# Different models have different names for each basin. Here, we rename everything so they all have the same name. #######################
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

# Function to process data and extract the global, atlantic and indopacific basin heat transports ########################################
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