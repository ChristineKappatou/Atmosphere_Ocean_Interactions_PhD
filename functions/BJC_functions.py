######################################################## Importing necessary packages ####################################################

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
from scipy.optimize import differential_evolution

#for nice plots
import cmcrameri
import matplotlib.pyplot as plt

#import dask
#for ignoring stuff
import warnings
warnings.simplefilter("ignore")
    
# for saving lists
import pickle 

############################################################ Functions ###################################################################

def rolling_average(data, window_size):

    model_s, time, lat_s = data.shape
    output_length = time - window_size + 1
    result = np.zeros((model_s, output_length, lat_s))

    for i in range(output_length):
        window = data[:, i:i+window_size, :]  # shape: (#models, window_size, #latitudes)
        result[:, i, :] = np.sum(window, axis=1) / window_size

    return result

def rolling_average_ts(data, window_size):

    time = (data.shape)[0]
    output_length = time - window_size + 1
    result = np.zeros(output_length)

    for i in range(output_length):
        window = data[i:i+window_size] # shape: (#models, window_size, #latitudes)
        result[i] = np.sum(window, axis=0) / window_size

    return result


# Function for single exponential with constant c
def single_exponential(x, a1, b1, c):
    return a1 * np.exp(b1 * x) + c

# Function for the double exponential with constant c
def double_exponential(x, a1, b1, a2, b2, c):
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + c

# Objective function (sum of squared residuals)
def objective_function_1(params, x, y):
    a1, b1, c = params
    y_pred = single_exponential(x, a1, b1, c)
    return np.sum((y - y_pred) ** 2)
    
def objective_function_2(params, x, y):
    a1, b1, a2, b2, c = params
    y_pred = double_exponential(x, a1, b1, a2, b2, c)
    return np.sum((y - y_pred) ** 2)