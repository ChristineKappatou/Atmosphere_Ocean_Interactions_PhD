# We're using a different yearly averaging function because the one we usually use makes the dask graph layers explode and the calculations become slow. 
# So now we're avoiding .groupby
import xarray as xr
import numpy as np

import os
import cftime





