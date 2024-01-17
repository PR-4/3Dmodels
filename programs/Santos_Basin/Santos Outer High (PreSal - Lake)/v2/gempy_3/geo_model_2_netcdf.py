import pickle
import xarray as xr
import numpy as np
from gempy.core.data import Grid

# Load geo_model.pkl
path_to_pickle = "../gempy_3/geo_model.pkl"

# Load the geo_model object from a pickle file
with open(path_to_pickle, "rb") as f:
    geo_model = pickle.load(f)


def dict_to_dataset(data):
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            processed_data[key] = dict_to_dataset(value)
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            if value.size > 0:
                processed_data[key] = xr.DataArray(value)
            else:
                processed_data[key] = xr.DataArray([np.nan])
        elif isinstance(value, Grid):
            # Handle Grid objects separately
            if value.values.size > 0:
                processed_data[key] = xr.DataArray(value.values)
            else:
                processed_data[key] = xr.DataArray([np.nan])
        else:
            # Convert other types to strings
            processed_data[key] = str(value)
    return xr.Dataset(processed_data)


# Your data
ds = dict_to_dataset(geo_model.__dict__)

# Convert to xarray Dataset
ds = dict_to_dataset(data)

# Save to netCDF
ds.to_netcdf("data.nc")
