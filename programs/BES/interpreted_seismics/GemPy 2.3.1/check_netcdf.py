import pandas as pd
import xarray as xr
import numpy as np
from netCDF4 import Dataset


# path
path = "../../../../../Privado/Python Libraries/Hydrogeological Model/hydrogeological_model_3D_voxel/hgsm_nor.nc"

# Abrir o arquivo NetCDF
nc = Dataset(path, "r")


print(nc.variables)
print(nc.groups)


# Acessar o grupo 'topography'
topography = nc["topography"]

# Acessar a variável 'easting' dentro do grupo 'topography'
easting = topography["easting"][:]

# Imprimir os dados da variável 'easting'
print(easting)
