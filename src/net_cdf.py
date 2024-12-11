import numpy as np
import pandas as pd
import xarray as xr

# Substitua 'path_to_file' pelo caminho do seu arquivo
extent = np.load("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_extent.npy")
lith_block = np.load("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_lith_block.npy")
resolution = np.load("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_resolution.npy")
sections = np.load("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_sections.npy")


# Carregar os dados em um DataArray
# data_array = xr.DataArray(extent)

# Carregar os dados de um arquivo CSV
faults = pd.read_csv("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_faults.csv", index_col=0)
faults_relations = pd.read_csv("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_faults_relations.csv", index_col=0)
kriging_data = pd.read_csv("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_kriging_data.csv", index_col=0)
options = pd.read_csv("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_options.csv", index_col=0)
orientations = pd.read_csv("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_orientations.csv", index_col=0)
rescaling_data = pd.read_csv("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_rescaling_data.csv", index_col=0)
series = pd.read_csv("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_series.csv", index_col=0)
surface_points = pd.read_csv("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_surface_points.csv", index_col=0)
surfaces = pd.read_csv("../output/BES/StartBR/v4/gempy_2.3.1/StratBR_BES_v4_surfaces.csv", index_col=0)

ds = xr.Dataset()
num_points = 100
x = np.linspace(extent[0], extent[1], num_points)
y = np.linspace(extent[2], extent[3], num_points)
z = np.linspace(extent[4], extent[5], num_points)

ds = ds.assign_coords(x=x, y=y, z=z)


# Adicione cada conjunto de dados como uma variável separada
ds["lith_block"] = xr.DataArray(lith_block)
ds.attrs["resolution"] = resolution


ds.attrs["faults"] = faults
ds.attrs["faults_relations"] = faults_relations
ds.attrs["kriging_data"] = kriging_data
ds.attrs["options"] = options
orientations_da = xr.DataArray(orientations)
surface_points_da = xr.DataArray(surface_points)

ds["orientations"] = orientations_da
ds["surface_points"] = surface_points_da


# Adicione os DataFrames como variáveis
ds.attrs["faults"] = xr.DataArray(faults.to_numpy(), dims=["x", "y"])
ds["faults_relations"] = xr.DataArray(faults_relations.to_numpy(), dims=["x", "y"])
ds["kriging_data"] = xr.DataArray(kriging_data.to_numpy(), dims=["x", "y"])
ds["options"] = xr.DataArray(options.to_numpy(), dims=["x", "y"])
ds["orientations"] = xr.DataArray(orientations.to_numpy(), dims=["x", "y"])
ds["rescaling_data"] = xr.DataArray(rescaling_data.to_numpy(), dims=["x", "y"])
ds["series"] = xr.DataArray(series.to_numpy(), dims=["x", "y"])
ds["surface_points"] = xr.DataArray(surface_points.to_numpy(), dims=["x", "y"])
ds["surfaces"] = xr.DataArray(surfaces.to_numpy(), dims=["x", "y"])

# Defina os limites do Dataset usando o objeto extent
ds = ds.assign_coords(x=extent[0], y=extent[1])
