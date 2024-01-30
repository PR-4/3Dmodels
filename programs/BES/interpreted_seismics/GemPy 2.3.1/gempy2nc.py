import pandas as pd
import xarray as xr
import numpy as np

# List of file names
files = [
    "result_v1/BES_model_seismic_v1_extent.npy",
    "result_v1/BES_model_seismic_v1_faults.csv",
    "result_v1/BES_model_seismic_v1_faults_relations.csv",
    "result_v1/BES_model_seismic_v1_kriging_data.csv",
    "result_v1/BES_model_seismic_v1_lith_block.npy",
    "result_v1/BES_model_seismic_v1_options.csv",
    "result_v1/BES_model_seismic_v1_orientations.csv",
    "result_v1/BES_model_seismic_v1_rescaling_data.csv",
    "result_v1/BES_model_seismic_v1_resolution.npy",
    "result_v1/BES_model_seismic_v1_sections.csv",
    "result_v1/BES_model_seismic_v1_sections.npy",
    "result_v1/BES_model_seismic_v1_series.csv",
    "result_v1/BES_model_seismic_v1_surface_points.csv",
    "result_v1/BES_model_seismic_v1_surfaces.csv",
    # Add other file names here
]

# Abra cada arquivo e imprima seu conteúdo
for file in files:
    print(f"Conteúdo do arquivo {file}:")
    if file.endswith(".npy"):
        array = np.load(file)
        print(array)
    print("\n")

# Abra cada arquivo e imprima seu conteúdo
for file in files:
    print(f"Conteúdo do arquivo {file}:")
    if file.endswith(".csv"):
        df = pd.read_csv(file)
        print(df.head())  # Imprime as primeiras 5 linhas do DataFrame
    elif file.endswith(".npy"):
        array = np.load(file)
        print(array)
    print("\n")

# Dicionário para armazenar os Datasets
datasets = {}


if file.endswith(".npy"):
    array = np.load(file)
    # Redimensione o array para que ele tenha a mesma dimensão que os outros Datasets
    array = array.reshape(-1, 1)
    ds = xr.DataArray(array).to_dataset(name=file)
    datasets[file] = ds

if file.endswith(".csv"):
    df = pd.read_csv(file)
    # Reindexe o DataFrame para que ele tenha a mesma dimensão que os outros Datasets
    df = df.reindex(range(max_dim_size))
    ds = xr.Dataset.from_dataframe(df)
    datasets[file] = ds


# Abra cada arquivo e crie um Dataset
for file in files:
    if file.endswith(".npy"):
        array = np.load(file)
        ds = xr.DataArray(array).to_dataset(name=file)
        datasets[file] = ds

# Abra cada arquivo e crie um Dataset
for file in files:
    if file.endswith(".csv"):
        df = pd.read_csv(file)
        ds = xr.Dataset.from_dataframe(df)
        datasets[file] = ds
    elif file.endswith(".npy"):
        array = np.load(file)
        ds = xr.DataArray(array).to_dataset(name=file)
        datasets[file] = ds

# Combine todos os Datasets em um único Dataset
combined_ds = xr.merge(list(datasets.values()))


# Save the merged Dataset as a NetCDF file
merged_ds.to_netcdf("merged_dataset.nc")
