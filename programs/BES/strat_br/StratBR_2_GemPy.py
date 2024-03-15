import xarray as xr
import glob
import numpy as np
import pandas as pd
import re
import os
from sklearn.preprocessing import MinMaxScaler

# Paths
path_nc = "../../../input/BES/stratbr/new_tests_v5/raw/BESv2_Mapas_089.0-100.0Ma.nc"
path_raw = "../../../input/BES/stratbr/new_tests_v5/raw/"
path_processed = "../../../input/BES/stratbr/new_tests_v5/processed/"

# -----------------------------#
# NC to DF to CSV
# -----------------------------#

# Open the NetCDF file
dset = xr.open_dataset(path_nc)
dset.variables

# Checking variables
paleobat = dset.paleobat
paleobat_df = paleobat.to_dataframe().reset_index()

dept = dset.depth_0
dept_df = dept.to_dataframe().reset_index()

estruct = dset.estrutural
estruct_df = estruct.to_dataframe().reset_index()


def extract_paleobat_gempy_format(dataset, time_id):
    # Select data for the given time_id
    paleobat_slice = dataset["paleobat"].sel(time_d=time_id)

    # Make into DF
    paleobat_df = paleobat_slice.to_dataframe().reset_index()

    # Add 'formation' column
    paleobat_df["formation"] = "top"

    # Rename columns
    paleobat_df.rename(columns={"easting_d": "X", "northing_d": "Y", "paleobat": "Z"}, inplace=True)

    # Drop time_d
    paleobat_df.drop(columns="time_d", inplace=True)

    # Reorder columns
    paleobat_df = paleobat_df[["X", "Y", "Z", "formation"]]

    # Reset index
    paleobat_df.reset_index(drop=True, inplace=True)

    # Save to CSV
    paleobat_df.to_csv(path_raw + f"top.csv", index=False)

    return paleobat_df


# Function to extract depth_0 and estrutural for a time slice and save to CSV
def extract_data_gempy_format(dataset, time_id):
    # Select data for the given time_id
    estrutural_slice = dataset["estrutural"].sel(time_d=time_id)

    # Make into DF
    estrutural_df = estrutural_slice.to_dataframe().reset_index()

    # Rename 'estrutural' column to 'depth_d'
    estrutural_df.rename(columns={"estrutural": "depth_d"}, inplace=True)

    # Add 'formation' column
    estrutural_df["formation"] = "bes_" + str(time_id)

    # Rename columns
    estrutural_df.rename(columns={"easting_d": "X", "northing_d": "Y", "depth_d": "Z"}, inplace=True)

    # Drop time_d
    estrutural_df.drop(columns="time_d", inplace=True)

    # Reorder columns
    estrutural_df = estrutural_df[["X", "Y", "Z", "formation"]]

    # Reset index
    estrutural_df.reset_index(drop=True, inplace=True)

    # Save to CSV
    estrutural_df.to_csv(path_raw + f"bes_{time_id}.csv", index=False)

    return estrutural_df


# Example usage:
df_89 = extract_data_gempy_format(dset, 89.0)
df_top = extract_paleobat_gempy_format(dset, 89.0)
df_89.info()

# Looping through all time slices
time_ids = dset["time_d"].values
for t in time_ids:
    extract_data_gempy_format(dset, t)
    print(f"Saved time slice {t} to CSV")

# -----------------------------#
# Merging dfs
# -----------------------------#


def merge_csvs(time_slices, include_top=False, path_raw=""):
    # Initialize an empty list to store DataFrames
    dfs = []

    # If include_top is True, read the top.csv and append the DataFrame to the list
    if include_top:
        df_top = pd.read_csv(path_raw + "top.csv")
        dfs.append(df_top)

    # Loop over the time slices
    for t in time_slices:
        # Read the corresponding CSV file
        df = pd.read_csv(path_raw + f"bes_{t}.csv")
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs)

    # Reset the index of the merged DataFrame
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


def merge_csvs(time_slices, include_top=False, path_raw=""):
    # Initialize an empty list to store DataFrames
    dfs = []

    # If include_top is True, read the top.csv and append the DataFrame to the list
    if include_top:
        df_top = pd.read_csv(path_raw + "top.csv")
        dfs.append(df_top)

    # Loop over the time slices
    for i, t in enumerate(time_slices):
        # Read the corresponding CSV file
        df = pd.read_csv(path_raw + f"bes_{t}.csv")
        if i != -1:
            # Subtract 1000 times from Z
            df["Z"] = df["Z"] - i * 1000
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs)

    # Reset the index of the merged DataFrame
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


time_slices = [89.0, 100.0]
merged_df = merge_csvs(time_slices, include_top=True, path_raw=path_raw)
merged_df.to_csv(path_processed + "sp_full_merged_top_89_100.csv", index=False)

# -----------------------------#
# Scaling df
# -----------------------------#


def escalar_coordenadas(df):
    """
    Função para escalar as coordenadas X e Y de um DataFrame.

    Parâmetros:
    df (pandas.DataFrame): DataFrame contendo as coordenadas X e Y.

    Retorna:
    df (pandas.DataFrame): DataFrame com as coordenadas X e Y escaladas.
    """
    # Definir o escalonador para X com o intervalo de 0 até a diferença entre o máximo e o mínimo de X
    scaler_x = MinMaxScaler(feature_range=(0, (df["X"].max() - df["X"].min())))

    # Definir o escalonador para Y com o intervalo de 0 até a diferença entre o máximo e o mínimo de Y
    scaler_y = MinMaxScaler(feature_range=(0, (df["Y"].max() - df["Y"].min())))

    # Aplicar o escalonador em X
    df[["X"]] = scaler_x.fit_transform(df[["X"]])

    # Aplicar o escalonador em Y
    df[["Y"]] = scaler_y.fit_transform(df[["Y"]])

    # Imprimir as estatísticas descritivas do DataFrame
    print("\n", df.describe())

    return df


# -----------------------------#
# Reducing points
# -----------------------------#


def reduzir_pontos(df, eixo, n_pontos=1000, scale=False):
    """
    Função para reduzir o número de pontos em um DataFrame a cada N metros nas coordenadas X, Y ou ambas.

    Parâmetros:
    df (pandas.DataFrame): DataFrame contendo os pontos.
    eixo (str): Eixo para o qual a redução deve ser aplicada. Deve ser 'x', 'y' ou 'xy'.
    n_pontos (int, opcional): Distância em metros entre os pontos após a redução. Por padrão é 1000.

    Retorna:
    df_reduzido (pandas.DataFrame): DataFrame contendo os pontos após a redução.
    """
    df = df.copy()

    if scale:
        df = escalar_coordenadas(df)

    # Verificar qual eixo foi escolhido
    if eixo.lower() == "x":
        # Arredondar os valores de X para o múltiplo de n_pontos mais próximo
        df["X_rounded"] = (df["X"] // n_pontos) * n_pontos
        # Remover os pontos duplicados com base na formação e no valor arredondado de X, mantendo apenas o primeiro ponto de cada grupo
        df_reduzido = df.drop_duplicates(subset=["formation", "X_rounded"], keep="first")
        # Remover a coluna X_rounded
        df_reduzido = df_reduzido.drop(columns=["X_rounded"])
    elif eixo.lower() == "y":
        # Arredondar os valores de Y para o múltiplo de n_pontos mais próximo
        df["Y_rounded"] = (df["Y"] // n_pontos) * n_pontos
        # Remover os pontos duplicados com base na formação e no valor arredondado de Y, mantendo apenas o primeiro ponto de cada grupo
        df_reduzido = df.drop_duplicates(subset=["formation", "Y_rounded"], keep="first")
        # Remover a coluna Y_rounded
        df_reduzido = df_reduzido.drop(columns=["Y_rounded"])
    elif eixo.lower() == "xy":
        # Arredondar os valores de X e Y para o múltiplo de n_pontos mais próximo
        df["X_rounded"] = (df["X"] // n_pontos) * n_pontos
        df["Y_rounded"] = (df["Y"] // n_pontos) * n_pontos
        # Remover os pontos duplicados com base na formação e nos valores arredondados de X e Y, mantendo apenas o primeiro ponto de cada grupo
        df_reduzido = df.drop_duplicates(subset=["formation", "X_rounded", "Y_rounded"], keep="first")
        # Remover as colunas X_rounded e Y_rounded
        df_reduzido = df_reduzido.drop(columns=["X_rounded", "Y_rounded"])
    else:
        # Se o eixo não for 'x', 'y' ou 'xy', levantar um erro
        raise ValueError("O eixo deve ser 'x', 'y' ou 'xy'")

    # Resetar o índice do DataFrame reduzido
    df_reduzido.reset_index(drop=True, inplace=True)

    return df_reduzido


path_processed = "../../../input/BES/stratbr/new_tests_v5/processed/"
df_89_clean = pd.read_csv(path_raw + "bes_89.0.csv")

print(df_89["X"].min(), df_89["X"].max())
print(df_89["Y"].min(), df_89["Y"].max())
print(df_89["Z"].min(), df_89["Z"].max())

points = 2500
df_reduced = reduzir_pontos(merged_df, "xy", points, scale=False)
df_reduced.to_csv(path_processed + f"sp_merged_top_89_100_reduced_{points}.csv", index=False)
print(df_reduced["X"].min(), df_reduced["X"].max())
print(df_reduced["Y"].min(), df_reduced["Y"].max())
print(df_reduced["Z"].min(), df_reduced["Z"].max())

# -----------------------------#
# Getting center for orientation
# -----------------------------#


def get_center(df, formation):
    center = df[df["formation"] == formation][["X", "Y", "Z"]].mean()
    return center


def get_first_point(df, formation):
    first_point = df[df["formation"] == formation][["X", "Y", "Z"]].iloc[0]
    return first_point


def create_orientation(df, formations):
    # Initialize an empty list to store the data
    data = []

    # Loop over the formations
    for formation in formations:
        # Get the first point of the formation
        first_point = get_first_point(df, formation)

        # Append the data to the list
        data.append(
            {
                "X": first_point["X"],
                "Y": first_point["Y"].round(2),
                "Z": first_point["Z"].round(2),
                "azimuth": 0,
                "dip": 0,
                "polarity": 1,
                "formation": formation,
            }
        )

    # Create the DataFrame
    df_orientation = pd.DataFrame(data)

    return df_orientation


# Usage
formations = ["top", "bes_89.0", "bes_100.0"]
df_orientation = create_orientation(df_reduced, formations)
df_orientation.to_csv(path_processed + f"op_merged_top_89_100_{points}.csv", index=False)
