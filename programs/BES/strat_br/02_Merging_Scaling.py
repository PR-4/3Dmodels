import xarray as xr
import glob
import numpy as np
import pandas as pd
import re
import os
from sklearn.preprocessing import MinMaxScaler


def reduce_dataframe(df, use_larger_list=False, reduce_y=False):
    df_red = df.copy()

    list_x = [
        0,
        10000,
        20000,
        30000,
        40000,
        50000,
        60000,
        70000,
        80000,
        90000,
        100000,
        110000,
        120000,
        130000,
        140000,
        150000,
        160000,
        170000,
        179000,
    ]

    if use_larger_list:
        list_x = [
            0,
            10000,
            20000,
            30000,
            40000,
            50000,
            60000,
            70000,
            80000,
            90000,
            100000,
            110000,
            120000,
            130000,
            140000,
            150000,
            160000,
            170000,
            179000,
        ]

    df_appended = []
    for x_value in list_x:
        ns_y = df_red["Y"].unique()
        new_df = df_red[(df_red["X"] == x_value) & (df_red["Y"].isin(ns_y))]
        df_appended.append(new_df)

    final_df = pd.concat(df_appended, ignore_index=True)

    if reduce_y:
        list_y = [
            0,
            5000,
            10000,
            15000,
            20000,
            25000,
            30000,
            35000,
            40000,
            45000,
            50000,
            55000,
            60000,
            65000,
            70000,
            75000,
            80000,
            85000,
            90000,
            95000,
            100000,
            105000,
            110000,
            115000,
            120000,
            125000,
            130000,
            135000,
            140000,
            145000,
            148000,
        ]

        df_appended = []
        for y_value in list_y:
            ns_x = final_df["X"].unique()
            new_df = final_df[(final_df["Y"] == y_value) & (final_df["X"].isin(ns_x))]
            df_appended.append(new_df)

        final_df = pd.concat(df_appended, ignore_index=True)

    return final_df


def add_topo(df):
    # Clone rows where formation is 'bes_89'
    df_topo = df[df["formation"] == "bes_89"].copy()

    # Change formation value to 'topo'
    df_topo["formation"] = "topo"

    # Add 1000 to 'Z'
    df_topo["Z"] += 1000

    # Concatenate the new rows to the original dataframe
    df = pd.concat([df, df_topo])

    return df


def sort_formations(df, has_topo=False):
    # Define the order of formations
    order = [
        "bes_89",
        "bes_90",
        "bes_91",
        "bes_92",
        "bes_93",
        "bes_94",
        "bes_95",
        "bes_96",
        "bes_97",
        "bes_98",
        "bes_99",
        "bes_100",
    ]

    if has_topo:
        order = ["topo"] + order

    # Create a categorical type with the specified order
    df["formation"] = pd.Categorical(df["formation"], categories=order, ordered=True)

    # Sort the dataframe by the 'formation' and 'X' columns
    df = df.sort_values(["formation", "X"])

    return df


# Path files
path_processed = "../../../input/BES/stratbr_grid_v4/processed/"
path_interim = "../../../input/BES/stratbr_grid_v4/interim/"

# Read DF
# df_sf = pd.read_csv(path_processed + "sf.csv")
df_89 = pd.read_csv(path_interim + "bes_89.csv")
df_99 = pd.read_csv(path_interim + "bes_99.csv")

# Concat DFS
df = pd.DataFrame()
df = pd.concat([df_89, df_99])

# Scaler
scaler_x = MinMaxScaler(feature_range=(0, (df["X"].max() - df["X"].min())))
scaler_y = MinMaxScaler(feature_range=(0, (df["Y"].max() - df["Y"].min())))
df[["X"]] = scaler_x.fit_transform(df[["X"]])
df[["Y"]] = scaler_y.fit_transform(df[["Y"]])

df.info()

# Save DF to CSV
df.to_csv(path_processed + "surfaces_points_scaled_merged_89_99_full.csv", index=False)

# Reduzindo número de pontos no DF
reduced = reduce_dataframe(df, use_larger_list=True, reduce_y=True)

# Adicionando topo
reduced = add_topo(reduced)

# Ordenando formações
reduced = sort_formations(reduced, has_topo=True)

# Checar topo min e max de Z para topo
# reduced.loc[reduced["formation"] == "topo", "Z"].min()
# reduced.loc[reduced["formation"] == "topo", "Z"].max()
# reduced.loc[reduced["formation"] == "bes_89", "Z"].min()
# reduced.loc[reduced["formation"] == "bes_89", "Z"].max()


# Salvar DF reduzido
reduced.to_csv(path_processed + "surfaces_points_scaled_merged_topo_89_99_reduced.csv", index=False)

# ----------------------------------------------
# Concanting and scaling all .csv
# ----------------------------------------------

# Path files
path_interim = "../../../input/BES/stratbr_grid_v4/interim/"

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(path_interim) if file.endswith(".csv")]
# Sort the CSV files based on the numeric values in their filenames
sorted_csv_files = sorted(csv_files, key=lambda x: int(re.search(r"\d+", x).group()))

# Initialize an empty DataFrame to store the concatenated data
concatenated_df = pd.DataFrame()

# Loop through the CSV files and concatenate them
for file_name in sorted_csv_files:
    file_path = os.path.join(path_interim, file_name)
    df = pd.read_csv(file_path)
    concatenated_df = pd.concat([concatenated_df, df])

# Create MinMaxScaler instances
scaler_x = MinMaxScaler(feature_range=(0, concatenated_df["X"].max() - concatenated_df["X"].min()))
scaler_y = MinMaxScaler(feature_range=(0, concatenated_df["Y"].max() - concatenated_df["Y"].min()))

# Apply scaling to the 'X' and 'Y' columns
concatenated_df[["X"]] = scaler_x.fit_transform(concatenated_df[["X"]])
concatenated_df[["Y"]] = scaler_y.fit_transform(concatenated_df[["Y"]])

# Ordenando formações
concatenated_df = sort_formations(concatenated_df, has_topo=False)

# Save the merged and scaled DataFrame to a CSV file
output_file_path = os.path.join(path_processed, "surfaces_points_scaled_merged_all_ages_full.csv")
concatenated_df.to_csv(output_file_path, index=False)

# ----------------------------------------------
# Reduzindo DF
# ----------------------------------------------

# Somente em X
reduced_df = reduce_dataframe(concatenated_df, use_larger_list=True, reduce_y=False)
# Adicionando topo
reduced_df = add_topo(reduced_df)
# Ordenando formações
reduced_df = sort_formations(reduced_df, has_topo=True)
# Salvando
reduced_df.to_csv(path_processed + "surfaces_points_scaled_merged_topo_all_ages_X_reduced.csv", index=False)

# X e Y
reduced_df = reduce_dataframe(concatenated_df, use_larger_list=True, reduce_y=True)
# Adicionando topo
reduced_df = add_topo(reduced_df)
# Ordenando formações
reduced_df = sort_formations(reduced_df, has_topo=True)
# Salvando
reduced_df.to_csv(path_processed + "surfaces_points_scaled_merged_topo_all_ages_X_and_Y_reduced.csv", index=False)


# ----------------------------------------------
# Ajustar a profundidade Z de cada formação
# ----------------------------------------------
"""
Adicionar -500 m para a formação das bes
"""

# Define the Z adjustments for each formation
z_adjustments = {
    "bes_90": -500,
    "bes_91": -1000,
    "bes_92": -1500,
    "bes_93": -2000,
    "bes_94": -2500,
    "bes_95": -3000,
    "bes_96": -3500,
    "bes_97": -4000,
    "bes_98": -4500,
    "bes_99": -5000,
    "bes_100": -5500,
    # Add more formations and adjustments as needed
}

# Apply Z adjustments based on the formation name
for formation, adjustment in z_adjustments.items():
    reduced_df.loc[reduced_df["formation"] == formation, "Z"] += adjustment

reduced_df.to_csv(
    path_processed + "surfaces_points_scaled_merged_topo_all_ages_X_and_Y_reduced_Z_ajusted.csv", index=False
)

reduced_df.describe()
reduced_df.info()

# ----------------------------------------------
# Duplicate all rows with formation name 'bes_89', rename to SF, and add +500 to the Z column
# read df
"""df_final = pd.read_csv(path_processed + "surfaces_points_scaled_merged_all_ages_X_and_Y_reduced_Z_ajusted.csv")

df_sf = df_final.loc[df_final["formation"] == "bes_89"].copy()

# Rename the formation name to 'SF'
df_sf["formation"] = "TOP"
# Add 500 to the Z column
df_sf["Z"] += 500
# Append the new rows to the DataFrame
df_final = df_final.append(df_sf, ignore_index=True)

# Save the DataFrame to a CSV file
df_final.to_csv(
    "../../../input/BES/stratbr_grid_v3/gempy_format/2_merged_sp_reduced_more_z_ajusted.csv",
    index=False,
)"""
