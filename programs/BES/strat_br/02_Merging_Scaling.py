import xarray as xr
import glob
import numpy as np
import pandas as pd
import re
import os
from sklearn.preprocessing import MinMaxScaler

# Path files
path_files = "../../../input/BES/stratbr_grid_v3/processed/"

# Read DF
df_sf = pd.read_csv(path_files + "sf.csv")
df_89 = pd.read_csv(path_files + "bes_89.csv")
df_99 = pd.read_csv(path_files + "bes_99.csv")

# Concat DFS
df = pd.concat([df_89, df_99])

# Scaler
scaler_x = MinMaxScaler(feature_range=(0, (df["X"].max() - df["X"].min())))
scaler_y = MinMaxScaler(feature_range=(0, (df["Y"].max() - df["Y"].min())))
df[["X"]] = scaler_x.fit_transform(df[["X"]])
df[["Y"]] = scaler_y.fit_transform(df[["Y"]])

# Save DF to CSV
df.to_csv(path_files + "merged_89_99_sp_full.csv", index=False)

# ----------------------------------------------
# Concanting and scaling all .csv

# Path files
folder_path = "../../../input/BES/stratbr_grid_v3/interim/"

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# Sort the CSV files based on the numeric values in their filenames
sorted_csv_files = sorted(csv_files, key=lambda x: int(re.search(r"\d+", x).group()))

# Initialize an empty DataFrame to store the concatenated data
concatenated_df = pd.DataFrame()

# Loop through the CSV files and concatenate them
for file_name in sorted_csv_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    concatenated_df = pd.concat([concatenated_df, df])

# Create MinMaxScaler instances
scaler_x = MinMaxScaler(
    feature_range=(0, concatenated_df["X"].max() - concatenated_df["X"].min())
)
scaler_y = MinMaxScaler(
    feature_range=(0, concatenated_df["Y"].max() - concatenated_df["Y"].min())
)

# Apply scaling to the 'X' and 'Y' columns
concatenated_df[["X"]] = scaler_x.fit_transform(concatenated_df[["X"]])
concatenated_df[["Y"]] = scaler_y.fit_transform(concatenated_df[["Y"]])

# Save the merged and scaled DataFrame to a CSV file
output_file_path = os.path.join(folder_path, "merged_dfs_sp_full.csv")
concatenated_df.to_csv(output_file_path, index=False)


# Reducing DF
df_red = concatenated_df.copy()

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

final_df.to_csv(path_files + "merged_sp_reduced.csv", index=False)

# Reducing Y points
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

df_appended = []  # list to append dataframes
# loop over list of y values
for y_value in list_y:
    ns_x = final_df["X"].unique()  # get unique x values
    new_df = final_df[
        (final_df["Y"] == y_value) & (final_df["X"].isin(ns_x))
    ]  # filter dataframe
    df_appended.append(new_df)  # append dataframe to list

final_df = pd.concat(df_appended, ignore_index=True)  # concatenate dataframes

final_df.to_csv(
    "../../../input/BES/stratbr_grid_v3/gempy_format/merged_all_sp_reduced_more.csv",
    index=False,
)

# ----------------------------------------------


df_final = pd.read_csv(path_files + "merged_sp_reduced.csv")

final_df.describe()
final_df.info()


# Reducing Y points
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

df_appended = []  # list to append dataframes
# loop over list of y values
for y_value in list_y:
    ns_x = df_final["X"].unique()  # get unique x values
    new_df = df_final[
        (df_final["Y"] == y_value) & (df_final["X"].isin(ns_x))
    ]  # filter dataframe
    df_appended.append(new_df)  # append dataframe to list

final_df = pd.concat(df_appended, ignore_index=True)  # concatenate dataframes

final_df.to_csv(path_files + "merged_sp_reduced_more.csv", index=False)

# Z adjustment

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
    final_df.loc[final_df["formation"] == formation, "Z"] += adjustment

final_df.to_csv(path_files + "merged_sp_reduced_more_z_ajusted.csv", index=False)

final_df.describe()

df_final = pd.read_csv(
    "../../../input/BES/stratbr_grid_v3/gempy_format/merged_sp_reduced_more_z_ajusted.csv"
)

# ----------------------------------------------
# Duplicate all rows with formation name 'bes_89', rename to SF, and add +500 to the Z column
# read df
df_final = pd.read_csv(
    "../../../input/BES/stratbr_grid_v3/gempy_format/merged_sp_reduced_more_z_ajusted.csv"
)

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
)
