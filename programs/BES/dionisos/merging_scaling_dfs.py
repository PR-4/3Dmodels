import pandas as pd
import os
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

# files path
path = "../../../input/BES/dionisos_horizons_v2/processed/"
save_path = "../../../input/BES/dionisos_horizons_v2/gempy_format/"


# ---------------------------------------------- #
# File by file
# ---------------------------------------------- #

# Read DF
df_1 = pd.read_csv(path + "Horizon_1.csv")
df_2 = pd.read_csv(path + "Horizon_2.csv")
df_27 = pd.read_csv(path + "Horizon_27.csv")

# Merge the dfs
df = pd.concat([df_1, df_2, df_27], ignore_index=True)

# Scaler
scaler_x = MinMaxScaler(feature_range=(0, (df["X"].max() - df["X"].min())))
scaler_y = MinMaxScaler(feature_range=(0, (df["Y"].max() - df["Y"].min())))
df[["X"]] = scaler_x.fit_transform(df[["X"]])
df[["Y"]] = scaler_y.fit_transform(df[["Y"]])

# ---------------------------------------------- #
# Multiple files
# ---------------------------------------------- #

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(path) if file.endswith(".csv")]

# Sort the CSV files based on the numeric values in their filenames
sorted_csv_files = sorted(csv_files, key=lambda x: int(re.search(r"\d+", x).group()))

# Initialize an empty DataFrame to store the concatenated data
concatenated_df = pd.DataFrame()

# Loop through the CSV files and concatenate them
for file_name in sorted_csv_files:
    file_path = os.path.join(path, file_name)
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
output_file_path = os.path.join(save_path, "merged_horizons_sp_full.csv")
concatenated_df.to_csv(output_file_path, index=False)

concatenated_df.describe()
concatenated_df.info()

# ---------------------------------------------- #
# Reducing horizons size
# ---------------------------------------------- #

df_red = concatenated_df.copy()

# X and Y to int
df_red["X"] = df_red["X"].astype(int)
df_red["Y"] = df_red["Y"].astype(int)

list_x = [
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
    150000,
    155000,
    160000,
    165000,
    170000,
    175000,
]

df_appended = []
for x_value in list_x:
    ns_y = df_red["Y"].unique()
    new_df = df_red[(df_red["X"] == x_value) & (df_red["Y"].isin(ns_y))]
    df_appended.append(new_df)

final_df = pd.concat(df_appended, ignore_index=True)

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

# ---------------------------------------------- #
# Z adjustment
# ---------------------------------------------- #

z_adjustments = {
    "Horizon_2": -500,
    "Horizon_3": -1000,
    "Horizon_4": -1500,
    "Horizon_5": -2000,
    "Horizon_6": -2500,
    "Horizon_7": -3000,
    "Horizon_8": -3500,
    "Horizon_9": -4000,
    "Horizon_10": -4500,
    "Horizon_11": -5000,
    "Horizon_12": -5500,
    "Horizon_13": -6000,
    "Horizon_14": -6500,
    "Horizon_15": -7000,
    "Horizon_16": -7500,
    "Horizon_17": -8000,
    "Horizon_18": -8500,
    "Horizon_19": -9000,
    "Horizon_20": -9500,
    "Horizon_21": -10000,
    "Horizon_22": -10500,
    "Horizon_23": -11000,
    "Horizon_24": -11500,
    "Horizon_25": -12000,
    "Horizon_26": -12500,
    "Horizon_27": -13000,
    # Add more formations and adjustments as needed
}

# Apply Z adjustments based on the formation name
for formation, adjustment in z_adjustments.items():
    final_df.loc[final_df["formation"] == formation, "Z"] += adjustment

final_df.to_csv(save_path + "merged_sp_z_ajusted.csv", index=False)

final_df.describe()
