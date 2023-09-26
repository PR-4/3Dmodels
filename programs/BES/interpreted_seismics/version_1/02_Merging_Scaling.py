import glob
import numpy as np
import pandas as pd
import re
import os
from sklearn.preprocessing import MinMaxScaler


# Path files
path = (
    "../../../../input/BES/interpreted_seismics/version_1/sp_one_seis_rescaled_test.csv"
)

# Read the CSV file into a DataFrame
df = pd.read_csv(path)

# Reducing DF
df_red = df.copy()

list_x = [
    10009.0,
    11004.0,
    12001.0,
    13006.0,
    14007.0,
    15007.0,
    16012.0,
    17003.0,
    18002.0,
    19008.0,
    20007.0,
    21005.0,
    22010.0,
    22904.0,
    24009.0,
    25002.0,
    26000.0,
    27006.0,
    28010.0,
    29106.0,
    30007.0,
    31004.0,
    32007.0,
    33000.0,
    34005.0,
    34999.0,
]

df_appended = []
for x_value in list_x:
    new_df = df_red[(df_red["X"] == x_value)]
    df_appended.append(new_df)

final_df = pd.concat(df_appended, ignore_index=True)

# X, y, z to int
final_df["X"] = final_df["X"].astype(int)
final_df["Y"] = final_df["Y"].astype(int)
final_df["Z"] = final_df["Z"].astype(int)


final_df.describe()
final_df.info()

final_df.to_csv(
    "../../../../input/BES/interpreted_seismics/version_1/sp_reduced_points_v7.csv",
    index=False,
)
