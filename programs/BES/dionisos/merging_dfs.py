import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# files path
path = "../../../input/BES/dionisos_horizons_v2/processed/"

df_1 = pd.read_csv(path + "Horizon_1.csv")
df_2 = pd.read_csv(path + "Horizon_2.csv")
df_27 = pd.read_csv(path + "Horizon_27.csv")

# Sum -2000 to the Z values
# df_27["Z"] = df_27["Z"] - 2000

# Merge the dfs
df = pd.concat([df_1, df_2, df_27], ignore_index=True)

# Scaler
scaler_x = MinMaxScaler(feature_range=(0, (df["X"].max() - df["X"].min())))
scaler_y = MinMaxScaler(feature_range=(0, (df["Y"].max() - df["Y"].min())))
df[["X"]] = scaler_x.fit_transform(df[["X"]])
df[["Y"]] = scaler_y.fit_transform(df[["Y"]])

# Info
df.describe()

# Save the dataframe to a csv file
save_path = "../../../input/BES/dionisos_horizons_v2/"
df.to_csv(save_path + "Horizons_1_2_27_merged_test.csv", index=False)
