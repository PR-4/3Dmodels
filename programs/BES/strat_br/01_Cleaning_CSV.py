import xarray as xr
import glob
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


# -----------------------------#
# NC to DF to CSV
# -----------------------------#
path_raw = "../../../input/BES/stratbr/new_tests_v5/raw/"
path_interim = "../../../input/BES/stratbr/new_tests_v5/interim/"
path_processed = "../../../input/BES/stratbr/new_tests_v5/processed/"


# ---------------------------------------------------------------------#
# --------------------- GEMPY FORMAT ROUTINE --------------------------#
# ---------------------------------------------------------------------#

# Sea Floor Creation
df_89 = pd.read_csv(path_raw + "merged_estrut_89.0.csv")
# df_top = df_89.copy().drop(columns=["time", "litho", "Z"])  # Drop time and litho columns
df_top = df_89.copy().drop(columns=["time", "litho"])  # Drop time and litho columns
# df_top["Z"] = 0  # Add Z column with 0 value
df_top["formation"] = "top"  # Add formation column with sf value
df_top["X"] = df_top["X"].astype(int)  # Convert float to int
df_top["Y"] = df_top["Y"].astype(int)  # Convert float to int
df_top["Z"] = df_top["Z"].astype(int)  # Convert float to int
df_top.describe()
df_top.info()
df_top.to_csv(path_processed + "1-top.csv", index=False)  # save to path_processed

# 89 Ma
df_89 = pd.read_csv(path_raw + "merged_estrut_89.0.csv")  # Read csv
# drop time and litho columns
df_89 = df_89.drop(columns=["time", "litho"])
# formation column with bes value
df_89["formation"] = "bes_89"
# Convert float to int
df_89["X"] = df_89["X"].astype(int)
df_89["Y"] = df_89["Y"].astype(int)
df_89["Z"] = df_89["Z"].astype(int)
# save
df_89.to_csv(path_processed + "bes_89.csv", index=False)

# 99 Ma
df_99 = pd.read_csv(path_raw + "merged_estrut_99.0.csv")  # Read csv
# drop time and litho columns
df_99 = df_99.drop(columns=["time", "litho"])
# formation column with bes value
df_99["formation"] = "bes_99"
# Convert float to int
df_99["X"] = df_99["X"].astype(int)
df_99["Y"] = df_99["Y"].astype(int)
df_99["Z"] = df_99["Z"].astype(int)
# Sum -3000 to Z
# df_99["Z"] = df_99["Z"] + -3000
# save
df_99.to_csv(path_processed + "bes_99.csv", index=False)

# ---------------------------------------------------------------------#
# Cleaning 100.0 Ma
df_100 = pd.read_csv(path_raw + "merged_estrut_100.0.csv")  # Read csv
# drop columns
# df_100 = df_100.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "time_l", "northing_l", "easting_l"])
# rename columns
# df_100.rename(
#    columns={"time_d": "time", "northing_d": "Y", "easting_d": "X", "estrutural": "Z"},
#    inplace=True,
# )
# Change Y and X column position
# df_100 = df_100[["time", "X", "Y", "Z", "litho"]]
df_100 = df_100.drop(columns=["time", "litho"])
df_100["X"] = df_100["X"].astype(int)
df_100["Y"] = df_100["Y"].astype(int)
df_100["Z"] = df_100["Z"].astype(int)
df_100["formation"] = "bes"
# save
df_100.to_csv(path_processed + "2-bes.csv", index=False)

# ---------------------------------------------------------------------#
# For all files
# ---------------------------------------------------------------------#
csv_files = [file for file in os.listdir(path_raw) if file.endswith(".csv")]

csv_files = sorted(csv_files, key=lambda x: float(x.split("_")[2].split(".")[0]))

for file_name in csv_files:
    # Read the CSV file
    df = pd.read_csv(os.path.join(path_raw, file_name))

    # Drop 'time' and 'litho' columns
    df = df.drop(columns=["time", "litho"])

    # Add 'formation' column with a specific value based on the file name
    formation_value = f"bes_{file_name.split('_')[2].split('.')[0]}"
    df["formation"] = formation_value

    # Convert 'X', 'Y', and 'Z' columns to integers
    df["X"] = df["X"].astype(int)
    df["Y"] = df["Y"].astype(int)
    df["Z"] = df["Z"].astype(int)

    # Save the modified DataFrame to a new CSV file
    new_file_name = f"bes_{file_name.split('_')[2].split('.')[0]}.csv"
    df.to_csv(os.path.join(path_interim, new_file_name), index=False)


# -----------------------------#
# Data Cleaning Full Grid
# -----------------------------#


df = pd.read_csv("../data/merged/merged_estrut_99.0.csv")  # Read csv
df["formation"] = "bes_99"  # Só uma formação
df.drop(columns=["time", "litho"], inplace=True)  # Drop time and litho columns

# Scaling X and Y
scaler_x = MinMaxScaler(feature_range=(0, (df["X"].max() - df["X"].min())))  # Define X scaler
scaler_y = MinMaxScaler(feature_range=(0, (df["Y"].max() - df["Y"].min())))  # Define Y scaler
df[["X"]] = scaler_x.fit_transform(df[["X"]])  # Apply X scaler
df[["Y"]] = scaler_y.fit_transform(df[["Y"]])  # Apply Y scaler
df["X"] = df["X"].astype(int)  # Convert X to int
df["Y"] = df["Y"].astype(int)  # Convert Y to int
df["Z"] = df["Z"].astype(int)  # Convert Z to int

df.to_csv("../data/gempy/sp_99_fullgrid.csv", index=False)  # Save csv

df.info()

# Concat 89 with 99
df_89 = pd.read_csv("../data/gempy/sp_89_fullgrid.csv")
df_99 = pd.read_csv("../data/gempy/sp_99_fullgrid.csv")
df_89n99 = pd.concat([df_89, df_99], ignore_index=True)
df_89n99.to_csv("../data/gempy/sp_89n99_fullgrid.csv", index=False)

# Adding new formation with -6000 Z
new_formation = df_89.copy()
new_formation["Z"] = new_formation["Z"] + -6000
new_formation = new_formation.drop(columns=["formation"])
new_formation["formation"] = "holder"
new_formation.describe()

df_new = pd.concat([df_89, new_formation], ignore_index=True)
df_new.to_csv("../data/gempy/sp_89_fullgrid_holder.csv", index=False)


# -----------------------------#
# Reducing points grid
# -----------------------------#
df = pd.read_csv("../data/gempy/sp_99_fullgrid.csv")  # Read csv
df.info()
df.describe()
# df["X"] = df["X"].astype(int)
# df["Y"] = df["Y"].astype(int)
# df["Z"] = df["Z"].astype(int)

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
    179000,
]
df_appended = []
for x_value in list_x:
    testing_df = df.copy()
    ns_y = testing_df["Y"].unique()
    new_df = testing_df[(testing_df["X"] == x_value) & (testing_df["Y"].isin(ns_y))]
    df_appended.append(new_df)

final_df = pd.concat(df_appended, ignore_index=True)
final_df.to_csv("../data/gempy/sp_99_lessgrid_1.csv", index=False)

# Concat 89 with 99

df_89 = pd.read_csv("../data/gempy/sp_89_lessgrid_1.csv")
df_99 = pd.read_csv("../data/gempy/sp_99_lessgrid_1.csv")
df_89n99 = pd.concat([df_89, df_99], ignore_index=True)
df_89n99.to_csv("../data/gempy/sp_89n99_lessgrid_1.csv", index=False)

# Adding new formation with -6000 Z
new_formation = df_89.copy()
new_formation["Z"] = new_formation["Z"] + -6000
new_formation = new_formation.drop(columns=["formation"])
new_formation["formation"] = "holder"
new_formation.describe()

df_new = pd.concat([df_89, new_formation], ignore_index=True)
df_new.to_csv("../data/gempy/sp_89_lessgrid_holder.csv", index=False)


# ------------------------------------------------------------------------ #
# ----------------------------- OTHER TESTS ------------------------------ #
# ------------------------------------------------------------------------ #


test_df = pd.read_csv("../data/merged/merged_estrut_89.0.csv")
test_df = test_df.sort_values(by="Z", ascending=False)
test_df["formation"] = "bes"  # Só uma formação
for index, row in test_df.iterrows():
    if row["Z"] <= 0 and row["Z"] >= -3000:
        test_df.loc[index, "formation"] = "T"
    elif row["Z"] < -3000 and row["Z"] >= -5000:
        test_df.loc[index, "formation"] = "M1"
    elif row["Z"] < -5000 and row["Z"] >= -7000:
        test_df.loc[index, "formation"] = "M2"
    elif row["Z"] < -7000 and row["Z"] >= -9000:
        test_df.loc[index, "formation"] = "M3"
    else:
        test_df.loc[index, "formation"] = "B"

test_df = test_df.drop(columns=["time", "litho"])
test_df.info()
test_df["formation"].unique()

# Apply scaling to 'X' and 'Y' columns
scaler_x = MinMaxScaler(feature_range=(0, (test_df["X"].max() - test_df["X"].min())))
scaler_y = MinMaxScaler(feature_range=(0, (test_df["Y"].max() - test_df["Y"].min())))
test_df[["X"]] = scaler_x.fit_transform(test_df[["X"]])
test_df[["Y"]] = scaler_y.fit_transform(test_df[["Y"]])
test_df["X"] = test_df["X"].astype(int)
test_df["Y"] = test_df["Y"].astype(int)
test_df["Z"] = test_df["Z"].astype(int)
test_df.describe()
test_df.info()
test_df.to_csv("../data/gempy/surface_points_89_full_1.csv", index=False)

# -----------------------------#
# EDA Merged Data Another test
# -----------------------------#
test_df = pd.read_csv("../data/merged/merged_estrut_89.0.csv")
test_df = test_df.sort_values(by="Z", ascending=False)
for index, row in test_df.iterrows():
    if row["Z"] <= 0 and row["Z"] >= -3000:
        test_df.loc[index, "formation"] = "TOP"
    elif row["Z"] < -8000 and row["Z"] >= -12000:
        test_df.loc[index, "formation"] = "BOT"
    else:
        test_df.loc[index, "formation"] = np.nan

test_df.dropna(inplace=True)
scaler_x = MinMaxScaler(feature_range=(0, (test_df["X"].max() - test_df["X"].min())))
scaler_y = MinMaxScaler(feature_range=(0, (test_df["Y"].max() - test_df["Y"].min())))
test_df[["X"]] = scaler_x.fit_transform(test_df[["X"]])
test_df[["Y"]] = scaler_y.fit_transform(test_df[["Y"]])
test_df.describe()
test_df.info()
test_df = test_df.drop(columns=["time", "litho"])
test_df.to_csv("../data/test_stratbr_grid_gempy_format_4.csv", index=False)


# -----------------------------#
# EDA Merged Data Another test
# -----------------------------#

test_df = pd.read_csv("../data/merged/merged_estrut_99.5.csv")
test_df.rename(columns={"litho": "formation"}, inplace=True)
test_df.drop(columns=["time"], inplace=True)
scaler_x = MinMaxScaler(feature_range=(0, (test_df["X"].max() - test_df["X"].min())))
scaler_y = MinMaxScaler(feature_range=(0, (test_df["Y"].max() - test_df["Y"].min())))
test_df[["X"]] = scaler_x.fit_transform(test_df[["X"]])
test_df[["Y"]] = scaler_y.fit_transform(test_df[["Y"]])
test_df.to_csv("../data/teste_surface_stratGempy.csv", index=False)


# -----------------------------#
# EDA Cleaning CSV
# -----------------------------#
test_df = pd.read_csv("../data/test_stratbr_grid_gempy_format_3.csv")
test_df["X"] = test_df["X"].astype(int)
test_df["Y"] = test_df["Y"].astype(int)
test_df["Z"] = test_df["Z"].astype(int)

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
    testing_df = test_df.copy()
    ns_y = testing_df["Y"].unique()
    new_df = testing_df[(testing_df["X"] == x_value) & (testing_df["Y"].isin(ns_y))]
    df_appended.append(new_df)

final_df = pd.concat(df_appended, ignore_index=True)

new_formation = final_df.copy()
new_formation["Z"] = new_formation["Z"] + -6000
new_formation = new_formation.drop(columns=["formation"])
new_formation["formation"] = "base"
new_formation.describe()


df_new = pd.concat([final_df, new_formation], ignore_index=True)
df_new.to_csv("../data/gempy/surface_points_89_less_1.csv", index=False)


# -----------------------------#
# 89 Ma Data Cleaning
# -----------------------------#

test_df = pd.read_csv("../data/merged/merged_estrut_89.0.csv")
test_df["formation"] = "bes"  # Só uma formação
test_df = test_df.drop(columns=["time", "litho"])
test_df["X"] = test_df["X"].astype(int)
test_df["Y"] = test_df["Y"].astype(int)
test_df["Z"] = test_df["Z"].astype(int)
test_df["formation"].unique()
test_df.info()

# Apply scaling to 'X' and 'Y' columns
scaler_x = MinMaxScaler(feature_range=(0, (test_df["X"].max() - test_df["X"].min())))
scaler_y = MinMaxScaler(feature_range=(0, (test_df["Y"].max() - test_df["Y"].min())))
test_df[["X"]] = scaler_x.fit_transform(test_df[["X"]])
test_df[["Y"]] = scaler_y.fit_transform(test_df[["Y"]])
test_df.describe()
test_df.info()
test_df.to_csv("../data/gempy/surface_points_89_full_1.csv", index=False)

test_df_2 = test_df.copy()
test_df_2["Z"] = test_df_2["Z"] + -6000
test_df_2 = test_df_2.drop(columns=["formation"])
test_df_2["formation"] = "base"
test_df_2.describe()

df_new = pd.concat([test_df, test_df_2], ignore_index=True)
df_new.to_csv("../data/gempy/surface_points_89_full_2.csv", index=False)


# -----------------------------#
# EDA Cleaning CSV
# -----------------------------#
test_df = pd.read_csv("../data/test_stratbr_grid_gempy_format_3.csv")
test_df["X"] = test_df["X"].astype(int)
test_df["Y"] = test_df["Y"].astype(int)
test_df["Z"] = test_df["Z"].astype(int)

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
    179000,
]
df_appended = []
for x_value in list_x:
    testing_df = test_df.copy()
    ns_y = testing_df["Y"].unique()
    new_df = testing_df[(testing_df["X"] == x_value) & (testing_df["Y"].isin(ns_y))]
    df_appended.append(new_df)

final_df = pd.concat(df_appended, ignore_index=True)

new_formation = final_df.copy()
new_formation["Z"] = new_formation["Z"] + -6000
new_formation = new_formation.drop(columns=["formation"])
new_formation["formation"] = "base"
new_formation.describe()


df_new = pd.concat([final_df, new_formation], ignore_index=True)
df_new.describe()
df_new.to_csv("../data/gempy/surface_points_89_less_2.csv", index=False)
