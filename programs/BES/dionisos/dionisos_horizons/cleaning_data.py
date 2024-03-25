import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# files path
path = "../../../input/BES/dionisos_horizons_v2/raw/"

# Read txt file and read line after the 9th line
with open(path + "Horizon 1") as f:
    lines = f.readlines()[9:]

# Create a list with the data
data = []
for line in lines:
    data.append(line.split())

# Create a dataframe with the data
df = pd.DataFrame(data, columns=["X", "Y", "Z"])

# Convert the data to float
df["X"] = df["X"].astype(float)
df["Y"] = df["Y"].astype(float)
df["Z"] = df["Z"].astype(float)

# Convert the data to int
df["X"] = df["X"].astype(int)
df["Y"] = df["Y"].astype(int)
df["Z"] = df["Z"].astype(int)

# Create a new column with the name of the horizon
df["formation"] = "Horizon_1"

# Save the dataframe to a csv file
save_path = "../../../input/BES/dionisos_horizons_v2/processed/"
df.to_csv(save_path + "Horizon_1.csv", index=False)

# -------------------------
# Multiple files

path = "../../../input/BES/dionisos_horizons_v2/raw/"  # Path to the txt files
save_path = (
    "../../../input/BES/dionisos_horizons_v2/processed/"  # Path to save the csv files
)
horizon_files = [
    f for f in os.listdir(path) if f.startswith("Horizon")
]  # List all the files that start with "Horizon"

processed_files = 0

# Iterate through each Horizon file
for fn in horizon_files:
    # Read txt file and read line after the 9th line
    with open(os.path.join(path, fn)) as f:
        lines = f.readlines()[9:]

    # Create a list with the data
    data = []
    for line in lines:
        data.append(line.split())

    # Create a dataframe with the data
    df = pd.DataFrame(data, columns=["X", "Y", "Z"])

    # Convert the data to float
    df["X"] = df["X"].astype(float)
    df["Y"] = df["Y"].astype(float)
    df["Z"] = df["Z"].astype(float)

    # Convert the data to int
    df["X"] = df["X"].astype(int)
    df["Y"] = df["Y"].astype(int)
    df["Z"] = df["Z"].astype(int)

    # Create a new column with the name of the horizon
    horizon_name = fn.replace(" ", "_").replace(".txt", "").strip()
    df["formation"] = horizon_name

    # Save the dataframe to a csv file
    csv_filename = os.path.join(save_path, horizon_name + ".csv")
    df.to_csv(csv_filename, index=False)

    # Increment the counter
    processed_files += 1

# Print the number of processed files
print(f"Processed and saved {processed_files} files.")
