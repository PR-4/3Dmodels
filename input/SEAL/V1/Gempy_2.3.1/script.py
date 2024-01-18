import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# read .txt as lines
with open("../Gempy_2.3.1/surface_points_full_crono.txt") as f:
    lines = f.readlines()

# remove the newline character at the end of each line and split each line by comma
data = [line.rstrip().split(",") for line in lines]

data = [[item for item in line.rstrip().split(",") if item] for line in lines]

# create a DataFrame from the data
df = pd.DataFrame(data[1:], columns=data[0])

# convert X and Y to float
df["X"] = df["X"].astype(float)
df["Y"] = df["Y"].astype(float)

# convert Z to int
df["Z"] = df["Z"].astype(float).astype(int)

df.info()
df.describe()
df.to_csv("../Gempy_2.3.1/surface_points_full_crono.csv", index=False)


scaler_x = MinMaxScaler(feature_range=(0, df["X"].max() - df["X"].min()))
scaler_y = MinMaxScaler(feature_range=(0, df["Y"].max() - df["Y"].min()))

df[["X"]] = scaler_x.fit_transform(df[["X"]])
df[["Y"]] = scaler_y.fit_transform(df[["Y"]])

df["X"] = df["X"].round(1)
df["Y"] = df["Y"].round(1)

df["X"].min()
df["X"].max()
df["Y"].min()
df["Y"].max()
df["Z"].min()
df["Z"].max()

# Salvar em csv
df.to_csv("../Gempy_2.3.1/surface_points_full_crono_scaled.csv", index=False)


# print the DataFrame
print(df)
