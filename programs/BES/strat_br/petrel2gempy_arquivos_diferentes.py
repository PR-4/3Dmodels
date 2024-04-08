# Bibliotecas necessárias
import pandas as pd
import re
import os
import xarray as xr
import glob
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


# -----------------------------#
# NC to DF to CSV
# -----------------------------#

path_nc = "../../../input/BES/stratbr/nc/raw/BESv2_Mapas_089.0-100.0Ma.nc"

dset = xr.open_dataset(path_nc)

dset.variables

lat = dset.northing_d
lat_df = lat.to_dataframe()
# lat_df.to_csv("../lat.csv")
lat_series = pd.Series(lat.values)

lon = dset.easting_d
lon_df = lon.to_dataframe()
lon_series = pd.Series(lon.values)

dept = dset.depth_0
dept_df = dept.to_dataframe().reset_index()
dept.shape

dept = dset.depth_0
dept_df = dept.to_dataframe().reset_index()
dept_df = dept_df[dept_df["time_d"] == 89]
dept_df.rename(columns={"depth_0": "dept"}, inplace=True)

lito = dset.litho
lito_df = lito.to_dataframe().reset_index()

estruct = dset.estrutural
estruct_df = estruct.to_dataframe().reset_index()
time_v = estruct_df["time_d"].unique()
for t in time_v:
    new_df = estruct_df[estruct_df["time_d"] == t]
    fn = f"estrut_{t}.csv"
    new_df.to_csv("../../../input/BES/stratbr/nc/estrutural/" + fn)
    print(f"Saved {fn}")

estruct = dset.estrutural
estruct_df = estruct.to_dataframe().reset_index()
time_v = estruct_df["time_d"].unique()
for t in time_v:
    new_df = estruct_df[estruct_df["time_d"] == t]
    # Reorder and rename the columns
    new_df = new_df[["easting_d", "northing_d", "estrutural", "time_d"]]
    new_df.columns = ["X", "Y", "Z", "formation"]
    # Add 'bes_' prefix to 'formation'
    new_df["formation"] = new_df["formation"].apply(lambda x: "bes_" + str(x))
    fn = f"estrut_{t}.csv"
    # Save the dataframe without the index
    new_df.to_csv("../../../input/BES/stratbr/nc/estrutural/" + fn, index=False)
    print(f"Saved {fn}")


# ------------------------------------------------------------#
# Para todos os arquivos .txt da pasta
# ------------------------------------------------------------#

path_save = "../../../input/BES/stratbr/testes_6/"
path_processed = "../../../input/BES/stratbr/testes_6/processed/"

csv_files = glob.glob("../../../input/BES/stratbr/nc/estrutural/*.csv")

csv_files = sorted(csv_files, key=lambda x: float(re.findall(r"\d+\.\d+", x)[0]))

dfs = [pd.read_csv(file, dtype={"formation": str}) for file in csv_files]

df_final = pd.concat(dfs)

df_final.info()

# Save DF full
df_final.to_csv(path_save + "sp_full.csv", index=False)

# ------------------------------------------------------------#
# Função pra scale
# ------------------------------------------------------------#


def escalar_coordenadas(df):
    from sklearn.preprocessing import MinMaxScaler

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

    return df


# ------------------------------------------------------------#
# Reduzir o número de pontos para cada formação
# ------------------------------------------------------------#


def reduzir_pontos(df, eixo, n_pontos=1000, scale=False):
    df = df.copy()

    if scale:
        df = escalar_coordenadas(df)

    if eixo.lower() == "x":
        df["X_rounded"] = np.round(df["X"] / n_pontos)
        df_reduzido = df.drop_duplicates(subset=["formation", "X_rounded"], keep="first")
        df_reduzido = df_reduzido.drop(columns=["X_rounded"])
    elif eixo.lower() == "y":
        df["Y_rounded"] = np.round(df["Y"] / n_pontos)
        df_reduzido = df.drop_duplicates(subset=["formation", "Y_rounded"], keep="first")
        df_reduzido = df_reduzido.drop(columns=["Y_rounded"])
    elif eixo.lower() == "xy":
        df["X_rounded"] = np.round(df["X"] / n_pontos)
        df["Y_rounded"] = np.round(df["Y"] / n_pontos)
        df_reduzido = df.drop_duplicates(subset=["formation", "X_rounded", "Y_rounded"], keep="first")
        df_reduzido = df_reduzido.drop(columns=["X_rounded", "Y_rounded"])
    else:
        raise ValueError("O eixo deve ser 'x', 'y' ou 'xy'")

    df_reduzido.reset_index(drop=True, inplace=True)

    return df_reduzido


# Chamar a função reduzir_pontos para reduzir o número de pontos no DataFrame df_final a cada 1000 metros nas coordenadas X e Y
valor_reduzido = 5000
scaled = False
scaled_str = "scaled" if scaled else "not_scaled"
print(scaled_str)  # Saída: "scaled"
df_reduzido = reduzir_pontos(df_final, "xy", valor_reduzido, scale=scaled)
print(df_reduzido["X"].min(), df_reduzido["X"].max())
print(df_reduzido["Y"].min(), df_reduzido["Y"].max())
print(df_reduzido["Z"].min(), df_reduzido["Z"].max())
print("")
print(df_reduzido)
print("")
# print(df_reduzido["formation"].unique())

# Salvar o DataFrame reduzido em um arquivo .csv
valor_red_str = str(valor_reduzido)
f_name = "sp_" + valor_red_str + "m" + "_" + scaled_str + ".csv"
df_reduzido.to_csv(path_processed + f_name, index=False)


# ------------------------------------------------------------#
# Diminuir profundidade de camadas (caso tenha overlap)
# ------------------------------------------------------------#


def adjust_z(df, valor):
    df = df.copy()
    formations = df["formation"].unique()
    for i, formation in enumerate(formations):
        if i > 0:  # Começa da segunda formação
            df.loc[df["formation"] == formation, "Z"] -= i * valor
    return df


n_ajustar = 300
df_ajustado = adjust_z(df_reduzido, n_ajustar)
f_ajustado = "sp_" + valor_red_str + "m" + "_" + scaled_str + "_" + str(n_ajustar) + "m_ajustado.csv"
df_ajustado.to_csv(dir_processed + f_ajustado, index=False)
print(df_ajustado["Y"].min(), df_ajustado["Y"].max())
print(df_ajustado["X"].min(), df_ajustado["X"].max())
print(df_ajustado["Z"].min(), df_ajustado["Z"].max())
