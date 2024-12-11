import xarray as xr
import glob
import numpy as np
import pandas as pd
from sys import platform
import os
from sklearn.preprocessing import MinMaxScaler

# configurações iniciais
lista_prof = sorted(glob.glob("./Mapas_de_Profundidade/*.grd"))

# veja que uso `os.sep` para que isso funcione tanto no windows quanto no linux
top = lista_prof[0].split(os.sep)[-1]
base = lista_prof[-1].split(os.sep)[-1]


# O nome da simulação é a primeira string
simulacao = top.split("_")[0]

# extrai o array de tempo a partir da lista dos arquivos fornecidos
age = np.zeros(len(lista_prof))
for ii, fin in enumerate(lista_prof):
    cage = fin.replace(fin[0 : fin.find("_Descompactada_") + 15], "")
    age[ii] = float(cage.replace(".grd", ""))

dt = np.diff(age).mean()
tmin = age.min()
tmax = age.max()

print(f"Simulação:{simulacao} Idades:{tmin}-{tmax} Ma delta-t:{dt} Ma")


# Bloco 1 - Dimensões gerais das grades

file_prof = "./Mapas_de_Profundidade/" + top
# estes são os dados de profundidade #############################################
df_dummy = pd.read_csv(
    file_prof, skiprows=2, nrows=2, header=None, delim_whitespace=True
)

xmin_depth, xmax_depth = df_dummy.values[0, :]  # pega os valores min e max de x e y
ymin_depth, ymax_depth = df_dummy.values[1, :]

df_dummy = pd.read_csv(
    file_prof, skiprows=1, nrows=1, header=None, delim_whitespace=True
)
nx_depth, ny_depth = df_dummy.values[0, :]  # pega o número de dados em x e y

dx = (xmax_depth - xmin_depth) / (nx_depth - 1)
dy = (ymax_depth - ymin_depth) / (ny_depth - 1)

# Calcula o array de coordenadas para x (easting) e y (northing), lembrando que M = nx_depth-1 e N é ny_depth-1
easting_d = np.linspace(xmin_depth, xmax_depth, num=nx_depth)
northing_d = np.linspace(ymin_depth, ymax_depth, num=ny_depth)  # para depth

easting_l = np.linspace(xmin_depth + dx / 2, xmax_depth - dx / 2, num=nx_depth - 1)
northing_l = np.linspace(
    ymin_depth + dy / 2, ymax_depth - dy / 2, num=ny_depth - 1
)  # para lito

time_d = np.linspace(tmin, tmax, num=int((tmax - tmin) / dt + 1))
time_l = np.linspace(tmin + dt / 2, tmax - dt / 2, num=int((tmax - tmin) / dt))

print(
    f"Sumário da simulaçao estratigráfica {simulacao}:\n"
    f"dimensões da grade prof M:{len(easting_d)} x N:{len(northing_d)} x t:{len(time_d)}\n"
    f"dimensões da grade lito M:{len(easting_l)} x N:{len(northing_l)} x t:{len(time_l)}\n"
    f"resolução das grades:{dx} x {dy} metros\n"
)


# Bloco 2

# profundidade original, descompactada
depths_0 = np.empty((len(time_d), len(northing_d), len(easting_d)))

paleobats = np.empty((len(time_d), len(northing_d), len(easting_d)))
estrutural = np.empty((len(time_d), len(northing_d), len(easting_d)))
lithos = np.empty((len(time_l), len(northing_l), len(easting_l)))

for ii, itime in enumerate(time_d):
    age = str(f"{itime:.1f}").zfill(5)
    fname_prof = (
        f"./Mapas_de_Profundidade/{simulacao}_Profundidade_Descompactada_{age}.grd"
    )
    fname_paleobat = f"./Mapas_de_Paleobatimetria/{simulacao}_Paleobatimetria_{age}.grd"
    fname_estrutural = f"./Mapas_Estruturais/{simulacao}_Estrutural_{age}.grd"
    fname_litho = f"./Mapas_de_Litofacies/{simulacao}_Litofacies_{age}.grd"

    # leitura das profundidades
    df_depth = pd.read_csv(fname_prof, skiprows=5, header=None, delim_whitespace=True)
    depths_0[ii, :, :] = df_depth.values

    # leitura das paleobatimetrias
    df_paleobat = pd.read_csv(
        fname_paleobat, skiprows=5, header=None, delim_whitespace=True
    )
    paleobats[ii, :, :] = df_paleobat.values

    # leitura do estrutural
    df_estrutural = pd.read_csv(
        fname_estrutural, skiprows=5, header=None, delim_whitespace=True
    )
    estrutural[ii, :, :] = df_estrutural.values

    if ii < len(time_l):
        # leitura das litologias
        df_litho = pd.read_csv(
            fname_litho, skiprows=5, header=None, delim_whitespace=True
        )
        lithos[ii, :, :] = df_litho.values

# verificação de consistencia
print(
    f"Grades lidas:\n"
    f"Grade de Litologia var lithos:{np.shape(lithos)}\n"
    f"Grade de Profundidades var depths_0:{np.shape(depths_0)}\n"
    f"Grade de Paleobatimetrias var paleobats:{np.shape(paleobats)}\n"
    f"Grade Estrutural var estrutural:{np.shape(estrutural)}\n"
)

# agora o arquivo descritivo das litologias será lido para adicionar no dset
file_lithotab = f"./{simulacao}_litofacies.lfc"
df_lithotab = pd.read_csv(file_lithotab, skiprows=1, header=None)
df_lithotab.columns = [
    "id",
    "nome",
    "porosidade_ini",
    "porosidade_min",
    "fator_decai",
    "comp_r",
    "comp_g",
    "comp_b",
    "sand_max",
    "sand_min",
    "sand_avg",
    "density",
]
df_lithotab.head()

# Bloco 3 - construção da estrutura NetCDF

# passo 3 - criação do netcdf
import datetime

ct = f"{datetime.datetime.now()}"  # timestamp para os atributos

nlitho = np.arange(0, len(df_lithotab))

dset = xr.Dataset(
    data_vars=dict(
        depth_0=(
            ["time_d", "northing_d", "easting_d"],
            depths_0,
            {
                "units": "[m]",
                "description": "Profundidade da grade original (descompactada)",
            },
        ),
        paleobat=(
            ["time_d", "northing_d", "easting_d"],
            paleobats,
            {"units": "[m]", "description": "Paleobatimetria no tempo time_d em Ma"},
        ),
        estrutural=(
            ["time_d", "northing_d", "easting_d"],
            estrutural,
            {
                "units": "[m]",
                "description": "Prof. Estrutural Atual referente ao tempo time_d em Ma",
            },
        ),
        litho=(
            ["time_l", "northing_l", "easting_l"],
            lithos,
            {"units": "[id]", "description": "Código da fácie litológica"},
        ),
        lithoid=(
            ["nlitho"],
            df_lithotab.id.values,
            {"units": "none", "description": "Código da litologia"},
        ),
        lithoname=(
            ["nlitho"],
            df_lithotab.nome.values,
            {"units": "none", "description": "Nome da fácie litológica"},
        ),
        litho_poro_ini=(
            ["nlitho"],
            df_lithotab.porosidade_ini.values,
            {"units": "%", "description": "Porosidade initial (descompactado)"},
        ),
        litho_poro_min=(
            ["nlitho"],
            df_lithotab.porosidade_min.values,
            {"units": "%", "description": "Porosidade mínimas da fácie"},
        ),
        litho_fact_decay=(
            ["nlitho"],
            df_lithotab.fator_decai.values,
            {"units": "none", "description": "Fator de decaimento da porosidade"},
        ),
        litho_sand_max=(
            ["nlitho"],
            df_lithotab.sand_max.values,
            {"units": "%", "description": "Fração de areia máxima da fácie"},
        ),
        litho_sand_min=(
            ["nlitho"],
            df_lithotab.sand_max.values,
            {"units": "%", "description": "Fração de areia mínima da fácie"},
        ),
        litho_sand_avg=(
            ["nlitho"],
            df_lithotab.sand_avg.values,
            {"units": "%", "description": "Média da Fração de areia da fácie"},
        ),
        litho_density=(
            ["nlitho"],
            df_lithotab.density.values,
            {
                "units": "g.cm-3",
                "description": "Valor de densidade aparente (descompactado)",
            },
        ),
    ),
    coords=dict(
        time_d=(
            ["time_d"],
            time_d,
            {
                "units": "[Ma]",
                "description": "Idade do ponto de grade de Profundidades em Milhões de anos",
            },
        ),
        time_l=(
            ["time_l"],
            time_l,
            {
                "units": "[Ma]",
                "description": "Idade do ponto de grade de Litofácies em Milhões de anos",
            },
        ),
        easting_d=(
            ["easting_d"],
            easting_d,
            {
                "units": "[m]",
                "description": "Posição relativa do ponto de grade de Profundidades no presente (UTM East)",
            },
        ),
        northing_d=(
            ["northing_d"],
            northing_d,
            {
                "units": "[m]",
                "description": "Posição relativa do ponto de grade de Profundidades no presente (UTM North)",
            },
        ),
        easting_l=(
            ["easting_l"],
            easting_l,
            {
                "units": "[m]",
                "description": "Posição relativa do ponto de grade de Litofácies no presente (UTM East)",
            },
        ),
        northing_l=(
            ["northing_l"],
            northing_l,
            {
                "units": "[m]",
                "description": "Posição relativa do ponto de grade de Litofácies no presente (UTM North)",
            },
        ),
        nlitho=(
            ["nlitho"],
            nlitho,
            {"units": "none", "description": "Número de fácies liotológicas"},
        ),
    ),
    attrs=dict(
        description=f"Simulação {simulacao} depth_0 na idade da grade [em metros], litologia na idade [código]",
        creation=ct,
        simulation=simulacao,
    ),
)

dset.to_netcdf(path="../Mapas_StratBR_BESv2_test.nc", mode="w", engine="scipy")


# -----------------------------#
# NC to DF to CSV
# -----------------------------#

dset.variables
lat = dset.northing_d
lat_df = lat.to_dataframe()
lat_df.to_csv("../lat.csv")
lat_series = pd.Series(lat.values)
lat_series.to_csv("../lat_series.csv")
lon = dset.easting_d
lon_series = pd.Series(lon.values)
lon_df = lon.to_dataframe()
dept = dset.depth_0
dept_df = dept.to_dataframe().reset_index()
dept_df.to_csv("../dept.csv")
dept.shape

dept = dset.depth_0
dept_df = dept.to_dataframe().reset_index()
dept_df = dept_df[dept_df["time_d"] == 89]
dept_df.rename(columns={"depth_0": "dept"}, inplace=True)
dept_df.to_csv("../dept.csv")


# -----------------------------#
# Merging structural with lito
# -----------------------------#

lito = dset.litho
lito_df = lito.to_dataframe().reset_index()
time_l = lito_df["time_l"].unique()
for t in time_l:
    new_lito_df = lito_df[lito_df["time_l"] == t]
    fn = f"lito_{t}.csv"
    new_lito_df.to_csv("../data/lito/" + fn)
lito_df.to_csv("../lito.csv")


estruct = dset.estrutural
estruct_df = estruct.to_dataframe().reset_index()
time_v = estruct_df["time_d"].unique()
for t in time_v:
    new_df = estruct_df[estruct_df["time_d"] == t]
    fn = f"estrut_{t}.csv"
    new_df.to_csv("../data/estrutural/" + fn)
    print(f"Saved {fn}")


estructural_folder = "../data/estrutural/"
lito_folder = "../data/lito/"
output_folder = "../data/merged/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

estructural_files = [
    f"estrut_{time_val}.csv"
    for time_val in [
        89.0,
        89.5,
        90.0,
        90.5,
        91.0,
        91.5,
        92.0,
        92.5,
        93.0,
        93.5,
        94.0,
        94.5,
        95.0,
        95.5,
        96.0,
        96.5,
        97.0,
        97.5,
        98.0,
        98.5,
        99.0,
        99.5,
        100.0,
    ]
]
lito_files = [
    f"lito_{time_val}.csv"
    for time_val in [
        89.25,
        89.75,
        90.25,
        90.75,
        91.25,
        91.75,
        92.25,
        92.75,
        93.25,
        93.75,
        94.25,
        94.75,
        95.25,
        95.75,
        96.25,
        96.75,
        97.25,
        97.75,
        98.25,
        98.75,
        99.25,
        99.75,
    ]
]

for e_file, l_file in zip(estructural_files, lito_files):
    estructural_df = pd.read_csv(os.path.join(estructural_folder, e_file))
    lito_df = pd.read_csv(os.path.join(lito_folder, l_file))
    merged_df = pd.concat([estructural_df, lito_df], axis=1)
    merged_df = merged_df.drop(
        columns=["Unnamed: 0", "time_l", "northing_l", "easting_l"]
    )
    new_order = ["time_d", "easting_d", "northing_d", "estrutural", "litho"]
    merged_df = merged_df.reindex(columns=new_order)
    merged_df = merged_df.rename(
        columns={
            "time_d": "time",
            "easting_d": "X",
            "northing_d": "Y",
            "estrutural": "Z",
        }
    )

    output_file = os.path.join(output_folder, f"merged_{e_file}")
    merged_df.to_csv(output_file, index=False)
    print(f"Merged and saved {output_file}")

last_e_file = estructural_files[-1]
last_l_file = lito_files[-1]
last_structural_df = pd.read_csv(os.path.join(estructural_folder, last_e_file))
last_lito_df = pd.read_csv(os.path.join(lito_folder, last_l_file))
last_merged_df = pd.concat([last_structural_df, last_lito_df], axis=1)

last_output_file = os.path.join(output_folder, f"merged_{last_e_file}")
last_merged_df.to_csv(last_output_file, index=False)
print(f"Merged and saved {last_output_file}")

# ---------------------------------------------------------------------#
# --------------------- GEMPY FORMAT ROUTINE --------------------------#
# ---------------------------------------------------------------------#

# -----------------------------#
# Data Cleaning Full Grid
# -----------------------------#

df = pd.read_csv("../data/merged/merged_estrut_99.0.csv")  # Read csv
df["formation"] = "bes_99"  # Só uma formação
df.drop(columns=["time", "litho"], inplace=True)  # Drop time and litho columns

# Scaling X and Y
scaler_x = MinMaxScaler(
    feature_range=(0, (df["X"].max() - df["X"].min()))
)  # Define X scaler
scaler_y = MinMaxScaler(
    feature_range=(0, (df["Y"].max() - df["Y"].min()))
)  # Define Y scaler
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
