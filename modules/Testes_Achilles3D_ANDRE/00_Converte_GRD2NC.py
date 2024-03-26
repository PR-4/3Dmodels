import xarray as xr
import glob
import numpy as np
import pandas as pd
from sys import platform
import os

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
df_testing_nc = lat.to_dataframe()


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

lito = dset.litho
lito_df = lito.to_dataframe().reset_index()
lito_df.to_csv("../lito.csv")

dept = dset.depth_0
dept_df = dept.to_dataframe().reset_index()
dept_df = dept_df[dept_df["time_d"] == 89]
dept_df.rename(columns={"depth_0": "dept"}, inplace=True)
dept_df.to_csv("../dept.csv")


estruct = dset.estrutural
estruct_df = estruct.to_dataframe().reset_index()
estruct_df = estruct_df[estruct_df["time_d"] == 89]
estruct_df.to_csv("../estruct.csv")


test_merge = pd.merge(dept_df, lito_df, left_index=True, right_index=True)
test_merge = test_merge.drop(columns=["time_l", "northing_l", "easting_l"])
# Sort DF by depth
test_merge.info()
test_merge = test_merge.sort_values(by="depth_0", ascending=False)
for dept in test_merge["depth_0"]:
    if dept >= -1000.0000:
        test_merge["formation"] = "top"
    elif dept < -1000.0000 and dept >= -2000.0000:
        test_merge["formation"] = "mid"
    else:
        test_merge["formation"] = "bot"


# Create a new column named formation and if depth_0 has 0 to -1000, the value of the row is top, if depth_0 has -1001 to -2000, the valeu of the row is mid, else the value of the row is bot.


test_merge
test_merge.to_csv("../test_gempy_stratbr.csv")


# Create a new DF with theses variables
df_test = pd.DataFrame(
    {
        "lat": lat.values,
        "lon": lon.values,
        "dept": dept.values,
        "lito": lito.values,
        "litoid": litoid.values,
    }
)
lat_series = pd.Series(lat.values)


# ----------------------------#
# Estudando a grade estrutural#
# ----------------------------#

import numpy as np
import scipy
import scipy.ndimage as ndimage


def find_Peaks(data_in, **kwargs):
    """Esta função procura picos na estrutura data_in baseado
    em valores de corte que são definidos por média e desvio padrão.
    Note que é possível controlar as janelas e os limites pelos argumentos
    de entrada, conforme descrito abaixo.
    """

    size = kwargs.get("size", 5)  # definição da janela padrão de busca

    func = kwargs.get("func", "stat")  # define os limites de corte por estatística.
    # a outra opção aqui é 'abs'
    threshold = kwargs.get("threshold", np.nan)  # se abs é definido, threshold também

    # note que é possível variar o zscore limite. O padrão é 2 x std
    zscore = kwargs.get("zscore", 2)

    if func == "stat":
        threshold = data_in.mean() + zscore * data_in.std()

    elif func == "abs":
        if np.isnan(threshold):
            print(f'Erro: Função "abs" escolhida mas threshold não foi definido')
            return np.nan, np.nan

        threshold = threshold  # neste caso, foi definido

    else:
        print(f"Erro: Função de corte desconhecida [{func}]")
        return np.nan, np.nan

    data_max = ndimage.maximum_filter(data_in, size)
    maxima = data_in == data_max
    data_min = ndimage.minimum_filter(data_in, size)
    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)

    return [np.array(x).astype(int), np.array(y).astype(int)]


# encontra os valores máximos com zscore acima de 3
estrutural_diff = dset.estrutural.isel(time_d=0) - dset.estrutural.isel(time_d=-1)
[xmax, ymax] = find_Peaks(estrutural_diff.values, zscore=2)

# outra maneira de fazer isso é mascarar valores com base em um filtro de mediana
im_filter = ndimage.median_filter(estrutural_diff.values, size=3)
mask = im_filter > np.quantile(
    im_filter.flatten(), 0.99
)  # extrai o valor de corte para 99% da grade

# plota os valores da grade estrutural (min/max)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=3, figsize=(21, 6))

estrutural_diff.plot(ax=ax[0], vmax=1000)
ax[0].plot(
    estrutural_diff.easting_d[xmax].values,
    estrutural_diff.northing_d[ymax].values,
    "or",
    mfc="none",
)
ax[0].set_title("Diferenças no campo estrutural (em metros)", fontsize=22)

# zoom sobre a área com problemas ao norte
estrutural_diff.sel(
    easting_d=slice(460000, 520000), northing_d=slice(7780000, 7799000)
).plot(ax=ax[1])
ax[1].plot(
    estrutural_diff.easting_d[xmax].values,
    estrutural_diff.northing_d[ymax].values,
    "or",
    mfc="none",
)
ax[1].set_title("zoom sobre a área norte", fontsize=22)

im = ax[2].imshow(mask, cmap=plt.cm.gray, origin="lower")
# fig.colorbar(im,ax=ax[2])
ax[2].set_title("Mascara Median Filter 3x3 (limite em 1%)", fontsize=22)

plt.tight_layout()
plt.show()
