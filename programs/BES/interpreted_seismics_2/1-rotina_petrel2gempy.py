# Bibliotecas necessárias
import pandas as pd
import re
import os
from function_petrel2gempyformat import (
    criar_df,
    create_final_df,
    escalar_coordenadas,
    reduzir_pontos,
    get_first_point,
    create_orientation,
    adjust_z,
)

# ------------------------------------------------------------#
# Paths
# ------------------------------------------------------------#


# Caminho para o diretório com os arquivos
dir_path = "../../../input/BES/interpreted_seismics_2/raw/"

# Caminho para salvar o arquivo intermediario em .csv
dir_interim = "../../../input/BES/interpreted_seismics_2/interim/"

# Caminho para salvar o arquivo final em .csv
dir_processed = "../../../input/BES/interpreted_seismics_2/processed/"

# ------------------------------------------------------------#
# Criando DF final
# ------------------------------------------------------------#

df_final = create_final_df(dir_path)

# ------------------------------------------------------------#
# Reduzir pontos e fazer scale
# ------------------------------------------------------------#

# Chamar a função reduzir_pontos para reduzir o número de pontos no DataFrame df_final a cada 1000 metros nas coordenadas X e Y
valor_reduzido = 500
scaled = False
scaled_str = "scaled" if scaled else "not_scaled"
print(scaled_str)  # Saída: "scaled"
df_reduzido = reduzir_pontos(df_final, "xy", valor_reduzido, scale=scaled)
print(df_reduzido["Y"].min(), df_reduzido["Y"].max())
print(df_reduzido["X"].min(), df_reduzido["X"].max())
print(df_reduzido["Z"].min(), df_reduzido["Z"].max())
print("")
print(df_reduzido)
print("")
print(df_reduzido["formation"].unique())

# Salvar o DataFrame reduzido em um arquivo .csv
valor_red_str = str(valor_reduzido)
f_name = "sp_" + valor_red_str + "m" + "_" + scaled_str + ".csv"
df_reduzido.to_csv(dir_processed + f_name, index=False)

# ------------------------------------------------------------#
# Criar pontos de orientação
# ------------------------------------------------------------#

# Usage
formations = ["top"]
# formations = df_reduzido["formation"].unique()
df_orientation = create_orientation(df_reduzido, formations)
formations_str = "_".join(formations)
op_fn = "op_" + formations_str + "_" + valor_red_str + "m" + "_" + scaled_str + ".csv"
df_orientation.to_csv(dir_processed + op_fn, index=False)


# ------------------------------------------------------------#
# Diminuir profundidade de camadas (caso tenha overlap)
# ------------------------------------------------------------#

n_ajustar = 500
df_ajustado = adjust_z(df_reduzido, n_ajustar)
f_ajustado = "sp_" + valor_red_str + "m" + "_" + scaled_str + "_" + str(n_ajustar) + "m_ajustado.csv"
df_ajustado.to_csv(dir_processed + f_ajustado, index=False)
print(df_ajustado["Y"].min(), df_ajustado["Y"].max())
print(df_ajustado["X"].min(), df_ajustado["X"].max())
print(df_ajustado["Z"].min(), df_ajustado["Z"].max())


# Usage
# formations = ["top"]
formations = df_ajustado["formation"].unique()
df_orientation = create_orientation(df_ajustado, formations)
formations_str = "_".join(formations)
op_fn = "op_" + formations_str + "_" + valor_red_str + "m" + "_" + scaled_str + "_" + str(n_ajustar) + "m_ajustado.csv"
df_orientation.to_csv(dir_processed + op_fn, index=False)
