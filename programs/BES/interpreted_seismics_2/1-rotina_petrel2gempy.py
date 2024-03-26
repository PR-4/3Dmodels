# Bibliotecas necessárias
import pandas as pd
import re
import os
from function_petrel2gempyformat import (
    criar_df,
    create_final_df,
    escalar_coordenadas,
    reduzir_pontos,
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

