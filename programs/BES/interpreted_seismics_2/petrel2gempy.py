# Bibliotecas necessárias
import pandas as pd
import re
import os

# ------------------------------------------------------------#
# Função para criar o DF no formato do GemPy a partir de um arquivo .txt
# ------------------------------------------------------------#


def criar_df(caminho_arquivo):
    """
    Cria um DataFrame a partir de um arquivo de texto.

    Parâmetros:
    caminho_arquivo (str): O caminho para o arquivo de texto.

    Retorna:
    df (pd.DataFrame): O DataFrame criado.
    """
    # ler arquivos
    with open(caminho_arquivo) as f:
        linhas = f.read().splitlines()

    # dividir os valores e pegar as últimas três palavras de cada linha
    valores = [linha.split()[-3:] for linha in linhas]

    # criar os cabeçalhos
    cabecalho = ["X", "Y", "Z"]

    # Criar um dicionário vazio
    dados = {}

    # Criar o dicionário com o cabeçalho como chave e os valores como valores
    for cabecalho, valor in zip(cabecalho, zip(*valores)):
        dados[cabecalho] = valor

    # Criar um dataframe
    df = pd.DataFrame(dados)

    # adicionar coluna com um nome
    df["formation"] = os.path.basename(caminho_arquivo)

    # multiplicar a coluna Z por -1
    df["Z"] = df["Z"].astype(float) * -1

    # 2 decimais
    df["Z"] = df["Z"].round(2)

    # mudar as colunas X e Y para float e arredondar para 2 casas decimais
    df["X"] = df["X"].astype(float).round(3)
    df["Y"] = df["Y"].astype(float).round(3)

    return df


# path = "../../../input/BES/interpreted_seismics_2/raw/MAASTRICHTIANO"
# df = criar_df(path)
# print(df)

# ------------------------------------------------------------#
# Para todos os arquivos .txt da pasta
# ------------------------------------------------------------#


# Caminho para o diretório com os arquivos
dir_path = "../../../input/BES/interpreted_seismics_2/raw/"

# Caminho para salvar o arquivo final em .csv
dir_save = "../../../input/BES/interpreted_seismics_2/interim/"

# Lista para armazenar os DataFrames
dfs = []

# Iterar sobre todos os arquivos no diretório
for filename in sorted(os.listdir(dir_path)):
    # Obter o caminho completo para o arquivo
    file_path = os.path.join(dir_path, filename)
    # Criar o DataFrame e adicioná-lo à lista
    df = criar_df(file_path)
    # Remover os números e o hífen do início do nome da formação
    df["formation"] = df["formation"].replace(r"^\d+-", "", regex=True)
    dfs.append(df)

# Concatenar todos os DataFrames
df_final = pd.concat(dfs)
print(df_final)
# print(df_final.describe())

# Save DF full
path_save = "../../../input/BES/interpreted_seismics_2/gempy_format/"
df_final.to_csv(path_save + "surface_points_full.csv", index=False)

# ------------------------------------------------------------#
# Reduzir o número de pontos para cada formação
# ------------------------------------------------------------#


def reduzir_pontos(df, eixo, n_pontos=1000):
    """
    Função para reduzir o número de pontos em um DataFrame a cada N metros nas coordenadas X, Y ou ambas.

    Parâmetros:
    df (pandas.DataFrame): DataFrame contendo os pontos.
    eixo (str): Eixo para o qual a redução deve ser aplicada. Deve ser 'x', 'y' ou 'xy'.
    n_pontos (int, opcional): Distância em metros entre os pontos após a redução. Por padrão é 1000.

    Retorna:
    df_reduzido (pandas.DataFrame): DataFrame contendo os pontos após a redução.
    """
    # Verificar qual eixo foi escolhido
    if eixo.lower() == "x":
        # Arredondar os valores de X para o múltiplo de n_pontos mais próximo
        df["X_rounded"] = (df["X"] // n_pontos) * n_pontos
        # Remover os pontos duplicados com base na formação e no valor arredondado de X, mantendo apenas o primeiro ponto de cada grupo
        df_reduzido = df.drop_duplicates(subset=["formation", "X_rounded"], keep="first")
        # Remover a coluna X_rounded
        df_reduzido = df_reduzido.drop(columns=["X_rounded"])
    elif eixo.lower() == "y":
        # Arredondar os valores de Y para o múltiplo de n_pontos mais próximo
        df["Y_rounded"] = (df["Y"] // n_pontos) * n_pontos
        # Remover os pontos duplicados com base na formação e no valor arredondado de Y, mantendo apenas o primeiro ponto de cada grupo
        df_reduzido = df.drop_duplicates(subset=["formation", "Y_rounded"], keep="first")
        # Remover a coluna Y_rounded
        df_reduzido = df_reduzido.drop(columns=["Y_rounded"])
    elif eixo.lower() == "xy":
        # Arredondar os valores de X e Y para o múltiplo de n_pontos mais próximo
        df["X_rounded"] = (df["X"] // n_pontos) * n_pontos
        df["Y_rounded"] = (df["Y"] // n_pontos) * n_pontos
        # Remover os pontos duplicados com base na formação e nos valores arredondados de X e Y, mantendo apenas o primeiro ponto de cada grupo
        df_reduzido = df.drop_duplicates(subset=["formation", "X_rounded", "Y_rounded"], keep="first")
        # Remover as colunas X_rounded e Y_rounded
        df_reduzido = df_reduzido.drop(columns=["X_rounded", "Y_rounded"])
    else:
        # Se o eixo não for 'x', 'y' ou 'xy', levantar um erro
        raise ValueError("O eixo deve ser 'x', 'y' ou 'xy'")

    # Resetar o índice do DataFrame reduzido
    df_reduzido.reset_index(drop=True, inplace=True)

    return df_reduzido


# Chamar a função reduzir_pontos para reduzir o número de pontos no DataFrame df_final a cada 1000 metros nas coordenadas X e Y
valor_reduzido = 100
df_reduzido = reduzir_pontos(df_final, "xy", valor_reduzido)
df_reduzido.describe()
df_reduzido["Y"].min()
df_reduzido["Y"].max()
df_reduzido["X"].min()
df_reduzido["X"].max()

df_reduzido["formation"].unique()

# Salvar o DataFrame reduzido em um arquivo .csv
path_save = "../../../input/BES/interpreted_seismics_2/gempy_format/"
valor_red_str = str(valor_reduzido)
f_name = "surface_points_" + valor_red_str + "m.csv"
df_reduzido.to_csv(path_save + f_name, index=False)
