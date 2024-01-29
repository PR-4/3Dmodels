# Bibliotecas necessárias
import pandas as pd
import re
import os

# ------------------------------------------------------------#
# Função para criar o DF no formato do GemPy a partir de um arquivo .txt
# ------------------------------------------------------------#


def criar_df(caminho_arquivo, nome_formacao):
    """
    Cria um DataFrame a partir de um arquivo de texto.

    Parâmetros:
    caminho_arquivo (str): O caminho para o arquivo de texto.
    nome_formacao (str): O nome da formação a ser adicionado ao DataFrame.

    Retorna:
    df (pd.DataFrame): O DataFrame criado.
    """
    # ler arquivos
    with open(caminho_arquivo) as f:
        linhas = f.read().splitlines()

    # substituir espaço por , para todas as linhas
    linhas = [re.sub(r"\s+", ",", linha) for linha in linhas]

    # criar os cabeçalhos
    cabecalho = ["A", "B", "C", "D", "X", "Y", "Z", "LINHA"]

    # dividir os valores
    valores = [linha.split(",") for linha in linhas]

    # Se o último valor estiver vazio, remova-o
    for i in range(len(valores)):
        if valores[i][-1] == "":
            valores[i] = valores[i][:-1]

    # Criar um dicionário vazio
    dados = {}

    # Criar o dicionário com o cabeçalho como chave e os valores como valores
    for cabecalho, valor in zip(cabecalho, zip(*valores)):
        dados[cabecalho] = valor

    # Criar um dataframe
    df = pd.DataFrame(dados)

    # descartar colunas
    df = df.drop(columns=["A", "B", "C", "D", "LINHA"])

    # adicionar coluna com um nome
    df["formacao"] = nome_formacao

    # multiplicar a coluna Z por -1
    df["Z"] = df["Z"].astype(float) * -1

    # 2 decimais
    df["Z"] = df["Z"].round(2)

    # mudar as colunas X e Y para float e arredondar para 2 casas decimais
    df["X"] = df["X"].astype(float).round(3)
    df["Y"] = df["Y"].astype(float).round(3)

    return df


# ------------------------------------------------------------#
# Teste em um arquivo
# ------------------------------------------------------------#


"""
# Teste um arquivo
file_path = "../../../../input/BES/interpreted_seismics/raw_data/h4.txt"
name_formation = "h4"

df_test = criar_df(file_path, name_formation)

df_test
"""

# ------------------------------------------------------------#
# Para todos os arquivos .txt da pasta
# ------------------------------------------------------------#

# Caminho para o diretório com os arquivos .txt
dir_path = "../../../../input/BES/interpreted_seismics/raw_data/"

# Caminho para salvar o arquivo final em .csv
dir_save = "../../../../input/BES/interpreted_seismics/interim_data/"

# Lista para armazenar os DataFrames
dfs = []

# Iterar sobre todos os arquivos no diretório
for filename in os.listdir(dir_path):
    # Verificar se o arquivo é um .txt
    if filename.endswith(".txt"):
        # Obter o caminho completo para o arquivo
        file_path = os.path.join(dir_path, filename)
        # Obter o nome da formação a partir do nome do arquivo
        name_formation = filename[:-4]  # remove a extensão .txt
        # Criar o DataFrame e adicioná-lo à lista
        df = criar_df(file_path, name_formation)
        dfs.append(df)

# Concatenar todos os DataFrames
df_final = pd.concat(dfs)

print(df_final.describe())

# Salvar o DF em .csv
# df_final.to_csv(dir_save + "df_seis_gempy_format.csv", index=False)

# ------------------------------------------------------------#
# Dar rescale nas coordenadas
# ------------------------------------------------------------#
from sklearn.preprocessing import MinMaxScaler


def escalar_coordenadas(df):
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

    # Imprimir as estatísticas descritivas do DataFrame
    print("\n", df.describe())

    return df


# Chamar a função escalar_coordenadas para escalar as coordenadas X e Y do DataFrame df_final
# df_final = escalar_coordenadas(df_final)

# ------------------------------------------------------------#
# Reduzir o número de pontos para cada formação
# ------------------------------------------------------------#


# TESTAR
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
        df_reduzido = df.drop_duplicates(subset=["formacao", "X_rounded"], keep="first")
        # Remover a coluna X_rounded
        df_reduzido = df_reduzido.drop(columns=["X_rounded"])
    elif eixo.lower() == "y":
        # Arredondar os valores de Y para o múltiplo de n_pontos mais próximo
        df["Y_rounded"] = (df["Y"] // n_pontos) * n_pontos
        # Remover os pontos duplicados com base na formação e no valor arredondado de Y, mantendo apenas o primeiro ponto de cada grupo
        df_reduzido = df.drop_duplicates(subset=["formacao", "Y_rounded"], keep="first")
        # Remover a coluna Y_rounded
        df_reduzido = df_reduzido.drop(columns=["Y_rounded"])
    elif eixo.lower() == "xy":
        # Arredondar os valores de X e Y para o múltiplo de n_pontos mais próximo
        df["X_rounded"] = (df["X"] // n_pontos) * n_pontos
        df["Y_rounded"] = (df["Y"] // n_pontos) * n_pontos
        # Remover os pontos duplicados com base na formação e nos valores arredondados de X e Y, mantendo apenas o primeiro ponto de cada grupo
        df_reduzido = df.drop_duplicates(subset=["formacao", "X_rounded", "Y_rounded"], keep="first")
        # Remover as colunas X_rounded e Y_rounded
        df_reduzido = df_reduzido.drop(columns=["X_rounded", "Y_rounded"])
    else:
        # Se o eixo não for 'x', 'y' ou 'xy', levantar um erro
        raise ValueError("O eixo deve ser 'x', 'y' ou 'xy'")

    # Resetar o índice do DataFrame reduzido
    df_reduzido.reset_index(drop=True, inplace=True)

    return df_reduzido


# Chamar a função reduzir_pontos para reduzir o número de pontos no DataFrame df_final a cada 1000 metros nas coordenadas X e Y
df_reduzido = reduzir_pontos(df_final, "xy", 1000)
