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
    df["formation"] = nome_formacao

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
# Selecionar região de interesse para grid
# ------------------------------------------------------------#


def filter_by_range(df, x_range, y_range):
    """
    Filter a DataFrame based on a range of X and Y values.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    x_range (tuple): A tuple specifying the minimum and maximum X values.
    y_range (tuple): A tuple specifying the minimum and maximum Y values.

    Returns:
    df (pd.DataFrame): The filtered DataFrame.
    """
    df = df[(df["X"] >= x_range[0]) & (df["X"] <= x_range[1])]
    df = df[(df["Y"] >= y_range[0]) & (df["Y"] <= y_range[1])]
    df.reset_index(drop=True, inplace=True)
    return df


# Or after concatenating all DataFrames
df_final = filter_by_range(df_final, (420000, 460000), (7770000.0, 7789833.0))


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
df_reduzido = reduzir_pontos(df_final, "xy", 1000)
df_reduzido.describe()
df_reduzido["Y"].min()
df_reduzido["Y"].max()
df_reduzido["X"].min()
df_reduzido["X"].max()

df_reduzido.to_csv(dir_save + "df_seis_gempy_format.csv", index=False)


# ------------------------------------------------------------#
# Criar pontos de orientação
# ------------------------------------------------------------#


def criar_pontos_de_orientacao(df, formacoes, pontos_de_orientacao):
    """
    Cria um DataFrame com pontos de orientação para cada formação.

    Parâmetros:
    df (pandas.DataFrame): DataFrame contendo os dados das formações.
    formacoes (list): Lista de formações únicas.
    pontos_de_orientacao (int): Número de pontos de orientação para criar para cada formação. Deve ser 1, 2, 3, 4 ou 5.

    Retorna:
    df_orientations (pandas.DataFrame): DataFrame contendo os pontos de orientação para cada formação.
    """
    df_orientations = pd.DataFrame()

    for formacao in formacoes:
        df_formacao = df[df["formation"] == formacao].sort_values("Z")
        if pontos_de_orientacao == 1:
            indices = [len(df_formacao) // 2]  # ponto médio
        elif pontos_de_orientacao == 2:
            indices = [
                (len(df_formacao) // 2 + 1) // 2,  # metade do ponto médio para o segundo ponto
                (len(df_formacao) // 2 + len(df_formacao) - 3) // 2,  # metade do ponto médio para o penúltimo ponto
            ]
        elif pontos_de_orientacao == 3:
            indices = [
                1,  # segundo ponto
                len(df_formacao) - 3,  # penúltimo ponto
                len(df_formacao) // 2,  # ponto médio
            ]
        elif pontos_de_orientacao == 4:
            indices = [
                1,  # segundo ponto
                len(df_formacao) - 3,  # penúltimo ponto
                (len(df_formacao) // 2 + 1) // 2,  # metade do ponto médio para o segundo ponto
                (len(df_formacao) // 2 + len(df_formacao) - 3) // 2,  # metade do ponto médio para o penúltimo ponto
            ]
        elif pontos_de_orientacao == 5:
            indices = [
                1,  # segundo ponto
                len(df_formacao) - 3,  # penúltimo ponto
                len(df_formacao) // 2,  # ponto médio
                (len(df_formacao) // 2 + 1) // 2,  # metade do ponto médio para o segundo ponto
                (len(df_formacao) // 2 + len(df_formacao) - 3) // 2,  # metade do ponto médio para o penúltimo ponto
            ]
        else:
            raise ValueError("pontos_de_orientacao deve ser 1, 2, 3, 4 ou 5")

        df_orientations = pd.concat([df_orientations, df_formacao.iloc[indices]])

    df_orientations["azimuth"] = 0
    df_orientations["dip"] = 0
    df_orientations["polarity"] = 1

    # Reordenar as colunas
    df_orientations = df_orientations[["X", "Y", "Z", "azimuth", "dip", "polarity", "formation"]]

    df_orientations.reset_index(drop=True, inplace=True)

    return df_orientations


# Rodar função para criar pontos de orientação
formacoes = df_reduzido["formation"].unique()
df_orientations = criar_pontos_de_orientacao(df_reduzido, formacoes, pontos_de_orientacao=5)
df_orientations.info()

# Salvar o DF em .csv
df_orientations.to_csv(dir_save + "orientation_points.csv", index=False)
"""
formations = ['h1', 'h2', 'h3']
df_orientations = create_orientation_points(df, formations, orienpoints=5)
"""
