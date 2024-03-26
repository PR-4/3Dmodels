# Bibliotecas necessárias
import pandas as pd
import re
import os

# ------------------------------------------------------------#
# Função para criar o DF no formato do GemPy a partir de um arquivo .txt
# ------------------------------------------------------------#


def criar_df(caminho_arquivo, skip_rows=None, neg=True):
    """
    Cria um DataFrame a partir de um arquivo de texto.

    Parâmetros:
    caminho_arquivo (str): O caminho para o arquivo de texto.
    skip_rows (int, opcional): O número de linhas a serem ignoradas no início do arquivo. Por padrão é None.
    neg (bool, opcional): Se True, não multiplica a coluna Z por -1. Por padrão é True.

    Retorna:
    df (pd.DataFrame): O DataFrame criado.
    """
    # ler arquivos
    with open(caminho_arquivo) as f:
        linhas = f.read().splitlines()

    if skip_rows is not None:
        linhas = linhas[skip_rows:]

    # dividir os valores e pegar as últimas três palavras de cada linha
    valores = [linha.split()[:3] for linha in linhas]

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

    df["Z"] = df["Z"].astype(float)
    if neg == False:
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
dir_path = "../../../input/BES/interpreted_seismics_2/horizontes/raw/"

# Caminho para salvar o arquivo intermediario em .csv
dir_interim = "../../../input/BES/interpreted_seismics_2/horizontes/"

# Caminho para salvar o arquivo final em .csv
dir_processed = "../../../input/BES/interpreted_seismics_2/horizontes/"

# Lista para armazenar os DataFrames
dfs = []

# Iterar sobre todos os arquivos no diretório
for filename in sorted(os.listdir(dir_path)):
    # Obter o caminho completo para o arquivo
    file_path = os.path.join(dir_path, filename)
    # Verificar se o caminho é para um arquivo e não para um diretório
    if os.path.isfile(file_path):
        # Criar o DataFrame e adicioná-lo à lista
        df = criar_df(file_path, 20)
        # Remover os números e o hífen do início do nome da formação
        df["formation"] = df["formation"].replace(r"^\d+-", "", regex=True)
        dfs.append(df)

# Concatenar todos os DataFrames
df_final = pd.concat(dfs)
print(df_final)
# print(df_final.describe())

# Save DF full
df_final.to_csv(dir_interim + "sp_full.csv", index=False)

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
    """
    Função para reduzir o número de pontos em um DataFrame a cada N metros nas coordenadas X, Y ou ambas.

    Parâmetros:
    df (pandas.DataFrame): DataFrame contendo os pontos.
    eixo (str): Eixo para o qual a redução deve ser aplicada. Deve ser 'x', 'y' ou 'xy'.
    n_pontos (int, opcional): Distância em metros entre os pontos após a redução. Por padrão é 1000.

    Retorna:
    df_reduzido (pandas.DataFrame): DataFrame contendo os pontos após a redução.
    """
    df = df.copy()

    if scale:
        df = escalar_coordenadas(df)

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
valor_reduzido = 1000
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
print(df_reduzido["formation"].unique())

# Salvar o DataFrame reduzido em um arquivo .csv
valor_red_str = str(valor_reduzido)
f_name = "sp_" + valor_red_str + "m" + "_" + scaled_str + ".csv"
df_reduzido.to_csv(dir_processed + f_name, index=False)

# ------------------------------------------------------------#
# Criando orientation points
# ------------------------------------------------------------#


def get_first_point(df, formation):
    first_point = df[df["formation"] == formation][["X", "Y", "Z"]].iloc[0]
    return first_point


def create_orientation(df, formations):
    # Initialize an empty list to store the data
    data = []

    # Loop over the formations
    for formation in formations:
        # Get the first point of the formation
        first_point = get_first_point(df, formation)

        # Append the data to the list
        data.append(
            {
                "X": first_point["X"],
                "Y": first_point["Y"].round(2),
                "Z": first_point["Z"].round(2),
                "azimuth": 0,
                "dip": 0,
                "polarity": 1,
                "formation": formation,
            }
        )

    # Create the DataFrame
    df_orientation = pd.DataFrame(data)

    return df_orientation


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
