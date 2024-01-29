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
# dir_path = "../../../../input/BES/interpreted_seismics/raw_data/"
dir_path = input("Qual é o path dos arquivos .txt? (Exemplo: ../../../../input/BES/interpreted_seismics/raw_data/)")
print("")

# Perguntar ao usuário se ele quer dar rescale nas coordenadas
rescale = input("Você deseja dar rescale nas coordenadas? (S/N) ").lower()
print("")

# Perguntar ao usuário se ele quer salvar o DataFrame
resposta = input("Você quer salvar o DF? (S/N) - Precisa da biblioteca sklearn no env ").lower()
print("")

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
print("")

if rescale in ["sim", "s"]:
    from sklearn.preprocessing import MinMaxScaler

    # Scaling X and Y
    scaler_x = MinMaxScaler(feature_range=(0, (df_final["X"].max() - df_final["X"].min())))  # Define X scaler
    scaler_y = MinMaxScaler(feature_range=(0, (df_final["Y"].max() - df_final["Y"].min())))  # Define Y scaler
    df_final[["X"]] = scaler_x.fit_transform(df_final[["X"]])  # Apply X scaler
    df_final[["Y"]] = scaler_y.fit_transform(df_final[["Y"]])  # Apply Y scaler

    print(df_final.describe())
    print("")

# Se a resposta for sim, perguntar o caminho para salvar o arquivo
if resposta in ["sim", "s"]:
    print("")
    dir_save = input(
        "Qual o path para salvar o resultado? (exemplo: ../../../../input/BES/interpreted_seismics/interim_data/)"
    )
    print("")
    # Se a resposta para dar rescale nas coordenadas for sim, salvar o DataFrame com o nome indicando que foi rescaled
    if rescale in ["sim", "s"]:
        df_final.to_csv(dir_save + "df_seis_gempy_format_rescaled.csv", index=False)
    else:
        df_final.to_csv(dir_save + "df_seis_gempy_format.csv", index=False)

    print("")
    print("Arquivo salvo com sucesso!")
