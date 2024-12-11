import pandas as pd
import re
import os
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime

def criar_df(caminho_arquivo):
    """
    Cria um DataFrame a partir de um arquivo de texto com coordenadas X, Y, Z.

    Parâmetros:
    caminho_arquivo (str): O caminho para o arquivo de texto.

    Retorna:
    pd.DataFrame: O DataFrame criado com colunas X, Y, Z e formation.
    """
    with open(caminho_arquivo, encoding='utf-8') as f:
        linhas = f.read().splitlines()

    valores = [linha.split()[-3:] for linha in linhas]
    
    # Criar DataFrame diretamente dos valores
    df = pd.DataFrame(valores, columns=["X", "Y", "Z"])
    
    # Converter colunas para float e fazer operações
    df = df.astype(float)
    df["Z"] = -df["Z"]  # Multiplicar Z por -1
    
    # Arredondar valores
    df = df.round({"X": 3, "Y": 3, "Z": 2})
    
    # Adicionar nome do arquivo como formação
    df["formation"] = os.path.basename(caminho_arquivo)
    
    return df

def reduzir_pontos(df, eixo, n_pontos=1000):
    """
    Reduz o número de pontos em um DataFrame a cada N metros nas coordenadas especificadas.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os pontos.
    eixo (str): Eixo para redução ('x', 'y' ou 'xy').
    n_pontos (int): Distância em metros entre os pontos após a redução.

    Retorna:
    pd.DataFrame: DataFrame com pontos reduzidos.
    """
    df = df.copy()
    
    if eixo.lower() not in ['x', 'y', 'xy']:
        raise ValueError("O eixo deve ser 'x', 'y' ou 'xy'")
    
    rounded_cols = []
    if 'x' in eixo.lower():
        df["X_rounded"] = (df["X"] // n_pontos) * n_pontos
        rounded_cols.append("X_rounded")
    
    if 'y' in eixo.lower():
        df["Y_rounded"] = (df["Y"] // n_pontos) * n_pontos
        rounded_cols.append("Y_rounded")
    
    subset = ["formation"] + rounded_cols
    df_reduzido = df.drop_duplicates(subset=subset, keep="first")
    df_reduzido = df_reduzido.drop(columns=rounded_cols)
    
    return df_reduzido.reset_index(drop=True)

def escalar_coordenadas(df):
    """
    Escala as coordenadas X e Y mantendo as proporções originais.

    Parâmetros:
    df (pd.DataFrame): DataFrame com coordenadas X e Y.

    Retorna:
    pd.DataFrame: DataFrame com coordenadas escaladas.
    """
    df = df.copy()
    
    x_range = df["X"].max() - df["X"].min()
    y_range = df["Y"].max() - df["Y"].min()
    
    scaler_x = MinMaxScaler(feature_range=(0, x_range))
    scaler_y = MinMaxScaler(feature_range=(0, y_range))
    
    df[["X"]] = scaler_x.fit_transform(df[["X"]])
    df[["Y"]] = scaler_y.fit_transform(df[["Y"]])
    
    return df

def main():
    tempo_inicio = time.time()
    print(f"\nIniciando processamento em {datetime.now().strftime('%H:%M:%S')}")
    
    # Configurar caminhos
    dir_path = "../input/BES/interpreted_seismics_2/horizontes/raw/"
    dir_save = "../input/Santos_Basin/sismicas_interpretadas/horizontes_interpolados/"
    
    print(f"\nCriando diretório de saída: {dir_save}")
    os.makedirs(dir_save, exist_ok=True)
    
    # Processar arquivos
    print("\nProcessando arquivos...")
    dfs = []
    for filename in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            print(f"Processando arquivo: {filename}")
            df = criar_df(file_path)
            df["formation"] = df["formation"].replace(r"^\d+-", "", regex=True)
            dfs.append(df)
    
    tempo_leitura = time.time()
    print(f"\nTempo de leitura dos arquivos: {(tempo_leitura - tempo_inicio):.2f} segundos")
    
    # Concatenar DataFrames
    print("\nConcatenando DataFrames...")
    df_final = pd.concat(dfs, ignore_index=True)
    print(f"Total de pontos: {len(df_final):,}")
    print(f"Formações encontradas: {df_final['formation'].unique()}")
    
    tempo_concat = time.time()
    print(f"Tempo de concatenação: {(tempo_concat - tempo_leitura):.2f} segundos")
    
    # Salvar DataFrame completo
    print("\nSalvando DataFrame completo...")
    df_final.to_csv(os.path.join(dir_save, "surface_points_full.csv"), index=False)
    
    tempo_save_full = time.time()
    print(f"Tempo de salvamento do arquivo completo: {(tempo_save_full - tempo_concat):.2f} segundos")
    
    # Reduzir pontos
    valor_reduzido = 300
    print(f"\nReduzindo pontos para {valor_reduzido}m...")
    df_reduzido = reduzir_pontos(df_final, "xy", valor_reduzido)
    print(f"Pontos após redução: {len(df_reduzido):,}")
    
    tempo_reducao = time.time()
    print(f"Tempo de redução dos pontos: {(tempo_reducao - tempo_save_full):.2f} segundos")
    
    # Salvar DataFrame reduzido
    print("\nSalvando DataFrame reduzido...")
    filename = f"surface_points_{valor_reduzido}m.csv"
    df_reduzido.to_csv(os.path.join(dir_save, filename), index=False)
    
    tempo_final = time.time()
    print(f"\nTempo total de execução: {(tempo_final - tempo_inicio):.2f} segundos")
    print(f"Finalizado em {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()