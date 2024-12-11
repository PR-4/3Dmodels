import pandas as pd
import re
import os
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime

def criar_df(caminho_arquivo):
    """
    Cria um DataFrame a partir de um arquivo de texto com colunas index, X, Y, Depth.

    Parâmetros:
    caminho_arquivo (str): O caminho para o arquivo de texto.

    Retorna:
    pd.DataFrame: O DataFrame criado com colunas X, Y, Z e formation.
    """
    df = pd.read_csv(caminho_arquivo, sep='\s+')
    
    if 'Depth' in df.columns:
        df = df.rename(columns={'Depth': 'Z'})
    
    df = df[['X', 'Y', 'Z']]
    
    df = df.astype(float)
    df = df.round({'X': 2, 'Y': 2, 'Z': 2})
    
    formation_name = os.path.splitext(os.path.basename(caminho_arquivo))[0].upper()
    df['formation'] = formation_name
    
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
    dir_path = "../input/Santos_Basin/sismicas_interpretadas/raw_surfaces/"
    dir_save = "../input/Santos_Basin/sismicas_interpretadas/horizontes_interpolados/itapema/"
    
    print(f"\nCriando diretório de saída: {dir_save}")
    os.makedirs(dir_save, exist_ok=True)
    
    # Processar arquivos
    print("\nProcessando arquivos...")
    dfs = []
    for filename in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, filename)
        if filename.endswith('.txt'):
            file_path = os.path.join(dir_path, filename)
            print(f"Processando arquivo: {filename}")
            df = criar_df(file_path)
            dfs.append(df)
            
    tempo_leitura = time.time()
    print(f"\nTempo de leitura dos arquivos: {(tempo_leitura - tempo_inicio):.2f} segundos")
    
    # Concatenar DataFrames
    print("\nConcatenando DataFrames...")
    df_final = pd.concat(dfs, ignore_index=True)
    print(f"Total de pontos: {len(df_final):}")
    print(f"Formações encontradas: {df_final['formation'].unique()}")
    
    tempo_concat = time.time()
    print(f"Tempo de concatenação: {(tempo_concat - tempo_leitura):.2f} segundos")
    
    # Salvar DataFrame completo
    print("\nSalvando DataFrame completo...")
    df_final.to_csv(os.path.join(dir_save, "surface_points_full.csv"), index=False)
    
    tempo_save_full = time.time()
    print(f"Tempo de salvamento do arquivo completo: {(tempo_save_full - tempo_concat):.2f} segundos")
    
    # Reduzir pontos
    valor_reduzido = 150
    print(f"\nReduzindo pontos para {valor_reduzido}m...")
    df_reduzido = reduzir_pontos(df_final, "xy", valor_reduzido)
    print(f"Pontos após redução: {len(df_reduzido):}")
    
    tempo_reducao = time.time()
    print(f"Tempo de redução dos pontos: {(tempo_reducao - tempo_save_full):.2f} segundos")
    
    # Salvar DataFrame reduzido
    print("\nSalvando DataFrame reduzido...")
    filename = f"surface_points_{valor_reduzido}m.csv"
    df_reduzido.to_csv(os.path.join(dir_save, filename), index=False)
    
    tempo_final = time.time()
    print(f"\nTempo total de execução: {(tempo_final - tempo_inicio):.2f} segundos")
    print(f"Finalizado em {datetime.now().strftime('%H:%M:%S')}")

    print(f"X min: {df_reduzido['X'].min()}")
    print(f"X max: {df_reduzido['X'].max()}")
    print(f"Y min: {df_reduzido['Y'].min()}")
    print(f"Y max: {df_reduzido['Y'].max()}")
    print(f"Z min: {df_reduzido['Z'].min()}")
    print(f"Z max: {df_reduzido['Z'].max()}")

if __name__ == "__main__":
    main()