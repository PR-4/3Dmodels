import pandas as pd

# Lê o arquivo .txt e transforma em um DataFrame
df = pd.read_csv("../../input/Paper_1/raw/synthetic_surfaces_6.txt", sep=" ")

# Mostra as primeiras linhas do DataFrame
print(df.head())

print(df.describe())
print(df.info())

# Transformar X em int
df["X"] = df["X"].astype(int)
df["Y"] = df["Y"].round(4)
df["Z"] = df["Z"].round(4) * -1

# Rename Formation to formation
df.rename(columns={"Formation": "formation"}, inplace=True)

# Pegar os uniques de formation
formations = df["formation"].unique()
print(formations)

df_0 = df.assign(X=0)
df_500 = df.assign(X=500)
df_1500 = df.assign(X=1500)
df_2000 = df.assign(X=2000)


df_final = pd.concat([df, df_0, df_500, df_1500, df_2000])
print(df_final.describe())

# Salvar o arquivo
df_final.to_csv("../../input/Paper_1/processed/synthetic_surfaces_6.csv", index=False)
