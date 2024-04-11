# Lista de arquivos de imagem
import imageio
import os
import re

pathx = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54_results/figs/cross_section_x/"
pathy = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54_results/figs/cross_section_y/"
pathz = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54_results/figs/cross_section_z/"

# Lista de arquivos de imagem
imagensx = [os.path.join(pathx, i) for i in os.listdir(pathx) if i.endswith(".png")]
imagensy = [os.path.join(pathy, i) for i in os.listdir(pathy) if i.endswith(".png")]
imagensz = [os.path.join(pathz, i) for i in os.listdir(pathz) if i.endswith(".png")]

# Extrai o número de cada nome de arquivo
numerosx = [int(re.search(r"row-(\d+)", i).group(1)) for i in imagensx if re.search(r"row-(\d+)", i)]
numerosy = [int(re.search(r"row-(\d+)", i).group(1)) for i in imagensy if re.search(r"row-(\d+)", i)]
numerosz = [int(re.search(r"depth-(\d+)", i).group(1)) for i in imagensz if re.search(r"depth-(\d+)", i)]

# Ordena as imagens com base nos números
imagensx = [x for _, x in sorted(zip(numerosx, imagensx))]
imagensy = [x for _, x in sorted(zip(numerosy, imagensy))]
imagensz = [x for _, x in sorted(zip(numerosz, imagensz))]

# Crie uma lista de imagens
imagens_data = []
for imagem in imagensx:
    imagens_data.append(imageio.v2.imread(imagem))

imagens_data_y = []
for imagem in imagensy:
    imagens_data_y.append(imageio.v2.imread(imagem))

imagens_data_z = []
for imagem in imagensz:
    imagens_data_z.append(imageio.v2.imread(imagem))

# Caminho para salvar o gif
path_save = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54_results/figs/gif/"

# Caminho para salvar o gif
gif_path_x = os.path.join(path_save, "cross_plot_x.gif")
gif_path_y = os.path.join(path_save, "cross_plot_y.gif")
gif_path_z = os.path.join(path_save, "cross_plot_z.gif")

# Escreva todas as imagens como um gif
imageio.mimsave(gif_path_x, imagens_data, fps=5)
imageio.mimsave(gif_path_y, imagens_data_y, fps=5)
imageio.mimsave(gif_path_z, imagens_data_z, fps=5)
