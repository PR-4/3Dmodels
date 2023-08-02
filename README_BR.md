# 3Dmodels
Modelos geológicos em 3D com Gempy

# Dependências:
* Gempy - 2.2.11 (windows)
* Gempy - 2.2.10 (linux)
* Anaconda - https://www.anaconda.com

# Como instalar no Windows:
- Crie um enviroment usando o gempy_windows.yml;
    - conda env create -f gempy_windows.yml

# Como instalar no Linux: 
- conda env create --name gempy
- conda activate gempy
- pip install python==3.7.11
- pip install gempy==2.2.10
- conda install gdal
- pip install pyvista==0.34.2
- pip install jupyter

# Como instalar no linux com .yml
- Crie um enviroment usando o gempy_linux.yml
    - conda env create -f gempy_linux.yml

# Versionamento

- Principais Alterações: 1.0.0 to 2.0.0
- Pequenas Alterações: 1.0.0 to 1.1.0
- Correções/Patch: 1.0.0 to 1.0.1
- Pré-lançamento(Alfa/Beta) - 1.0.0-Beta