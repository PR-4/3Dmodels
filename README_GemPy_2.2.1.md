# 3Dmodels
3D Geomodelling with Gempy

# Dependencies:
* Gempy - 2.2.11 (Windows)
* Gempy - 2.2.10 (Linux)
* Anaconda - https://www.anaconda.com

YML: gempy_2.2.1_windows.yml

# How to install on Windows:
- Create a new enviroment using gempy_windows.yml;
    - conda env create -f gempy_2.2.1_windows.yml

# How to install on Linux: 
- conda env create --name gempy
- conda activate gempy
- pip install python==3.7.11
- pip install gempy==2.2.10
- conda install gdal
- pip install pyvista==0.34.2
- pip install jupyter

# How to install on Linux with .yml
- Create a new enviroment using gempy_linux.yml
    - conda env create -f gempy_linux.yml

# Versioning

- Major Changes: 1.0.0 to 2.0.0
- Minor Changes: 1.0.0 to 1.1.0
- Fix/Patch: 1.0.0 to 1.0.1
- Pre-Release (Alpha/Beta): - 1.0.0-Beta

# Folders
- input: Input data. Create subfolder for each model.
- modules: Local for modules of the program
- output: Output data. Create subfolder for each model.
- programs: Program/notebook. Create subfolder for each model.
- references: References used to create the model. Create subfolder for each model.