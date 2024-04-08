
# Installing Geological packages together

Contains Gempy, GemGIS, GSTools, PVGeo and Jupyter Notebook

YML: geomodeling_package_windows.yml

## GEMPY 2.3.1
- conda create -n NAME python =3.10
- conda activate NAME
- conda install -c conda-forge aesara
- pip install gempy
- pip install gstools
- pip install gemgis
- pip install PVGeo==2.1
- pip install notebook

## GEMPY 3.0 (Não funciona)
- conda create -n NAME python =3.10
- conda activate NAME
- pip install gemgis
- pip install gempy --pre
- pip install gempy_viewer
- pip install gempy_plugins
- pip install scipy (optional)
- pip install scikit-image (optional)
- pip install torch torchvision torchaudio (TORCH PARA WINDOWS E CPU - https://pytorch.org/get-started/locally/)
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (PARA WINDOWS e CUDA 12.1- https://pytorch.org/get-started/locally/)
- pip install gstools
- pip install notebook
- pip install PVGeo

