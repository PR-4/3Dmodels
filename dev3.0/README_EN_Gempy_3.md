# How to install pre-version Gempy 3.0 on Windows:
- Create a new enviroment:
    - conda create -n gempy python=3.10
    - conda activate gempy
    - pip install gempy --pre
    - pip install pandas
    - pip install gempy_viewer
    - pip install gempy_plugins
    - pip install pyvista
    - pip install pooch (optional)
    - pip install scipy (optional)
    - pip install scikit-image (optional)
    - pip install torch torchvision torchaudio (TORCH PARA WINDOWS E CPU - https://pytorch.org/get-started/locally/)
    - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (PARA WINDOWS e CUDA 12.1- https://pytorch.org/get-started/locally/)
    - pip install notebook


https://pyinstaller.org/en/stable/
