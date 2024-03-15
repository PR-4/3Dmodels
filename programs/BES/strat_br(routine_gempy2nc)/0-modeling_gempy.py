# Dependencies
import time
import math
import gempy as gp
import pickle
import datetime

import warnings

warnings.filterwarnings("ignore")

start_time = time.time()

# Creating model object and indicating path
data_path = "../../../input/BES/stratbr_grid_v3/gempy_format/"
save_path_model = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/"

model_name = "StratBR2GemPy"
extent = [0, 179000, 0, 148000, -17500, 1000]
resolution = [10, 10, 10]

# Loading model, the extension, the resolution and importing the data
geo_model = gp.create_model("StratBR2GemPy_test_4")
gp.init_data(
    geo_model,
    extent=extent,
    resolution=resolution,
    path_i=data_path + "2_merged_sp_reduced_more_z_ajusted.csv",
    path_o=data_path + "orientations_points_v3_7.csv",
)

# Ordenando as surfaces (Estratigrafia (topo para base)) 1
gp.map_stack_to_surfaces(
    geo_model,
    {
        "Strat_Series": (
            "TOP",
            "bes_89",
            "bes_90",
            "bes_91",
            "bes_92",
            "bes_93",
            "bes_94",
            "bes_95",
            "bes_96",
            "bes_97",
            "bes_98",
            "bes_99",
            "bes_100",
        ),
        "Basement_series": ("basement",),
    },
)

gp.set_interpolator(
    geo_model,
    compile_theano=True,
    theano_optimizer="fast_run",  # fast_compile, fast_run
    dtype="float64",  # for model stability
)

sol = gp.compute_model(
    geo_model,
    compute_mesh=True,
)

# Seção transversal do modelo
gp.plot_2d(geo_model, direction="y", show_data=False, show_lith=True, ve=2, legend=True)
gp.plot_2d(geo_model, direction="x", show_data=False, show_lith=True, ve=5, legend=True)

# Ver os dados em 3D
gpv = gp.plot_3d(geo_model, image=False, plotter_type="background", show_data=False, ve=5)

# Save geo_model to a file
date_t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
fn = f"{model_name}_{resolution[0]}x_{resolution[1]}y_{resolution[2]}z_{date_t}.pkl"
with open(save_path_model + fn, "wb") as f:
    pickle.dump(geo_model, f)

end_time = time.time()
execution_time = math.ceil((end_time - start_time) / 60)

print(f"Execution time: {execution_time} minutes")
