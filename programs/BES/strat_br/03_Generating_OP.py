import pandas as pd
import numpy as np


path_processed = "../../../input/BES/stratbr_grid_v4/processed/"

# Carregue seus dados
surface_points_df = pd.read_csv(path_processed + "surfaces_points_scaled_merged_topo_all_ages_X_and_Y_reduced.csv")
orientation_points_df = pd.read_csv(path_processed + "orientations_points_example.csv")

# Filtrar pontos de superfície para a formação 'topo'
topo_surface_points = surface_points_df[surface_points_df["formation"] == "topo"]

# Crie um novo DataFrame para armazenar os pontos de orientação
new_orientation_points = pd.DataFrame(columns=orientation_points_df.columns)

# Obtenha os limites e o centro do grid
xmin, xmax = topo_surface_points["X"].min(), topo_surface_points["X"].max()
ymin, ymax = topo_surface_points["Y"].min(), topo_surface_points["Y"].max()
xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2

distances = np.sqrt((topo_surface_points["X"] - xcenter) ** 2 + (topo_surface_points["Y"] - ycenter) ** 2)
closest_index = distances.idxmin()

# Obtenha o valor z do ponto central
zcenter = topo_surface_points.loc[closest_index, "Z"]
zcenter -= 20  # Subtract 20 from z
zmin_x = topo_surface_points[topo_surface_points["X"] == xmin]["Z"].values[0] - 20
zmax_x = topo_surface_points[topo_surface_points["X"] == xmax]["Z"].values[0] - 20
zmin_y = topo_surface_points[topo_surface_points["Y"] == ymin]["Z"].values[0] - 20
zmax_y = topo_surface_points[topo_surface_points["Y"] == ymax]["Z"].values[0] - 20

# Get the rows for the boundary points
row_min_x_min_y = topo_surface_points[(topo_surface_points["X"] == xmin) & (topo_surface_points["Y"] == ymin)].iloc[0]
row_min_x_max_y = topo_surface_points[(topo_surface_points["X"] == xmin) & (topo_surface_points["Y"] == ymax)].iloc[0]
row_max_x_min_y = topo_surface_points[(topo_surface_points["X"] == xmax) & (topo_surface_points["Y"] == ymin)].iloc[0]
row_max_x_max_y = topo_surface_points[(topo_surface_points["X"] == xmax) & (topo_surface_points["Y"] == ymax)].iloc[0]
row_center = topo_surface_points.loc[closest_index]

# Create the 5 orientation points
new_points = pd.DataFrame(
    {
        "X": [row_center["X"], row_min_x_min_y["X"], row_min_x_max_y["X"], row_max_x_min_y["X"], row_max_x_max_y["X"]],
        "Y": [row_center["Y"], row_min_x_min_y["Y"], row_min_x_max_y["Y"], row_max_x_min_y["Y"], row_max_x_max_y["Y"]],
        "Z": [
            row_center["Z"] - 20,
            row_min_x_min_y["Z"] - 20,
            row_min_x_max_y["Z"] - 20,
            row_max_x_min_y["Z"] - 20,
            row_max_x_max_y["Z"] - 20,
        ],
        "azimuth": [0, 0, 0, 0, 0],
        "dip": [0, 0, 0, 0, 0],
        "polarity": [1, 1, 1, 1, 1],
        "formation": ["topo", "topo", "topo", "topo", "topo"],
    }
)

if new_orientation_points.empty:
    new_orientation_points = new_points
else:
    new_orientation_points = pd.concat([new_orientation_points, new_points])

# Salve o novo DataFrame para um arquivo CSV
new_orientation_points.to_csv(path_processed + "generated_orientations_points.csv", index=False)
