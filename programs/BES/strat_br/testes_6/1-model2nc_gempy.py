import os
import pandas as pd
import numpy as np
import gempy as gp
import xarray as xr
import glob
import copy

# Location of partial model results
path_model = "../../../../output/BES/StartBR/v6/"
pkl_files = glob.glob(path_model + "*.pkl")
model = pkl_files[0] if pkl_files else None
model_n = os.path.splitext(model)[0]
fn_results = model_n + "_results"
path_output = os.path.join(path_model, fn_results)

# Cria o diretório de saida se não existir
if not os.path.exists(path_output):
    os.makedirs(path_output)

# Loading the model
geo_model = gp.load_model_pickle(path_model + model)

# Export as csv: surfpoints_df, orientations_df, series_df, surfaces_df
surfpoints = copy.copy(geo_model.surface_points.df)
orientations = copy.copy(geo_model.orientations.df)
surfaces = copy.copy(geo_model.surfaces.df)
surfaces = surfaces.drop(columns=["vertices", "edges"])
series = copy.copy(geo_model.series.df)
series.insert(1, "series", series.index)

# Cria a pasta se não existir
csv_results_path = os.path.join(path_output, "csv_results")
if not os.path.exists(csv_results_path):
    os.makedirs(csv_results_path)

# The input data is stored in csv files since the data types of the columns
# differ. Therefore, they cannot be put into netCDF variables
path_sfp = os.path.join(csv_results_path, "surface_points.csv")
surfpoints.to_csv(path_sfp, index=False)

path_ori = os.path.join(csv_results_path, "orientations.csv")
orientations.to_csv(path_ori, index=False)

path_series = os.path.join(csv_results_path, "series.csv")
series.to_csv(path_series, index=False)

path_surf = os.path.join(csv_results_path, "surfaces.csv")
surfaces.to_csv(path_surf, index=False)

# Additional data -----
ad = geo_model.additional_data

kriging_data_k = copy.copy(ad.kriging_data.df)
kriging_data_k.insert(0, "Model_ID", model_n)
path_kriging = os.path.join(csv_results_path, "kriging_parameters.csv")
kriging_data_k.to_csv(path_kriging, index=False, mode="a", header=not os.path.exists(path_kriging))

rescaling_data_k = copy.copy(ad.rescaling_data.df)
rescaling_data_k.insert(0, "Model_ID", model_n)
path_rescale = os.path.join(csv_results_path, "rescaling_parameters.csv")
rescaling_data_k.to_csv(path_rescale, index=False, mode="a", header=not os.path.exists(path_rescale))

# Extract regular grid data -----
# Export as csv: resolution, spacing, extent
# Export with netCDF: x_rg, y_rg, z_rg, mask_topo
colnames_rg = ["x", "y", "z"]

resolution = geo_model.grid.regular_grid.resolution.reshape(1, -1)
resolution_df = pd.DataFrame(data=resolution, columns=colnames_rg)
path_res = os.path.join(csv_results_path, "resolution.csv")
resolution_df.to_csv(path_res, index=False, mode="a", header=not os.path.exists(path_res))

spacing = np.array(geo_model.grid.regular_grid.get_dx_dy_dz()).reshape(1, -1)
spacing_df = pd.DataFrame(data=spacing, columns=colnames_rg)
path_space = os.path.join(csv_results_path, "spacing.csv")
spacing_df.to_csv(path_space, index=False, mode="a", header=not os.path.exists(path_space))

extent = geo_model.grid.regular_grid.extent.reshape(1, -1)
extent_df = pd.DataFrame(data=extent, columns=["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])
path_extent = os.path.join(csv_results_path, "extent.csv")
extent_df.to_csv(path_extent, index=False, mode="a", header=not os.path.exists(path_extent))

# Extract elevation and sort descending
z_rg = geo_model.grid.regular_grid.z[::-1]

x_rg = geo_model.grid.regular_grid.x
y_rg = geo_model.grid.regular_grid.y

# Create the dimensions
nx = x_rg.size
ny = y_rg.size
nz = z_rg.size

# Create coords
coords = {"Model_ID": model_n, "nx": x_rg, "ny": y_rg, "nz": z_rg}

# Create the Dataset
ds = xr.Dataset(coords=coords)

# Add as variaveis ao ds
ds["lon"] = ("nx", x_rg)
ds["lat"] = ("ny", y_rg)
ds["depth"] = ("nz", z_rg)

# Add os atributos das variaveis
ds["lon"].attrs = {
    "long_name": "posição espacial longitudinal dos voxels",
    "unit": "metro",
    "var_desc": "CRS is EPSG:",
}

ds["lat"].attrs = {
    "long_name": "posição espacial latitudinal dos voxels",
    "unit": "metro",
    "var_desc": "CRS is EPSG:",
}

ds["depth"].attrs = {
    "long_name": "depth dos voxels",
    "unit": "metro",
    "var_desc": "depth é dada à unidade abaixo do nível do mar",
}

# Extract solution data -----
# # Export with netCDF: lith_block, scalar_field_surfpoints, scalar_field_matrix
# block_matrix, mask_matrix, mask_matrix_pad
points_rg = geo_model.solutions.grid.get_grid("regular")
n_series_active = series.index.size - 1
n_surfaces_active = surfaces.index.size - 1

lith_block_points = geo_model.solutions.lith_block
lith_block_k = np.full((resolution[0][0], resolution[0][1], resolution[0][2]), np.nan)

# -1 because last series and surface are inactive bceause it's basement
scalar_matrix_points = geo_model.solutions.scalar_field_matrix
scalar_field_matrix_k = np.full((n_series_active, resolution[0][0], resolution[0][1], resolution[0][2]), np.nan)

block_matrix_points = geo_model.solutions.block_matrix
block_matrix_k = np.full((n_series_active, resolution[0][0], resolution[0][1], resolution[0][2]), np.nan)

mask_matrix_points = geo_model.solutions.mask_matrix
mask_matrix_k = np.full((n_series_active, resolution[0][0], resolution[0][1], resolution[0][2]), np.nan)

mask_matrix_pad_k = np.array(geo_model.solutions.mask_matrix_pad)

for idx, (x, y, z) in enumerate(points_rg):
    is_x = x == x_rg
    is_y = y == y_rg
    is_z = z == z_rg

    lith_block_k[is_x, is_y, is_z] = lith_block_points[idx]

    for i in range(min(n_surfaces_active, n_series_active)):
        scalar_field_matrix_k[i, is_x, is_y, is_z] = scalar_matrix_points[i, idx]
        block_matrix_k[i, is_x, is_y, is_z] = block_matrix_points[i, 0, idx]
        mask_matrix_k[i, is_x, is_y, is_z] = mask_matrix_points[i, idx]

lith_block_k = np.swapaxes(lith_block_k, 0, 2)
scalar_field_matrix_k = np.swapaxes(scalar_field_matrix_k, 1, 3)
block_matrix_k = np.swapaxes(block_matrix_k, 1, 3)
mask_matrix_k = np.swapaxes(mask_matrix_k, 1, 3)
mask_matrix_pad_k = np.swapaxes(mask_matrix_pad_k, 1, 3)

# Scalar field value of interfaces
scalar_field_surfpoints = geo_model.solutions.scalar_field_at_surface_points
# scalar_field_surfpoints = (geo_model.solutions.scalar_field_at_surface_points)[np.newaxis, ...]

# Add the variables to the dataset
ds["lith_block"] = (("nz", "ny", "nx"), lith_block_k)
ds["scalar_field_matrix"] = (("n_active_series", "nz", "ny", "nx"), scalar_field_matrix_k)
ds["block_matrix"] = (("n_active_series", "nz", "ny", "nx"), block_matrix_k)
ds["mask_matrix"] = (("n_active_series", "nz", "ny", "nx"), mask_matrix_k)
ds["mask_matrix_pad"] = (("n_active_series", "nz", "ny", "nx"), mask_matrix_pad_k)
ds["scalar_field_at_surface_points"] = (("n_active_series", "n_active_surfaces"), scalar_field_surfpoints)

# Add attributes to the variables
ds["lith_block"].attrs = {
    "long_name": "ID values of the defined surfaces",
    "unit": "-",
    "var_desc": "values are float but the ID is the rounded integer value",
}

ds["scalar_field_matrix"].attrs = {
    "long_name": "Array with values of the scalar field",
    "unit": "-",
    "var_desc": "values of the scalar field at each location in the regular grid",
}

ds["block_matrix"].attrs = {
    "long_name": "Array holding interpolated ID values",
    "unit": "-",
    "var_desc": "array with all interpolated values for all series at each location in the regular grid",
}

ds["mask_matrix"].attrs = {
    "long_name": "Boolean array holding information for series combination",
    "unit": "-",
    "var_desc": "contains the logic to combine multiple series to obtain the final model at each location in the regular grid",
}

ds["mask_matrix_pad"].attrs = {
    "long_name": "Boolean array holding information for series combination (?)",
    "unit": "-",
    "var_desc": "mask matrix padded 2 block in order to guarantee that the layers intersect each other after marching cubes",
}

ds["scalar_field_at_surface_points"].attrs = {
    "long_name": "value of the scalar field at each interface",
    "unit": "-",
    "var_desc": "axis 0 is each series and axis 1 is each surface ordered by their id",
}

ds.to_netcdf(os.path.join(path_output, f"{model_n}.nc"))


vertices = [np.empty((0, 3), dtype=float)] * len(geo_model.solutions.vertices)
edges = [np.empty((0, 3), dtype=int)] * len(geo_model.solutions.edges)

path_surf = os.path.join(path_output, "triangulated_surfaces")
if not os.path.isdir(path_surf):
    os.mkdir(path_surf)

vertices_k = geo_model.solutions.vertices
edges_k = geo_model.solutions.edges

for idx in range(len(vertices)):
    # Append the spatial locations of the triangulation points.
    # Exception catches nan entries occuring when surface is not present
    # in the sub-model.
    try:
        vertices[idx] = np.append(vertices[idx], vertices_k[idx], axis=0)
    except ValueError:
        pass

    # Find maximum edge number so far and increase values by this, then
    # append
    try:
        # Maximum only valid if array already has an entry.
        max_edge = np.max(edges[idx]) if not edges[idx].size == 0 else 0
        edges_k[idx] += int(max_edge)
        edges[idx] = np.append(edges[idx], edges_k[idx], axis=0)
    except ValueError:
        pass

for idx, (vv, ee) in enumerate(zip(vertices, edges)):
    vert = pd.DataFrame(data=vv, columns=["x", "y", "z"])
    path_vert = os.path.join(path_surf, "vertices_id-" + str(idx) + ".csv")
    vert.to_csv(path_vert, index=False, mode="a", header=not os.path.exists(path_vert))

    edge = pd.DataFrame(data=ee, columns=["idx1", "idx2", "idx3"])
    path_edge = os.path.join(path_surf, "edges_id-" + str(idx) + ".csv")
    edge.to_csv(path_edge, index=False, mode="a", header=not os.path.exists(path_edge))

print(f"Done model id {model_n}, {path_model}")
