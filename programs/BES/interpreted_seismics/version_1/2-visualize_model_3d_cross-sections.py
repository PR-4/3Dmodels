import os
import numpy as np
import pandas as pd
import netCDF4 as ncdf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib

os.chdir(os.path.dirname(os.path.realpath(__file__)))

path_model = "../version_1/output_test/BES_model_seismic_v8.nc"

path_extent = "../version_1/output_test/extent.csv"
extent = pd.read_csv(path_extent)

xmin = extent.xmin[0]
xmax = extent.xmax[0]
ymin = extent.ymin[0]
ymax = extent.ymax[0]
zmin = extent.zmin.min()
zmax = extent.zmax.max()

# Get cell spacing
path_spacing = "../version_1/output_test/spacing.csv"
spacing = pd.read_csv(path_spacing)
dx = spacing.dx[0]
dy = spacing.dy[0]

# Read netCDF file
spatial_data = ncdf.Dataset(path_model, "r")

# Grid data
x = spatial_data["regular_grid"]["easting"][:].data
y = spatial_data["regular_grid"]["northing"][:].data
z = spatial_data["regular_grid"]["elevation"][:].data

# Unit IDs
surface = np.round(spatial_data["solution"]["lith_block"][:].data)
# Get model resolution
nrow, ncol, nlay = surface.shape

# Further model information
path_seriesdef = "../version_1/output_test/series.csv"
series_def = pd.read_csv(path_seriesdef)

# Plot cross-section
# Rows
nsurf = series_def.series.size - 1


import pickle
import pyvista as pv
import pyvistaqt as pvqt
import numpy as np
import gempy as gp

# Load geo_model from pickle file
with open("../version_1/BES_model_seismic_v8.pickle", "rb") as file:
    geo_model = pickle.load(file)

p = pvqt.BackgroundPlotter()

# Iterate over all surfaces
for k in range(len(geo_model.surfaces.df)):
    vertices, edges = geo_model.solutions.compute_all_surfaces(step_size=1)
    vertices = geo_model.solutions.vertices
    edges = geo_model.solutions.edges

    qh_points = pv.PolyData(vertices[k], faces=np.insert(edges[k], 0, 3, axis=1).ravel(), n_faces=edges[k].shape[0])
    qh_surf = qh_points.delaunay_2d()
    p.add_mesh(qh_surf)

p.set_scale(zscale=50)

p = gp.plot_3d(geo_model, plotter_type="background", show_data=False, show_lith=False, ve=50)
