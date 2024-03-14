import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors

path = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/pickle_model/"
path_model = os.path.join(path, "BES-model")
path_save_fig = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/figures/"

# Model extent
path_results = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/pickle_model/output/"
path_extent = os.path.join(path_results, "extent.csv")
extent = pd.read_csv(path_extent)

xmin = extent.xmin[0]
xmax = extent.xmax[0]
ymin = extent.ymin[0]
ymax = extent.ymax[0]
zmin = extent.zmin.min()
zmax = extent.zmax.max()

# Get cell spacing
path_spacing = os.path.join(path_results, "spacing.csv")
spacing = pd.read_csv(path_spacing)
dx = spacing.x[0]
dy = spacing.y[0]

# Read data from netCDF file
path_nc = os.path.join(path, "BES-model.nc")
spatial_data = xr.open_dataset(path_nc)

# Grid data
x = spatial_data["easting"][:].data
y = spatial_data["northing"][:].data
z = spatial_data["depth"][:].data

# Unit IDs
surface = np.round(spatial_data["lith_block"][:].data)
# Get model resolution
nrow, ncol, nlay = surface.shape

# Further model information
path_series = os.path.join(path_results, "series.csv")
series = pd.read_csv(path_series)
# series = series[series.series != "Basement_series"]

# Surface points from surfaces
path_surf = os.path.join(path_results, "surfaces.csv")
surfpoints = pd.read_csv(path_surf)
# surfpoints = surfpoints[surfpoints.surface != "basement"]

# Color
cmap = colors.ListedColormap(surfpoints.color.values)
# Rows
nsurf = surfpoints.color.size

"""# Crossplot Y single column
X, Z = np.meshgrid(x, z)
S = surface[:, 99, :]  # replace 10 with the index of the y value you want
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.pcolormesh(X, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)
zticks = np.arange(zmin, np.ceil(zmax / 1000) * 1000 + 1, 1000)
ax.hlines(zticks, xmin=xmin, xmax=xmax, color="k", linewidth=0.1)
ax.set_yticks(zticks)
ax.set_ylim(zmin, zmax)
ax.set_xlabel("Easting", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)
ax.set_title("Cross-section, west-to-east, row no.: " + str(99) + ", northing: " + str(ypos), pad=10, fontsize=20)
plt.tight_layout()
plt.show()

# Crossplot X single column
Y, Z = np.meshgrid(y, z)
S = surface[:, 99, :]  # replace 10 with the index of the y value you want
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.pcolormesh(Y, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)
zticks = np.arange(zmin, np.ceil(zmax / 1000) * 1000 + 1, 1000)
ax.hlines(zticks, xmin=ymin, xmax=ymax, color="k", linewidth=0.1)
ax.set_yticks(zticks)
ax.set_ylim(zmin, zmax)
ax.set_xlabel("Easting", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)
ax.set_title("Cross-section, west-to-east, row no.: " + str(99) + ", northing: " + str(99), pad=10, fontsize=20)
plt.tight_layout()
plt.show()

# Crossplot Z single column
X, Y = np.meshgrid(x, y)
S = surface[20, :, :]  # replace 99 with the index of the z value you want
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.pcolormesh(X, Y, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)
ax.set_xlabel("Easting", fontsize=15)
ax.set_ylabel("Northing", fontsize=15)
ax.set_title("Cross-section, top-to-bottom, depth: " + str(z[20]), pad=10, fontsize=20)
plt.tight_layout()
plt.show()"""

# Cross-sections Y
for idx, ypos in enumerate(y):
    # Create meshgrid
    X, Z = np.meshgrid(x, z)
    S = surface[:, idx, :]

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    ax.pcolormesh(X, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)

    zticks = np.arange(zmin, np.ceil(zmax / 1000) * 1000 + 1, 1000)

    ax.hlines(zticks, xmin=xmin, xmax=xmax, color="k", linewidth=0.1)

    ax.set_yticks(zticks)

    ax.set_ylim(zmin, zmax)

    ax.set_xlabel("Easting", fontsize=15)
    ax.set_ylabel("Depth [m]", fontsize=15)

    ax.set_title("Cross-section, west-to-east, row no.: " + str(idx) + ", northing: " + str(ypos), pad=10, fontsize=20)

    figname = os.path.join(path_save_fig, "cross_section_y", "cs_y_column-" + str(idx) + ".png")
    fig.savefig(figname, bbox_inches="tight", dpi=300)

    fig.clf()
    plt.close(fig)
    print("Plotted col " + str(idx))

# Cross-sections X
for idx, xpos in enumerate(x):
    Y, Z = np.meshgrid(y, z)
    S = surface[:, idx, :]

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    ax.pcolormesh(Y, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)

    zticks = np.arange(zmin, np.ceil(zmax / 1000) * 1000 + 1, 1000)

    ax.hlines(zticks, xmin=ymin, xmax=ymax, color="k", linewidth=0.1)

    ax.set_yticks(zticks)

    ax.set_ylim(zmin, zmax)

    ax.set_xlabel("Northing", fontsize=15)
    ax.set_ylabel("Depth [m]", fontsize=15)

    ax.set_title("Cross-section, south-to-north, row no.: " + str(idx) + ", easting: " + str(xpos), pad=10, fontsize=20)

    figname = os.path.join(path_save_fig, "cross_section_x", "cs_x_row-" + str(idx) + ".png")
    fig.savefig(figname, bbox_inches="tight", dpi=300)

    fig.clf()
    plt.close(fig)
    print("Plotted row " + str(idx))

# Cross-sections Z
for idx, zpos in enumerate(z):
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    S = surface[idx, :, :]

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    ax.pcolormesh(X, Y, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)

    ax.set_xlabel("Easting", fontsize=15)
    ax.set_ylabel("Northing", fontsize=15)

    ax.set_title("Cross-section, top-to-bottom, depth: " + str(zpos), pad=10, fontsize=20)

    figname = os.path.join(path_save_fig, "cross_section_z", "cs_z_depth-" + str(idx) + ".png")
    fig.savefig(figname, bbox_inches="tight", dpi=300)

    fig.clf()
    plt.close(fig)
    print("Plotted depth " + str(idx))


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Com GEMPY

import pyvista as pv
import pyvistaqt as pvqt
import gempy as gp

path_model = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/pickle_model/1-geo_model_higher_res.pkl"
# Loading the model
geo_model = gp.load_model_pickle(path_model)

p = gp.plot_3d(geo_model, plotter_type="background", show_data=False, show_lith=False, ve=5)


# Com pyvista e arquivos .csv salvos
import pandas as pd
import os

path_model = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/pickle_model/output/triangulated_surfaces/"

p = pvqt.BackgroundPlotter()

# Loop through all the files
for i in range(13):  # assuming you have files from 0 to 12
    # Read the vertices and edges from the csv files
    vertices = pd.read_csv(os.path.join(path_model, f"vertices_id-{i}.csv")).values
    edges = pd.read_csv(os.path.join(path_model, f"edges_id-{i}.csv")).values

    # Create the PolyData object
    qh_points = pv.PolyData(vertices, faces=np.insert(edges, 0, 3, axis=1).ravel(), n_faces=edges.shape[0])

    # Compute the Delaunay surface and add it to the plotter
    qh_surf = qh_points.delaunay_2d()
    p.add_mesh(qh_surf)

p.set_scale(zscale=5)


# Com matplotlib e arquivos .csv salvos
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

path_model = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/pickle_model/output/triangulated_surfaces/"

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Loop through all the files
for i in range(13):  # assuming you have files from 0 to 12
    # Read the vertices and edges from the csv files
    vertices = pd.read_csv(os.path.join(path_model, f"vertices_id-{i}.csv")).values
    edges = pd.read_csv(os.path.join(path_model, f"edges_id-{i}.csv")).values

    # Plot the surface
    x, y, z = vertices.T
    ax.plot_trisurf(x, y, z, triangles=edges, alpha=0.5)

plt.show()
