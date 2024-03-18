import os
from tkinter import font
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors

model = "StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54.nc"
model_n = os.path.splitext(model)[0]
path_model = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/"
fn_results = model_n + "_results"
path_output = os.path.join(path_model, fn_results)
path_figs = os.path.join(path_output, "figs")
if not os.path.exists(path_figs):
    os.makedirs(path_figs)

# Model extent
path_csv_results = os.path.join(path_output, "csv_results")
path_extent = os.path.join(path_csv_results, "extent.csv")
extent = pd.read_csv(path_extent)

xmin = extent.xmin[0]
xmax = extent.xmax[0]
ymin = extent.ymin[0]
ymax = extent.ymax[0]
zmin = extent.zmin.min()
zmax = extent.zmax.max()

# Get cell spacing
path_spacing = os.path.join(path_csv_results, "spacing.csv")
spacing = pd.read_csv(path_spacing)
dx = spacing.x[0]
dy = spacing.y[0]
dz = spacing.z[0]

# Read data from netCDF file
path_nc = os.path.join(path_output, model)
spatial_data = xr.open_dataset(path_nc)

# Grid data
x = spatial_data["lon"][:].data
y = spatial_data["lat"][:].data
z = spatial_data["depth"][:].data

# Unit IDs
surface = np.round(spatial_data["lith_block"][:].data)

# Get model resolution
nrow, ncol, nlay = surface.shape

# Further model information
path_series = os.path.join(path_csv_results, "series.csv")
series = pd.read_csv(path_series)
# series = series[series.series != "Basement_series"]

# Surface points from surfaces
path_surf = os.path.join(path_csv_results, "surfaces.csv")
surfpoints = pd.read_csv(path_surf)
# surfpoints = surfpoints[surfpoints.surface != "basement"]

# Color
cmap = colors.ListedColormap(surfpoints.color.values)

# N Rows de superfície
nsurf = surfpoints.id.size


# Criando as pastas para salvar as figuras
path_figs_y = os.path.join(path_figs, "cross_section_y")
path_figs_x = os.path.join(path_figs, "cross_section_x")
path_figs_z = os.path.join(path_figs, "cross_section_z")

for path in [path_figs, path_figs_y, path_figs_x, path_figs_z]:
    if not os.path.exists(path):
        os.makedirs(path)


# Plot de uma seção específica
idx1 = pos1 = 99  # substitua pelo valor que quiser plotar (resolução do modelo, nesse caso de 0 a 100)
ypos1 = y[pos1]
xpos1 = x[pos1]
zpos1 = z[pos1]

# Crossplot Y single column
X, Z = np.meshgrid(x, z)
S = surface[:, idx1, :]
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.pcolormesh(X, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)
zticks = np.arange(zmin, np.ceil(zmax / 1000) * 1000 + 1, 1000)
ax.hlines(zticks, xmin=xmin, xmax=xmax, color="k", linewidth=0.1)
ax.set_yticks(zticks)
ax.set_ylim(zmin, zmax)
ax.set_xlabel("X [m]", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)
ax.set_title(
    "Cross-section, south-to-north, row no.: " + str(idx1) + " - northing, inline, y: " + str(ypos1),
    pad=10,
    fontsize=20,
)
plt.show()

# Crossplot X single column
Y, Z = np.meshgrid(y, z)
S = surface[:, :, idx1]
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.pcolormesh(Y, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)
zticks = np.arange(zmin, np.ceil(zmax / 1000) * 1000 + 1, 1000)
ax.hlines(zticks, xmin=ymin, xmax=ymax, color="k", linewidth=0.1)
ax.set_yticks(zticks)
ax.set_ylim(zmin, zmax)
ax.set_xlabel("Y [m]", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)
ax.set_title(
    "Cross-section, west-to-east row no.: " + str(idx1) + " - easting, xline, x: " + str(xpos1),
    pad=10,
    fontsize=20,
)
plt.show()

# Crossplot Z single column
X, Y = np.meshgrid(x, y)
S = surface[idx1, :, :]
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.pcolormesh(X, Y, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)
ax.set_xlabel("X [m]", fontsize=15)
ax.set_ylabel("Y [m]", fontsize=15)
ax.set_title("Cross-section, top-to-bottom, row no.: " + str(idx1) + " - depth, z: " + str(zpos1), pad=10, fontsize=20)
plt.show()

# Cross-sections Y
for idx, ypos in enumerate(y):
    # Create meshgrid
    X, Z = np.meshgrid(x, z)
    S = surface[:, idx, :]

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.pcolormesh(X, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)

    zticks = np.arange(zmin, np.ceil(zmax / 1000) * 1000 + 1, 1000)
    ax.set_xlabel("X [m]", fontsize=15)
    ax.set_ylabel("Depth [m]", fontsize=15)
    ax.set_title(
        "Cross-section, south-to-north, row no.: " + str(idx) + " - northing, inline, y: " + str(ypos),
        pad=10,
        fontsize=20,
    )

    fn = "cs_y_row-" + str(idx) + "_y-" + str(ypos) + ".png"
    figname = os.path.join(path_figs_y, fn)
    fig.savefig(figname, bbox_inches="tight", dpi=300)

    fig.clf()
    plt.close(fig)
    print("Plotted col " + str(idx) + " y " + str(ypos))


# Cross-sections X
for idx, xpos in enumerate(x):
    Y, Z = np.meshgrid(y, z)
    S = surface[:, :, idx]

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.pcolormesh(Y, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)

    zticks = np.arange(zmin, np.ceil(zmax / 1000) * 1000 + 1, 1000)
    ax.hlines(zticks, xmin=ymin, xmax=ymax, color="k", linewidth=0.1)
    ax.set_yticks(zticks)
    ax.set_ylim(zmin, zmax)

    ax.set_xlabel("Y [m]", fontsize=15)
    ax.set_ylabel("Depth [m]", fontsize=15)
    ax.set_title(
        "Cross-section, west-to-east row no.: " + str(idx) + " - easting, xline, x: " + str(xpos), pad=10, fontsize=20
    )
    fn = "cs_x_row-" + str(idx) + "_x-" + str(xpos) + ".png"
    figname = os.path.join(path_figs_x, fn)
    fig.savefig(figname, bbox_inches="tight", dpi=300)

    fig.clf()
    plt.close(fig)
    print("Plotted row " + str(idx) + " x " + str(xpos))


# Cross-sections Z
for idx, zpos in enumerate(z):
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    S = surface[idx, :, :]

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.pcolormesh(X, Y, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)

    ax.set_xlabel("X [m]", fontsize=15)
    ax.set_ylabel("Y [m]", fontsize=15)
    ax.set_title(
        "Cross-section, top-to-bottom, row no.: " + str(idx) + " - depth, z: " + str(zpos), pad=10, fontsize=20
    )

    fn = "cs_z_row-" + str(idx) + " z " + str(zpos) + ".png"
    figname = os.path.join(path_figs_z, "cs_z_depth-" + str(idx) + ".png")
    fig.savefig(figname, bbox_inches="tight", dpi=300)

    fig.clf()
    plt.close(fig)
    print("Plotted row " + str(idx) + " z " + str(zpos))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 3D Pyvista Grid
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

import pyvista as pv
import pyvistaqt as pvqt
import PVGeo
import matplotlib

# Crie um plotter
p = pvqt.BackgroundPlotter()

# Create a UniformGrid from the data
grid = pv.UniformGrid((nrow, ncol, nlay))
grid.origin = (xmin, ymin, zmin)
grid.spacing = (dx, dy, dz)
grid["lith_block"] = surface.ravel(order="C")

# Create a colormap from surfpoints.color.values
alpha = 1
colorsz = [matplotlib.colors.to_rgba(color, alpha) for color in surfpoints.color.values]
cmapz = matplotlib.colors.ListedColormap(colorsz)

# Plot the grid
p.add_mesh(grid, scalars="lith_block", cmap=cmapz, show_edges=False, lighting=True)

p.set_scale(zscale=5)
# p.show_grid(color="black", xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]", font_size=10, location="furthest")
p.show_bounds(font_size=10, location="furthest", color="black", xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
# p.set_axes_font_size(10)  # set the font size of the axes labels to 10
# p.set_scale(zscale=5)
p.show()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Visualizando o modelo por diferentes métodos
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Com GEMPY

import pyvista as pv
import pyvistaqt as pvqt
import gempy as gp

model_pkl = "StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54.pkl"
# Loading the model
geo_model = gp.load_model_pickle(path_model + model_pkl)

cn = 50
gp.plot_2d(geo_model, cell_number=cn, direction="y", show_data=False, ve=5)
gp.plot_2d(geo_model, cell_number=cn, direction="x", show_data=False, ve=5)
gp.plot_2d(geo_model, cell_number=cn, direction="z", show_data=False, ve=5)
p = gp.plot_3d(geo_model, plotter_type="background", show_data=False, show_lith=False, ve=5)


# Com pyvista e arquivos .csv salvos
import pandas as pd
import os

surface_path = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54_results/csv_results/"
path_model = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54_results/triangulated_surfaces/"
surfaces_color = pd.read_csv(os.path.join(surface_path, "surfaces.csv"))
colors = surfaces_color.color.values[:-1]

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
    p.add_mesh(qh_surf, color=colors[i])

p.set_scale(zscale=5)


# Com matplotlib e arquivos .csv salvos
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

surface_path = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54_results/csv_results/"
path_model = "../../../output/BES/StartBR/novos_testes/gempy_2.3.1/StratBR2GemPy_100x_100y_100z_2024-03-14-10-33-54_results/triangulated_surfaces/"
surfaces_color = pd.read_csv(os.path.join(surface_path, "surfaces.csv"))
colors = surfaces_color.color.values[:-1]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Loop through all the files
for i in range(13):  # assuming you have files from 0 to 12
    # Read the vertices and edges from the csv files
    vertices = pd.read_csv(os.path.join(path_model, f"vertices_id-{i}.csv")).values
    edges = pd.read_csv(os.path.join(path_model, f"edges_id-{i}.csv")).values

    # Plot the surface
    x, y, z = vertices.T
    ax.plot_trisurf(x, y, z, triangles=edges, color=colors[i])

plt.show()
