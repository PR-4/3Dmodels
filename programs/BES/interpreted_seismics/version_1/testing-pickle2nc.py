"""
gempy Model Export Script

Overview:
This script extracts geological model information generated using the GemPy library and exports it in various formats, including CSV and NetCDF. The output is organized into a specified directory structure.

Dependencies:
- gempy: GemPy is a geological modeling library.
- pandas: Data manipulation library.
- numpy: Numerical computing library.
- os: Operating system interface for directory and file operations.
- shutil: File operations utility.
- scipy: Scientific computing library.
- re: Regular expression operations.
- netCDF4: NetCDF data file format support.

Usage:
1. Input Model: Provide the path to the GemPy model pickle file (path_model).
2. Output Directory: Set the directory where the exported files will be stored (output_dir).
3. Topography Nodata Value: Specify the nodata value for topography (nodata_topo).
4. Create NetCDF File: A NetCDF file is created for storing gridded data. If a file with the same name already exists, it is removed.
5. Export CSV Files: Various CSV files are generated, including surface points, orientations, series, surfaces, kriging parameters, rescaling parameters, and grid information.
6. Export NetCDF Files: Gridded data such as resolution, spacing, extent, and topography are exported to the NetCDF file.
7. Export Solution Data: Lithology block, scalar field matrix, block matrix, mask matrix, and other solution data are exported to the NetCDF file.
8. Export Triangulated Surfaces: Triangulated surface vertices and edges are exported as CSV files.

Output Structure:
- CSV Files: Surface points, orientations, series, and surfaces CSV files are stored in the specified output directory.
- NetCDF File: A NetCDF file is created to store gridded data, and it includes information such as topography, resolution, spacing, and solution data.
- Triangulated Surfaces: Vertices and edges of triangulated surfaces are stored in a subdirectory named "triangulated_surfaces" within the output directory.

Notes:
- The script dynamically adapts to the model structure, and additional data exported depends on the specific GemPy model.

Example:
python 1-pickle2nc.py
"""

import gempy as gp
import pandas as pd
import numpy as np
import os
from shutil import copyfile
from scipy.spatial.distance import pdist, squareform
import re
import netCDF4 as ncdf

# ----------------------------------------
# PATHS HERE
# ----------------------------------------

# Location of the pickle file of the model
path_model = "BES_model_seismic_v8.pickle"

# Output directory
output_dir = "output_test/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ----------------------------------------
# Script here
# ----------------------------------------

# Load model
geo_model = gp.load_model_pickle(path_model)

grid = np.array(geo_model.solutions.grid.get_grid("regular"))

grid[:, 2] = np.flip(grid[:, 2])

ncfile = ncdf.Dataset("grid.nc", "w", format="NETCDF4")

# Create dimensions
ncfile.createDimension("x", grid.shape[0])
ncfile.createDimension("y", grid.shape[1])

grid_var = ncfile.createVariable("grid", "f8", ("x", "y"))

grid_var[:] = grid

# Close the file
ncfile.close()

import pyvista as pv
import pyvistaqt as pvqt

# Create a mesh from the grid points
mesh = pv.PolyData(grid)

# Create a plotter
plotter = pvqt.BackgroundPlotter()

# Add the mesh to the plotter
plotter.add_mesh(mesh, color="red")

# Show the plot
plotter.show()


import netCDF4
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt

# Load the NetCDF file
dataset = netCDF4.Dataset("../version_1/output_test/BES_model_seismic_v8.nc")

# Access the groups
regular_grid_group = dataset.groups["regular_grid"]
solution_group = dataset.groups["solution"]

# Print the variables in each group
print("Variables in regular_grid:", list(regular_grid_group.variables.keys()))
print("Variables in solution:", list(solution_group.variables.keys()))

# Get the coordinates
x = np.array(regular_grid_group.variables["easting"])
y = np.array(regular_grid_group.variables["northing"])
z = np.array(regular_grid_group.variables["elevation"])

# Create a 3D grid of coordinates
x, y, z = np.meshgrid(x, y, z, indexing="ij")

# Create a structured grid
grid = pv.StructuredGrid(x, y, z)

# Get the lith_block data
data = np.array(solution_group.variables["lith_block"])

# Set the values of the grid to your data
grid.point_data["lith_block"] = data.flatten(order="C")  # Flatten the array in column-major (Fortran-style) order

# Create a plotter
plotter = pvqt.BackgroundPlotter()

# Add the grid to the plotter
plotter.add_mesh(grid.outline(), color="k")
plotter.add_volume(grid, cmap="jet")

# Show the plot
plotter.show()


# Set topography nodata value
nodata_topo = -9999

# Extract and merge solution data
resolution = np.empty((0, 3), dtype=int)
spacing = np.empty((0, 3))
extent = np.empty((0, 6))

z_rg = np.empty(0)
colnames_rg = ["x", "y", "z"]

# Create netCDF file
path_ncdf = os.path.join(output_dir, geo_model.meta.project_name + ".nc")

# Remove existing NetCDF file if it exists
if os.path.isfile(path_ncdf):
    os.remove(path_ncdf)

# Create a new NetCDF file
out = ncdf.Dataset(path_ncdf, "w", format="NETCDF4")
file_exist = False


surfpoints = geo_model.surface_points.df
orientations = geo_model.orientations.df
surfaces = geo_model.surfaces.df
surfaces = surfaces.drop(columns=["vertices", "edges"])
series = geo_model.series.df
series.insert(1, "series", series.index)

surfpoints.to_csv(os.path.join(output_dir, "surface_points.csv"), index=False)
orientations.to_csv(os.path.join(output_dir, "orientations.csv"), index=False)
series.to_csv(os.path.join(output_dir, "series.csv"), index=False)
surfaces.to_csv(os.path.join(output_dir, "surfaces.csv"), index=False)

# Additional data -----
ad = geo_model.additional_data

# Krigin data
kriging_data = ad.kriging_data.df
kriging_data.insert(0, "Model_ID", 1)  # Assuming only one model
kriging_data.to_csv(os.path.join(output_dir, "kriging_parameters.csv"), index=False)

# Rescaling data
rescaling_data = ad.rescaling_data.df
rescaling_data.insert(0, "Model_ID", 1)  # Assuming only one model
rescaling_data.to_csv(os.path.join(output_dir, "rescaling_parameters.csv"), index=False)

# Regular grid data
# Export as csv: resolution, spacing, extent
# Export with netCDF: x_rg, y_rg, z_rg, mask_topo
resolution = geo_model.grid.regular_grid.resolution.reshape(1, -1)
resolution_df = pd.DataFrame(data=resolution, columns=["nx", "ny", "nz"])
resolution_df.to_csv(os.path.join(output_dir, "resolution.csv"), index=False)

spacing = np.array(geo_model.grid.regular_grid.get_dx_dy_dz()).reshape(1, -1)
spacing_df = pd.DataFrame(data=spacing, columns=["dx", "dy", "dz"])
spacing_df.to_csv(os.path.join(output_dir, "spacing.csv"), index=False)

extent = geo_model.grid.regular_grid.extent.reshape(1, -1)
extent_df = pd.DataFrame(data=extent, columns=["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])
extent_df.to_csv(os.path.join(output_dir, "extent.csv"), index=False)

# Extract elevation
z_rg = geo_model.grid.regular_grid.z[::-1]

# Extract x and y coordinates
x_rg = geo_model.grid.regular_grid.x
y_rg = geo_model.grid.regular_grid.y

# Create mask for topography
mask_topo = np.empty((resolution[0][0], resolution[0][1], 0), dtype=bool)

out.createDimension("nx", x_rg.size)
out.createDimension("ny", y_rg.size)
out.createDimension("nz", z_rg.size)

try:
    # Topography grid -----
    # Export with netCDF: x_tg, y_tg, topography

    # This has to be performed only once since the same topography raster is
    # passed to all sub-models.
    x_tg = geo_model.solutions.grid.topography.x
    y_tg = geo_model.solutions.grid.topography.y

    z_tg = geo_model.solutions.grid.get_grid("topography").T[2]
    # The points are ordered in a way, that for each x all y locations are iterated.
    # Thus given the shape of the array, the filling is rowwise.
    # Row 0 is the smallest y location, thus the one most southwards.
    topography = np.reshape(z_tg, (y_tg.size, x_tg.size), order="F")

    tg = out.createGroup("topography")

    tg.createDimension("nodata", 1)

    xvar_tg = tg.createVariable("easting", "f8", "nx")
    xvar_tg[:] = x_tg
    xvar_tg.setncatts(
        {
            "long_name": "longitudinal spatial position of raster cells",
            "unit": "meter",
            "var_desc": "CRS is EPSG:25832 (UTM 32N)",
        }
    )

    yvar_tg = tg.createVariable("northing", "f8", "ny")
    yvar_tg[:] = y_rg
    yvar_tg.setncatts(
        {
            "long_name": "latitudinal spatial position of raster cells",
            "unit": "meter",
            "var_desc": "CRS is EPSG:25832 (UTM 32N)",
        }
    )

    zvar_tg = tg.createVariable("elevation", "f8", ("ny", "nx"))
    zvar_tg[:] = topography
    zvar_tg.setncatts(
        {
            "long_name": "elevations of the raster cells of the DEM",
            "unit": "meter",
            "var_desc": "elevation is given referenced to unit above sea level",
        }
    )

    nd_tg = tg.createVariable("nodata", "f8", "nodata")
    nd_tg[:] = nodata_topo
    nd_tg.setncatts(
        {
            "long_name": "no data value of the DEM",
            "unit": "meter",
            "var_desc": "elevation is given referenced to unit above sea level",
        }
    )
except AttributeError:
    print("Don't have topo")
    pass

# Regular grid data -----
rg = out.createGroup("regular_grid")

xvar_rg = rg.createVariable("easting", "f8", "nx")
xvar_rg[:] = x_rg
xvar_rg.setncatts(
    {
        "long_name": "longitudinal spatial position of voxels",
        "unit": "meter",
        "var_desc": "CRS is EPSG:32724 (UTM zone 24S)",
    }
)
yvar_rg = rg.createVariable("northing", "f8", "ny")
yvar_rg[:] = y_rg
yvar_rg.setncatts(
    {
        "long_name": "latitudinal spatial position of voxels",
        "unit": "meter",
        "var_desc": "CRS is EPSG:32724 (UTM zone 24S)",
    }
)
zvar_rg = rg.createVariable("elevation", "f8", "nz")
zvar_rg.setncatts(
    {
        "long_name": "elevation voxels",
        "unit": "meter",
        "var_desc": "elevation is given referenced to unit above sea level",
    }
)
mt_tg = rg.createVariable("mask_topo", "b", ("nz", "ny", "nx"))
mt_tg.setncatts(
    {
        "long_name": "Logical values indicating which voxels are at elevations above topography",
        "unit": "-",
        "var_desc": "True indicates elevation above topography",
    }
)

# Extract solution data -----
# Export with netCDF: lith_block, scalar_field_surfpoints, scalar_field_matrix
# block_matrix, mask_matrix, mask_matrix_pad

points_rg = geo_model.solutions.grid.get_grid("regular")
lith_block_points = geo_model.solutions.lith_block
lith_block_k = np.full((resolution[0][0], resolution[0][1], z_rg.size), np.nan)

scalar_matrix_points = geo_model.solutions.scalar_field_matrix
n_series_active = series.index.size - 1
n_surfaces_active = surfaces.index.size - 1
scalar_field_matrix_k = np.full((n_series_active, resolution[0][0], resolution[0][1], z_rg.size), np.nan)

block_matrix_points = geo_model.solutions.block_matrix
block_matrix_k = np.full((n_series_active, resolution[0][0], resolution[0][1], z_rg.size), np.nan)

mask_matrix_points = geo_model.solutions.mask_matrix
mask_matrix_k = np.full((n_series_active, resolution[0][0], resolution[0][1], z_rg.size), np.nan)

mask_matrix_pad_k = np.array(geo_model.solutions.mask_matrix_pad)

sol = out.createGroup("solution")

sol.createDimension("n_active_series", n_series_active)
sol.createDimension("n_active_surfaces", n_surfaces_active)

lithvar = sol.createVariable("lith_block", "f8", ("nz", "ny", "nx"))
lithvar.setncatts(
    {
        "long_name": "ID values of the defined surfaces",
        "unit": "-",
        "var_desc": "values are float but the ID is the rounded integer value",
    }
)

sfm = sol.createVariable("scalar_field_matrix", "f8", ("n_active_series", "nz", "ny", "nx"))
sfm.setncatts(
    {
        "long_name": "Array with values of the scalar field",
        "unit": "-",
        "var_desc": "values of the scalar field at each location in the regular grid",
    }
)

bm = sol.createVariable("block_matrix", "f8", ("n_active_series", "nz", "ny", "nx"))
bm.setncatts(
    {
        "long_name": "Array holding interpolated ID values",
        "unit": "-",
        "var_desc": "array with all interpolated values for all series at each location in the regular grid",
    }
)

mm = sol.createVariable("mask_matrix", "b", ("n_active_series", "nz", "ny", "nx"))
mm.setncatts(
    {
        "long_name": "Boolean array holding information for series combination",
        "unit": "-",
        "var_desc": "contains the logic to combine multiple series to obtain the final model at each location in the regular grid",
    }
)

mmp = sol.createVariable("mask_matrix_pad", "b", ("n_active_series", "nz", "ny", "nx"))
mmp.setncatts(
    {
        "long_name": "Boolean array holding information for series combination (?)",
        "unit": "-",
        "var_desc": "mask matrix padded 2 block in order to guarantee that the layers intersect each other after marching cubes",
    }
)

sfsp = sol.createVariable("scalar_field_at_surface_points", "f8", ("n_active_series", "n_active_surfaces"))
sfsp.setncatts(
    {
        "long_name": "value of the scalar field at each interface",
        "unit": "-",
        "var_desc": "axis 0 is each series and axis 1 is each surface ordered by their id",
    }
)

# The ordering is assumed to be the same in all matrix related arrays.
for idx, (x, y, z) in enumerate(points_rg):
    is_x = x == x_rg
    is_y = y == y_rg
    is_z = z == z_rg

    lith_block_k[is_x, is_y, is_z] = lith_block_points[idx]

    for i in range(n_surfaces_active):
        scalar_field_matrix_k[i, is_x, is_y, is_z] = scalar_matrix_points[i, idx]
        block_matrix_k[i, is_x, is_y, is_z] = block_matrix_points[i, 0, idx]
        mask_matrix_k[i, is_x, is_y, is_z] = mask_matrix_points[i, idx]

lith_block_k = np.swapaxes(lith_block_k, 0, 2)
lithvar[:, :, :] = lith_block_k

scalar_field_matrix_k = np.swapaxes(scalar_field_matrix_k, 1, 3)
sfm[::, :, :, :] = scalar_field_matrix_k

block_matrix_k = np.swapaxes(block_matrix_k, 1, 3)
bm[::, :, :, :] = block_matrix_k

mask_matrix_k = np.swapaxes(mask_matrix_k, 1, 3)
mm[::, :, :, :] = mask_matrix_k

mask_matrix_pad_k = np.swapaxes(mask_matrix_pad_k, 1, 3)
mmp[::, :, :, :] = mask_matrix_pad_k

# Scalar field value of interfaces
scalar_field_surfpoints = geo_model.solutions.scalar_field_at_surface_points
sfsp[:] = scalar_field_surfpoints

vertices = [np.empty((0, 3), dtype=float)] * len(geo_model.solutions.vertices)
edges = [np.empty((0, 3), dtype=int)] * len(geo_model.solutions.edges)

path_surf = os.path.join(output_dir, "triangulated_surfaces")
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

print(f"Done model id, {output_dir}")

out.close()
