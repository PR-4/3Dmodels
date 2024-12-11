import os
import numpy as np
import pandas as pd
import xarray as xr
import gempy as gp
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pyvista as pv
import pyvistaqt as pvqt

class GempyModelProcessor:
    def __init__(self, model_name, base_path):
        """
        Initialize the GemPy model processor.
        
        Args:
            model_name (str): Name of the model without extension
            base_path (str): Base path to the model directory
        """
        self.model_name = model_name
        self.base_path = base_path
        self.path_model = os.path.join(base_path)
        self.path_output = os.path.join(self.path_model, f"{model_name}_results")
        self.path_csv = os.path.join(self.path_output, "csv_results")
        self.path_figs = os.path.join(self.path_output, "figs")
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.path_output, self.path_csv, self.path_figs]:
            if not os.path.exists(path):
                os.makedirs(path)
                
    def load_model(self):
        """Load the GemPy model from pickle file."""
        try:
            self.geo_model = gp.load_model_pickle(os.path.join(self.path_model, self.model_name))
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
    def export_dataframes(self):
        """Export model dataframes to CSV files."""
        try:
            # Copy dataframes
            surfpoints = copy.copy(self.geo_model.surface_points.df)
            orientations = copy.copy(self.geo_model.orientations.df)
            surfaces = copy.copy(self.geo_model.surfaces.df)
            surfaces = surfaces.drop(columns=["vertices", "edges"])
            series = copy.copy(self.geo_model.series.df)
            series.insert(1, "series", series.index)
            
            # Export to CSV
            dataframes = {
                "surface_points.csv": surfpoints,
                "orientations.csv": orientations,
                "series.csv": series,
                "surfaces.csv": surfaces
            }
            
            for filename, df in dataframes.items():
                df.to_csv(os.path.join(self.path_csv, filename), index=False)
        except Exception as e:
            print(f"Error exporting dataframes: {e}")
            raise
            
    def export_additional_data(self):
        """Export additional model data to CSV files."""
        try:
            ad = self.geo_model.additional_data
            
            # Kriging data
            kriging_data = copy.copy(ad.kriging_data.df)
            kriging_data.insert(0, "Model_ID", self.model_name)
            kriging_path = os.path.join(self.path_csv, "kriging_parameters.csv")
            kriging_data.to_csv(kriging_path, index=False, mode="a", header=not os.path.exists(kriging_path))
            
            # Rescaling data
            rescaling_data = copy.copy(ad.rescaling_data.df)
            rescaling_data.insert(0, "Model_ID", self.model_name)
            rescaling_path = os.path.join(self.path_csv, "rescaling_parameters.csv")
            rescaling_data.to_csv(rescaling_path, index=False, mode="a", header=not os.path.exists(rescaling_path))
        except Exception as e:
            print(f"Error exporting additional data: {e}")
            raise
            
    def export_grid_data(self):
        """Export regular grid data to CSV and prepare coordinates for NetCDF."""
        try:
            colnames = ["x", "y", "z"]
            
            # Resolution
            resolution = self.geo_model.grid.regular_grid.resolution.reshape(1, -1)
            pd.DataFrame(data=resolution, columns=colnames).to_csv(
                os.path.join(self.path_csv, "resolution.csv"), index=False)
            
            # Spacing
            spacing = np.array(self.geo_model.grid.regular_grid.get_dx_dy_dz()).reshape(1, -1)
            pd.DataFrame(data=spacing, columns=colnames).to_csv(
                os.path.join(self.path_csv, "spacing.csv"), index=False)
            
            # Extent
            extent = self.geo_model.grid.regular_grid.extent.reshape(1, -1)
            pd.DataFrame(data=extent, columns=["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]).to_csv(
                os.path.join(self.path_csv, "extent.csv"), index=False)
                
            return self.geo_model.grid.regular_grid
        except Exception as e:
            print(f"Error exporting grid data: {e}")
            raise
            
    def create_netcdf_dataset(self, regular_grid):
        """Create and populate NetCDF dataset."""
        try:
            # Extract and prepare coordinates
            z_rg = regular_grid.z[::-1]
            x_rg = regular_grid.x
            y_rg = regular_grid.y
            
            # Create dataset with coordinates
            ds = xr.Dataset(coords={
                "Model_ID": self.model_name,
                "nx": x_rg,
                "ny": y_rg,
                "nz": z_rg
            })
            
            # Add variables
            ds["lon"] = ("nx", x_rg)
            ds["lat"] = ("ny", y_rg)
            ds["depth"] = ("nz", z_rg)
            
            # Add variable attributes
            self._add_variable_attributes(ds)
            
            return ds, x_rg, y_rg, z_rg
        except Exception as e:
            print(f"Error creating NetCDF dataset: {e}")
            raise
        
    def _add_variable_attributes(self, ds):
        """Add attributes to NetCDF variables."""
        attrs = {
            "lon": {
                "long_name": "posição espacial longitudinal dos voxels",
                "unit": "metro",
                "var_desc": "CRS is EPSG:"
            },
            "lat": {
                "long_name": "posição espacial latitudinal dos voxels",
                "unit": "metro",
                "var_desc": "CRS is EPSG:"
            },
            "depth": {
                "long_name": "depth dos voxels",
                "unit": "metro",
                "var_desc": "depth é dada à unidade abaixo do nível do mar"
            }
        }
        
        for var, attr in attrs.items():
            ds[var].attrs = attr
            
    def _process_solution_arrays(self, points_rg, x_rg, y_rg, z_rg, lith_block_k, 
                               scalar_field_matrix_k, block_matrix_k, mask_matrix_k,
                               n_surfaces_active, n_series_active):
        """Process the solution arrays and fill them with data."""
        try:
            for idx, (x, y, z) in enumerate(points_rg):
                is_x = x == x_rg
                is_y = y == y_rg
                is_z = z == z_rg

                lith_block_k[is_x, is_y, is_z] = self.geo_model.solutions.lith_block[idx]

                for i in range(min(n_surfaces_active, n_series_active)):
                    scalar_field_matrix_k[i, is_x, is_y, is_z] = self.geo_model.solutions.scalar_field_matrix[i, idx]
                    block_matrix_k[i, is_x, is_y, is_z] = self.geo_model.solutions.block_matrix[i, 0, idx]
                    mask_matrix_k[i, is_x, is_y, is_z] = self.geo_model.solutions.mask_matrix[i, idx]

            # Swap axes for correct orientation
            lith_block_k = np.swapaxes(lith_block_k, 0, 2)
            scalar_field_matrix_k = np.swapaxes(scalar_field_matrix_k, 1, 3)
            block_matrix_k = np.swapaxes(block_matrix_k, 1, 3)
            mask_matrix_k = np.swapaxes(mask_matrix_k, 1, 3)

            return lith_block_k, scalar_field_matrix_k, block_matrix_k, mask_matrix_k
        except Exception as e:
            print(f"Error processing solution arrays: {e}")
            raise
            
    def _add_solution_to_dataset(self, ds, lith_block_k, scalar_field_matrix_k,
                              block_matrix_k, mask_matrix_k):
        """Add processed solution data to the dataset."""
        try:
            # Add the variables to the dataset
            ds["lith_block"] = (("nz", "ny", "nx"), lith_block_k)
            ds["scalar_field_matrix"] = (("n_active_series", "nz", "ny", "nx"), scalar_field_matrix_k)
            ds["block_matrix"] = (("n_active_series", "nz", "ny", "nx"), block_matrix_k)
            ds["mask_matrix"] = (("n_active_series", "nz", "ny", "nx"), mask_matrix_k)
            
            # Add solution field at surface points
            scalar_field_surfpoints = self.geo_model.solutions.scalar_field_at_surface_points
            ds["scalar_field_at_surface_points"] = (
                ("n_active_series", "n_active_surfaces"), 
                scalar_field_surfpoints
            )
            
            # Add variable attributes
            solution_attrs = {
                "lith_block": {
                    "long_name": "ID values of the defined surfaces",
                    "unit": "-",
                    "var_desc": "values are float but the ID is the rounded integer value"
                },
                "scalar_field_matrix": {
                    "long_name": "Array with values of the scalar field",
                    "unit": "-",
                    "var_desc": "values of the scalar field at each location in the regular grid"
                },
                "block_matrix": {
                    "long_name": "Array holding interpolated ID values",
                    "unit": "-",
                    "var_desc": "array with all interpolated values for all series"
                }
            }
            
            for var, attr in solution_attrs.items():
                ds[var].attrs = attr
                
        except Exception as e:
            print(f"Error adding solution to dataset: {e}")
            raise
            
    def process_solution_data(self, ds, x_rg, y_rg, z_rg):
        """Process solution data and add to NetCDF dataset."""
        try:
            points_rg = self.geo_model.solutions.grid.get_grid("regular")
            resolution = self.geo_model.grid.regular_grid.resolution
            n_series_active = len(self.geo_model.series.df.index) - 1
            n_surfaces_active = len(self.geo_model.surfaces.df.index) - 1
            
            # Initialize arrays
            lith_block_k = np.full((resolution[0], resolution[1], resolution[2]), np.nan)
            scalar_field_matrix_k = np.full((n_series_active, resolution[0], resolution[1], resolution[2]), np.nan)
            block_matrix_k = np.full((n_series_active, resolution[0], resolution[1], resolution[2]), np.nan)
            mask_matrix_k = np.full((n_series_active, resolution[0], resolution[1], resolution[2]), np.nan)
            
            # Process solutions
            lith_block_k, scalar_field_matrix_k, block_matrix_k, mask_matrix_k = \
                self._process_solution_arrays(points_rg, x_rg, y_rg, z_rg, 
                                         lith_block_k, scalar_field_matrix_k,
                                         block_matrix_k, mask_matrix_k,
                                         n_surfaces_active, n_series_active)
            
            # Add to dataset
            self._add_solution_to_dataset(ds, lith_block_k, scalar_field_matrix_k,
                                      block_matrix_k, mask_matrix_k)
        except Exception as e:
            print(f"Error processing solution data: {e}")
            raise

    def save_cross_sections(self):
        """Save all cross sections to files without displaying them."""
        try:
            # Load necessary data
            spatial_data = xr.open_dataset(os.path.join(self.path_output, f"{self.model_name}.nc"))
            surfpoints = pd.read_csv(os.path.join(self.path_csv, "surfaces.csv"))
            
            # Create directories for cross sections
            path_figs_y = os.path.join(self.path_figs, "cross_section_y")
            path_figs_x = os.path.join(self.path_figs, "cross_section_x")
            path_figs_z = os.path.join(self.path_figs, "cross_section_z")
            
            for path in [path_figs_y, path_figs_x, path_figs_z]:
                if not os.path.exists(path):
                    os.makedirs(path)

            # Get data
            x = spatial_data["lon"][:].data
            y = spatial_data["lat"][:].data
            z = spatial_data["depth"][:].data
            surface = np.round(spatial_data["lith_block"][:].data)
            nsurf = surfpoints.id.size
            cmap = colors.ListedColormap(surfpoints.color.values)
            
            # Save Y cross-sections
            for idx, ypos in enumerate(y):
                X, Z = np.meshgrid(x, z)
                S = surface[:, idx, :]
                
                fig, ax = plt.subplots(1, 1, figsize=(15, 8))
                ax.pcolormesh(X, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)
                ax.set_xlabel("X [m]", fontsize=15)
                ax.set_ylabel("Depth [m]", fontsize=15)
                ax.set_title(f"Cross-section, south-to-north, row no.: {idx} - northing, inline, y: {ypos}",
                            pad=10, fontsize=20)
                
                figname = os.path.join(path_figs_y, f"cs_y_row-{idx}_y-{ypos}.png")
                fig.savefig(figname, bbox_inches="tight", dpi=300)
                plt.close(fig)
            print("Saved all Y cross-section")
                
            # Save X cross-sections
            for idx, xpos in enumerate(x):
                Y, Z = np.meshgrid(y, z)
                S = surface[:, :, idx]
                
                fig, ax = plt.subplots(1, 1, figsize=(15, 8))
                ax.pcolormesh(Y, Z, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)
                ax.set_xlabel("Y [m]", fontsize=15)
                ax.set_ylabel("Depth [m]", fontsize=15)
                ax.set_title(f"Cross-section, west-to-east row no.: {idx} - easting, xline, x: {xpos}",
                            pad=10, fontsize=20)
                
                figname = os.path.join(path_figs_x, f"cs_x_row-{idx}_x-{xpos}.png")
                fig.savefig(figname, bbox_inches="tight", dpi=300)
                plt.close(fig)
            print("Saved all X cross-section")
                
            # Save Z cross-sections
            for idx, zpos in enumerate(z):
                X, Y = np.meshgrid(x, y)
                S = surface[idx, :, :]
                
                fig, ax = plt.subplots(1, 1, figsize=(15, 8))
                ax.pcolormesh(X, Y, S, shading="nearest", cmap=cmap, zorder=0, vmin=1, vmax=nsurf)
                ax.set_xlabel("X [m]", fontsize=15)
                ax.set_ylabel("Y [m]", fontsize=15)
                ax.set_title(f"Cross-section, top-to-bottom, row no.: {idx} - depth, z: {zpos}",
                            pad=10, fontsize=20)
                
                figname = os.path.join(path_figs_z, f"cs_z_depth-{idx}.png")
                fig.savefig(figname, bbox_inches="tight", dpi=300)
                plt.close(fig)
            print("Saved all Z cross-section")
                
        except Exception as e:
            print(f"Error saving cross sections: {e}")
            raise

    def visualize_3d_pyvista(self):
        """Create a 3D visualization using PyVista."""
        try:
            # Load data
            spatial_data = xr.open_dataset(os.path.join(self.path_output, f"{self.model_name}.nc"))
            surfpoints = pd.read_csv(os.path.join(self.path_csv, "surfaces.csv"))
            extent = pd.read_csv(os.path.join(self.path_csv, "extent.csv"))
            spacing = pd.read_csv(os.path.join(self.path_csv, "spacing.csv"))
            
            # Create grid
            surface = np.round(spatial_data["lith_block"][:].data)
            nrow, ncol, nlay = surface.shape
            
            grid = pv.UniformGrid((nrow, ncol, nlay))
            grid.origin = (extent.xmin[0], extent.ymin[0], extent.zmin[0])
            grid.spacing = (spacing.x[0], spacing.y[0], spacing.z[0])
            grid["lith_block"] = surface.ravel(order="C")
            
            # Create colormap
            colorsz = [colors.to_rgba(color, 1) for color in surfpoints.color.values]
            cmapz = colors.ListedColormap(colorsz)
            
            # Create plotter
            p = pvqt.BackgroundPlotter()
            p.add_mesh(grid, scalars="lith_block", cmap=cmapz, show_edges=False, lighting=True)
            p.set_scale(zscale=5)
            p.show_bounds(font_size=10, location="furthest", color="black",
                        xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
            return p
            
        except Exception as e:
            print(f"Error creating PyVista visualization: {e}")
            raise

    def visualize_3d_gempy(self):
        """Create a 3D visualization using GemPy."""
        try:
            # Load model
            geo_model = gp.load_model_pickle(os.path.join(self.path_model, self.model_name))
            
            # Create plot
            p = gp.plot_3d(geo_model, plotter_type="background", show_data=False,
                        show_lith=False, ve=5)
            return p
            
        except Exception as e:
            print(f"Error creating GemPy visualization: {e}")
            raise

    def export_triangulated_surfaces(self):
        """Export triangulated surfaces data."""
        try:
            path_surf = os.path.join(self.path_output, "triangulated_surfaces")
            if not os.path.isdir(path_surf):
                os.mkdir(path_surf)

            vertices = [np.empty((0, 3), dtype=float)] * len(self.geo_model.solutions.vertices)
            edges = [np.empty((0, 3), dtype=int)] * len(self.geo_model.solutions.edges)

            vertices_k = self.geo_model.solutions.vertices
            edges_k = self.geo_model.solutions.edges

            for idx in range(len(vertices)):
                try:
                    vertices[idx] = np.append(vertices[idx], vertices_k[idx], axis=0)
                except ValueError:
                    pass

                try:
                    max_edge = np.max(edges[idx]) if not edges[idx].size == 0 else 0
                    edges_k[idx] += int(max_edge)
                    edges[idx] = np.append(edges[idx], edges_k[idx], axis=0)
                except ValueError:
                    pass

            for idx, (vv, ee) in enumerate(zip(vertices, edges)):
                vert = pd.DataFrame(data=vv, columns=["x", "y", "z"])
                path_vert = os.path.join(path_surf, f"vertices_id-{idx}.csv")
                vert.to_csv(path_vert, index=False, mode="a", header=not os.path.exists(path_vert))

                edge = pd.DataFrame(data=ee, columns=["idx1", "idx2", "idx3"])
                path_edge = os.path.join(path_surf, f"edges_id-{idx}.csv")
                edge.to_csv(path_edge, index=False, mode="a", header=not os.path.exists(path_edge))

        except Exception as e:
            print(f"Error exporting triangulated surfaces: {e}")
            raise

    def visualize_3d(self, ve=5):
        """
        Create a 3D visualization of the model using PyVista.
        
        Args:
            ve (float): Vertical exaggeration factor
        """
        try:
            # Load required data
            spatial_data = xr.open_dataset(os.path.join(self.path_output, f"{self.model_name}.nc"))
            surfpoints = pd.read_csv(os.path.join(self.path_csv, "surfaces.csv"))
            extent = pd.read_csv(os.path.join(self.path_csv, "extent.csv"))
            spacing = pd.read_csv(os.path.join(self.path_csv, "spacing.csv"))

            # Create grid
            surface = np.round(spatial_data["lith_block"][:].data)
            nrow, ncol, nlay = surface.shape

            grid = pv.UniformGrid((nrow, ncol, nlay))
            grid.origin = (extent.xmin[0], extent.ymin[0], extent.zmin[0])
            grid.spacing = (spacing.x[0], spacing.y[0], spacing.z[0])
            grid["lith_block"] = surface.ravel(order="C")

            # Create colormap
            alpha = 1
            colorsz = [colors.to_rgba(color, alpha) for color in surfpoints.color.values]
            cmapz = colors.ListedColormap(colorsz)

            # Create plotter
            p = pvqt.BackgroundPlotter()
            p.add_mesh(grid, scalars="lith_block", cmap=cmapz, show_edges=False, lighting=True)
            p.set_scale(zscale=ve)
            p.show_bounds(
                font_size=10,
                location="furthest",
                color="black",
                xlabel="X [m]",
                ylabel="Y [m]",
                zlabel="Z [m]"
            )
            
            return p

        except Exception as e:
            print(f"Error creating 3D visualization: {e}")
            raise
            
    def run_processing(self):
        """Run the complete model processing pipeline."""
        try:
            print(f"Processing model: {self.model_name}")
            
            # Load and process model
            self.load_model()
            self.export_dataframes()
            self.export_additional_data()
            
            # Process grid and create NetCDF
            regular_grid = self.export_grid_data()
            ds, x_rg, y_rg, z_rg = self.create_netcdf_dataset(regular_grid)
            self.process_solution_data(ds, x_rg, y_rg, z_rg)
            
            # Export triangulated surfaces
            self.export_triangulated_surfaces()
            
            # Save NetCDF file
            ds.to_netcdf(os.path.join(self.path_output, f"{self.model_name}.nc"))
            print(f"Processing complete. Output saved to {self.path_output}")
            
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            raise

def main():
    """Main function to demonstrate usage."""
    try:
        model_name = "Santos-Inter-Sismicas-Itapema"
        base_path = "../../../output/Santos_Basin/sismicas_interpretadas/itapema/"
        
        # Initialize and run processor
        processor = GempyModelProcessor(model_name, base_path)
        
        # Run processing pipeline
        processor.run_processing()
        
        # Save cross sections
        processor.save_cross_sections()
        
        # Optional: Create 3D visualization with either PyVista or GemPy
        visualization_type = "gempy"  # or "gempy"
        if visualization_type == "pyvista":
            viewer = processor.visualize_3d_pyvista()
        else:
            viewer = processor.visualize_3d_gempy()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()