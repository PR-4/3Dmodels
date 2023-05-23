#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing dependency
import gempy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#internal modules
import sys
sys.path.insert(0,'../../../modules')
import dio as dio
from dio import debug
import modeling as md



seismic = '../../../input/Santos_Basin/Santos_Basin_114ma_seismic/modelo_completo/114ma.png'

ma = md.transform(seismic)


x,y = md.horizon(ma)





pd.set_option('precision', 2)

# Data path, creating the model object and the name
data_path = '../../../input/Santos_Basin/Santos_Basin_114ma_seismic/modelo_ajustado/'
geo_model = gp.create_model('LakePreSal_CFF')

dio.stop()



# Extension of the model, resolution of the model and the paths of surface and orientation
gp.init_data(geo_model,
             extent=[0, 4000, 13750, 40000, -10000, 0],
             resolution=[250, 250, 250],
             path_i=data_path + "surfaces_points2.csv",
             path_o=data_path + "orientations_points.csv")


# In[3]:


# The surfaces of the model
gp.get_data(geo_model, 'surfaces')


# In[4]:


# Sorting the surfaces and defining a serie for them (Stratigraphy - Top to Bottom)
gp.map_stack_to_surfaces(geo_model,                         
                         {"Fault_series": ('FALHA_1'),
                          "Strat_Series": ('UBV', 'LBV', 'ITAPEMA', 'CAMBORIU', 'basement')})

# Ver o grid do modelo
geo_model.grid


# In[5]:


# Declaring the fault serie
geo_model.set_is_fault(['Fault_series'])


# In[6]:


geo_model.surfaces


# In[7]:


# Plot 2D of the data in X, Y and Z direction
gp.plot_2d(geo_model, direction=['x'], show_data=True, show_boundaries=True, legend=False, show=True)

#gp.plot_2d(geo_model, direction=['y'], show_data=True, show_boundaries=True, legend=False, show=True)

#gp.plot_2d(geo_model, direction=['z'], show_data=True)


# In[8]:


# Plot in 3D
gpv = gp.plot_3d(geo_model, plotter_type='basic', image=False, show_data=True, show_surfaces=True, show_scalar=True, show_boundaries=True)


# In[9]:


get_ipython().run_cell_magic('time', '', "# Interpolating\ngp.set_interpolator(geo_model,\n                    theano_optimizer='fast_run',\n                    compile_theano=True\n                    )")


# In[10]:


get_ipython().run_cell_magic('time', '', '# Computing a solution for the model\nsol = gp.compute_model(geo_model, compute_mesh=True)')


# In[17]:


# Plot 2D of Y and X
#gp.plot_2d(geo_model, direction="y", show_data=False, show_lith=True)
gp.plot_2d(geo_model, direction="x", show_data=True, show_lith=True)


# In[19]:


# Plot 3D
ver, sim = gp.get_surfaces(geo_model)
gpv = gp.plot_3d(geo_model, image=False, plotter_type='basic', show_data=False, show_results=True, show_lith=True, show_boundaries=True)


# In[13]:


# Saving the model in a .zip file
'''
This code saves the model in a .zip file in 
the same folder that the .ipynb file is located
'''

#gp.save_model(geo_model, compress=True)

