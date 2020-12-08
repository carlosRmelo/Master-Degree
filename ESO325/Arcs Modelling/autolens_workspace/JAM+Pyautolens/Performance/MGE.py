#!/usr/bin/env python
# coding: utf-8

# In[15]:


import autolens as al
import autolens.plot as aplt
import time
import numpy as np


# In[52]:


imaging = al.Imaging.from_fits(
        image_path=f"arcs_resized.fits",
        noise_map_path=f"noise_map_resized.fits",
        psf_path=f"psf_resize.fits",
        pixel_scales=0.04,
    )

mask = al.Mask.circular_annular(centre=(0.0, 0.0),inner_radius=2, outer_radius=3.5,
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1,
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)
masked_imaging.grid.shape_2d


# In[53]:


#aplt.Imaging.subplot_imaging(
#    imaging=imaging, mask=mask, include=aplt.Include(border=True)
#)


# In[54]:


mass = np.ones(35)*10
sigma = np.ones(35)*2
q = np.ones(35)*0.8


# In[55]:


mass_profile = al.mp.MGE(centre=(0.0, 0.0))                        
mass_profile.MGE_comps(M=mass, sigma=sigma,
                       q=q, z_l=0.035, z_s=2.1)       

mass_profile.MGE_Grid_parameters(masked_imaging.grid)              
                                                                        


# In[56]:



lens_galaxy = al.Galaxy(
    redshift=0.035,
    mass=mass_profile,
)


# In[57]:


start = time.time()

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=2.1)])
source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[1]

rectangular = al.pix.Rectangular(shape=(80, 80))
mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)

inversion = al.Inversion(
        masked_dataset=masked_imaging,
        mapper=mapper,
        regularization=al.reg.Constant(coefficient=1.0),
    )

print("Final", (time.time() - start))


# In[ ]:




