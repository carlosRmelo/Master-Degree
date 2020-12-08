#!/usr/bin/env python
# coding: utf-8

# In[1]:


import autolens as al
import autolens.plot as aplt
import time
import numpy as np


# In[2]:


Total_Mass, Total_sigma_RAD, Total_q_proj = np.loadtxt("Input.txt", unpack=True)


# In[3]:


z_lens = 0.035
z_source = 2.1

inc = 90.  


# In[5]:


imaging = al.Imaging.from_fits(
        image_path="arcs_resized.fits",
        noise_map_path="noise_map_resized.fits",
        psf_path="psf.fits",
        pixel_scales=0.04,
    )

mask_custom = al.Mask.from_fits(
    file_path="mask gui.fits", hdu=0, pixel_scales=imaging.pixel_scales
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask_custom)


# In[7]:


mass_profile = al.mp.MGE(centre=(0.0, 0.0),method='sciquad')                         
mass_profile.MGE_comps(M=Total_Mass, sigma=Total_sigma_RAD,
                       q=Total_q_proj, z_l=z_lens, z_s=z_source)        

mass_profile.MGE_Grid_parameters(masked_imaging.grid)              
                                                                         

lens_galaxy = al.Galaxy(                                            
        redshift=0.035,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=(0,0)),
    )


# In[8]:


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




