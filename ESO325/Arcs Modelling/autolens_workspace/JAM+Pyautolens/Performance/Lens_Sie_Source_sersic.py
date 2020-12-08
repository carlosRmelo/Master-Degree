#!/usr/bin/env python
# coding: utf-8

# In[62]:


import autolens as al
import autolens.plot as aplt
import time


# In[63]:


imaging = al.Imaging.from_fits(
        image_path=f"image.fits",
        noise_map_path=f"noise.fits",
        psf_path=f"psf.fits",
        pixel_scales=0.04,
    )

mask = al.Mask.circular(centre=(0.0, 0.0),
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=4, radius=3.5
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)


# In[64]:


#aplt.Imaging.subplot_imaging(
#    imaging=imaging, mask=mask, include=aplt.Include(border=True)
#)


# In[65]:


sis_mass_profile = al.mp.EllipticalNFW()
lens_galaxy = al.Galaxy(
    redshift=0.035,
    mass=sis_mass_profile,
)


# In[66]:


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




