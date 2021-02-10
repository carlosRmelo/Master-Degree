#!/usr/bin/env python
# coding: utf-8

# In[2]:




import autolens as al
import autolens.plot as aplt


# In[3]:


grid = al.Grid.uniform(shape_2d=(80, 80), pixel_scales=0.1, sub_size=1)
psf = al.Kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)


# In[4]:


NFW_profile = al.mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1, scale_radius=1.5)
lens_galaxy = al.Galaxy(
    redshift=0.035,
    mass=NFW_profile,
)
NFW_profile.einstein_radius_in_units()


# In[5]:


source_galaxy = al.Galaxy(
    redshift=2.1,
    light=al.lp.EllipticalSersic(
        centre=(0.2, -0.1),
        elliptical_comps=(0.3, 0.111111),
        intensity=3,
        effective_radius=0.5,
        sersic_index=1,
    ),
)

print(source_galaxy)


# In[33]:


tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])


# In[36]:


mask = al.Mask.circular(
    shape_2d=grid.shape_2d, pixel_scales=grid.pixel_scales, sub_size=1,radius=3 ,centre=(0, 0)
)


# In[35]:


aplt.Galaxy.image(galaxy=source_galaxy, grid=grid)


# In[64]:


aplt.Tracer.image(tracer=tracer, grid=grid)


# In[65]:


normal_image = tracer.image_from_grid(grid=grid)
padded_image = tracer.padded_image_from_grid_and_psf_shape(
    grid=grid, psf_shape_2d=psf.shape_2d
)
print(normal_image.shape)
print(padded_image.shape)


# In[66]:


simulator = al.SimulatorImaging(
    exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
    psf=psf,
    background_sky_map=al.Array.full(fill_value=0.1, shape_2d=grid.shape_2d),
    add_noise=True,
)

imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)


# In[67]:


aplt.Imaging.image(imaging=imaging)


# In[68]:


from pyprojroot import here

workspace_path = str(here())
dataset_path = f"{workspace_path}/howtolens/MGE/Data"
print("Dataset Path: ", dataset_path)


# In[69]:


imaging.output_to_fits(
    image_path=f"{dataset_path}/ImageNFW.fits",
    noise_map_path=f"{dataset_path}/NoiseNFW.fits",
    psf_path=f"{dataset_path}/PSFNFW.fits",
    overwrite=True,
)


# In[ ]:




