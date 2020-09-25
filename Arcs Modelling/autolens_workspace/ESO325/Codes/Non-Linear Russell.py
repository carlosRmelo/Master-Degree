#!/usr/bin/env python
# coding: utf-8

# In[1]:


from autoconf import conf
import autofit as af  # <- This library is used for non-linear fitting.
import autolens as al
import autolens.plot as aplt


from time import perf_counter as clock

from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u
import numpy as np


# In[2]:


from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

conf.instance = conf.Config(
    config_path=f"{workspace_path}/howtolens/config",
    output_path=f"{workspace_path}/ESO325/Output/Voronoi",
)


# In[4]:


#Reading data

dataset_type = "ESO325"
dataset_name = "Data"
dataset_path = f"{workspace_path}/{dataset_type}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/arcs_resized.fits",
    noise_map_path=f"{dataset_path}/noise_map_resized.fits",
    psf_path=f"{dataset_path}/psf.fits",
    pixel_scales=0.04,
)


# In[5]:


#Criando uma máscara e plotando tudo junto

mask = al.Mask.circular_annular(
    shape_2d=imaging.shape_2d, 
    pixel_scales=imaging.pixel_scales, 
    sub_size=1,
    inner_radius=2.2,
    outer_radius=3.8, 
    centre=(0.2, -0.2)

)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

aplt.Imaging.subplot_imaging(
    imaging=imaging, mask=mask, include=aplt.Include(border=True),
)


# In[6]:


settings = al.PhaseSettingsImaging(grid_class=al.Grid, sub_size=1)


# In[7]:


elliptical_comps = al.convert.elliptical_comps_from(axis_ratio=1-0.28, phi=158)

print(elliptical_comps)


# In[44]:


#Defindo os parâmetros da lente
lens_galaxy=al.GalaxyModel(
    redshift=0.035, 
    mass=al.mp.EllipticalIsothermal
)


#Componentes do perfil de matéria
lens_galaxy.mass.einstein_radius = 2.85
lens_galaxy.mass.elliptical_comps.elliptical_comps_0 = elliptical_comps[0]
lens_galaxy.mass.elliptical_comps.elliptical_comps_1 = elliptical_comps[1]

print(lens_galaxy.info)


# In[61]:


phase = al.PhaseImaging(
        phase_name="Russel_sie",
        galaxies=dict(
            lens=lens_galaxy,
            source=al.GalaxyModel(redshift=2.1, 
                                  pixelization=al.pix.VoronoiMagnification(shape=(40, 40)), 
                                  regularization=al.reg.Constant(coefficient=8.0),
                                 )
                        ),
        settings=settings,
        search = af.Emcee(nwalkers=50, nsteps=500),
    )


# In[62]:


result =  phase.run(dataset=imaging, mask=mask)


# In[ ]:




