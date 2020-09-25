#!/usr/bin/env python
# coding: utf-8

# Author: Carlos Roberto de Melo
#     
# Date: 08/10/2020
#     
# Obj: Modelar ESO325 com um perfil de An _EllipticalSersic LightProfile for the lens galaxy’s light e EllipticalIsothermal (SIE) MassProfile for the lens galaxy’s mass. A fonte será obtida a partir da inversão em uma grid regular. 

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
    output_path=f"{workspace_path}/ESO325/Output/Rectangular",
)


# In[3]:


#Reading data

dataset_type = "ESO325"
dataset_name = "Data"
dataset_path = f"{workspace_path}/{dataset_type}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits", image_hdu=1,
    noise_map_path=f"{dataset_path}/noise_map.fits",
    psf_path=f"{dataset_path}/psf.fits",
    pixel_scales=0.04,
)


# In[4]:


plotter = aplt.SubPlotter(cmap=aplt.ColorMap.sub(norm_min=0, norm_max=10))


# In[5]:


mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1,radius=8 ,centre=(0, 0)
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)
'''
aplt.Imaging.subplot_imaging(sub_plotter=plotter,
    imaging=imaging, mask=mask, include=aplt.Include(border=True),
)
'''


# In[10]:


settings = al.PhaseSettingsImaging(grid_class=al.Grid, sub_size=2)


# In[11]:


elliptical_comps = al.convert.elliptical_comps_from(axis_ratio=1-0.25, phi=45.0+90)

print(elliptical_comps)


# In[20]:


#Defindo os parâmetros da lente

lens_galaxy=al.GalaxyModel(redshift=0.035, mass=al.mp.EllipticalIsothermal, light=al.lp.EllipticalSersic)

#Componentes do perfil de luz
lens_galaxy.light.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
lens_galaxy.light.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
lens_galaxy.light.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=5.0)
lens_galaxy.light.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(mean=-0.14285714285714285, sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
lens_galaxy.light.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(mean=0, sigma=0.1, lower_limit=-1.0, upper_limit=1.0)

#Componentes do perfil de matéria
lens_galaxy.mass.einstein_radius = af.GaussianPrior(mean=2.95, sigma=0.5, lower_limit=0.0, upper_limit=np.inf)

print(lens_galaxy.light.info)
print(lens_galaxy.mass.info)


# In[21]:


phase = al.PhaseImaging(
        phase_name="ESO325_sie",
        galaxies=dict(
            lens=lens_galaxy,
            source=al.GalaxyModel(redshift=2.1, pixelization=al.pix.Rectangular(shape=(25,25)), regularization=al.reg.Constant),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )


# In[22]:


result =  phase.run(dataset=imaging, mask=mask)


# In[ ]:




