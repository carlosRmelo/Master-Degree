#!/usr/bin/env python
# coding: utf-8

# Title: Source Reconstruct of Spherical Power-Law Lens
# 
# Date: 08/03/2020
# 
# Obj: Using the MGE formalism to reconstruct que the original source galaxy deflected by a Spherical Power-Law (SPL) mass model

# In[1]:


import autolens as al
import autolens.plot as aplt
from pyprojroot import here
import numpy as np

from time import perf_counter as clock

from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u

workspace_path = str(here())
print("Workspace Path: ", workspace_path)


# In[2]:


#Lens parameters

z_lens = 0.035
z_source = 2.1

D_l = cosmo.angular_diameter_distance(z_lens)
D_s = cosmo.angular_diameter_distance(z_source)
D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s

#MGE parameters
#Reading data
Surf_mge, sigma, qObs = np.loadtxt('SPL MGE.txt', unpack=True)

#Converting and computing some quantities
Surf_mge = Surf_mge*(u.solMass/u.pc**2)                        #Surface Density in M_sun/pc²
sigma = sigma*u.arcsec                                         #Sigma in arcsec
sigma_pc = (sigma*D_l).to(u.pc, u.dimensionless_angles())      #Convert sigma in arcsec to sigma in pc
Mass_mge = 2*np.pi*Surf_mge*(sigma_pc**2)*qObs                 #Total mass per gaussian component in M_sun




#Defining inputs for the integral below
i = np.deg2rad(90)*u.rad                                       #Inclination angle in rad
M0 = Mass_mge                                                  #Mass per gaussian component in M_sun
q0 = np.sqrt(qObs**2 - np.cos(i)**2)/np.sin(i)                 #Deprojected axial ratio
sigma0 = (sigma).to(u.rad)                                     #Sigma per gaussian in rad

M0.sum()


# In[3]:


#Reading simulated data
dataset_type = "Testes with MGE/Spherical Power-Law"
dataset_name = "Data"
dataset_path = f"{workspace_path}/howtolens/{dataset_type}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/ImageSPL.fits",
    noise_map_path=f"{dataset_path}/NoiseSPL.fits",
    psf_path=f"{dataset_path}/PsfSPL.fits",
    pixel_scales=0.1,
)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1,radius=6.5 ,centre=(0., 0)
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

aplt.Imaging.subplot_imaging(
    imaging=imaging, mask=mask, include=aplt.Include(border=True)
)


# __Defining the MGE mass model for the lens galaxy and performing the ray tracing__

# In[4]:


time = clock()

mass_profile = al.mp.MGE(centre=(0.0, 0.0))
mass_profile.MGE_comps(M=M0.value, sigma=sigma0.value, q=q0.value, z_l=z_lens, z_s=z_source)
    
lens_galaxy = al.Galaxy(
        redshift=0.035,
        mass=mass_profile,
    )

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=2.1)])
source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[1]

print(lens_galaxy)

print(clock() - time)


# __Rectangular Inversion__

# In[5]:


time = clock()

rectangular = al.pix.Rectangular(shape=(80, 80))

mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)

aplt.Mapper.subplot_image_and_mapper(
    image=imaging.image,
    mapper=mapper,
    include=aplt.Include(mask=False, inversion_grid=True),
)

print(clock() - time)


# In[6]:


time = clock()

inversion = al.Inversion(
    masked_dataset=masked_imaging,
    mapper=mapper,
    regularization=al.reg.Constant(coefficient=8.0),
)

print(clock() - time)


# In[80]:


aplt.Inversion.reconstruction(inversion=inversion)


# In[7]:


#Defining some output configs

sub_plotter = aplt.SubPlotter(output=aplt.Output(path='/home/carlos/autolens_workspace/howtolens/Testes with MGE/Spherical Power-Law/Image Output/Rectangular Reconstruction/',
                                          filename='Rectangular Inversion Plots',
                                          format='png'),        
                              )

plotter = aplt.Plotter(output=aplt.Output(path='/home/carlos/autolens_workspace/howtolens/Testes with MGE/Spherical Power-Law/Image Output/Rectangular Reconstruction/',
                                          format='png'),         
                          )

include = aplt.Include(inversion_grid=False,
                       inversion_pixelization_grid=False,
                       inversion_border=True,
                       inversion_image_pixelization_grid=False,
                      )


# In[8]:


#Ploting Results
aplt.Inversion.subplot_inversion(inversion=inversion,
                                include=(
                                     aplt.Include(
                                               inversion_grid=False, 
                                               inversion_pixelization_grid=False, 
                                               inversion_image_pixelization_grid=False)
                                         ),
                                 sub_plotter=sub_plotter
                                )
 
aplt.Inversion.subplot_inversion(inversion=inversion,
                                include=(
                                     aplt.Include(
                                               inversion_grid=False, 
                                               inversion_pixelization_grid=False, 
                                               inversion_image_pixelization_grid=False)
                                         ),
                                )


# In[9]:


#Save image results
aplt.Inversion.individuals(inversion=inversion, 
                           plotter=plotter,
                           include=include,
                           plot_interpolated_reconstruction=True, 
                           plot_errors=True,
                           plot_reconstruction=True, 
                           plot_residual_map=True, 
                           plot_chi_squared_map=True,
                           plot_normalized_residual_map=True,
                               )


# ####################################################################################################################

# In[9]:


#Here we define an usulfull function
def fit_masked_imaging_with_source_galaxy(masked_imaging, source_galaxy):
    mass_profile = al.mp.MGE(centre=(0.0, 0.0))
    mass_profile.MGE_comps(M=M0.value, sigma=sigma0.value, q=q0.value, z_l=z_lens, z_s=z_source)
    
    lens_galaxy = al.Galaxy(
        redshift=0.035,
        mass=mass_profile,
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)


# __Inversion with Constante Voronoi__

# In[10]:


time = clock(
)
source_magnification = al.Galaxy(
    redshift=2.1,
    pixelization=al.pix.VoronoiMagnification(shape=(25, 25)),
    regularization=al.reg.Constant(coefficient=7),
)

fit = fit_masked_imaging_with_source_galaxy(
    masked_imaging=masked_imaging, source_galaxy=source_magnification
)

print(clock() - time)


# In[ ]:


sub_plotter = aplt.SubPlotter(output=aplt.Output(path='/home/carlos/autolens_workspace/howtolens/Testes with MGE/Spherical Power-Law/Image Output/ConstVoronoi/',
                                          filename='Constant Voronoi Plots',
                                          format='png'),        
                              )


plotter = aplt.Plotter(output=aplt.Output(path='/home/carlos/autolens_workspace/howtolens/Testes with MGE/Spherical Power-Law/Image Output/ConstVoronoi/',
                                          format='png'),         
                       )


include = aplt.Include(inversion_grid=False,
                       inversion_pixelization_grid=False,
                       inversion_border=True,
                       inversion_image_pixelization_grid=False,
                      )


# In[11]:


aplt.FitImaging.subplot_fit_imaging(fit=fit, 
                                    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
                                    sub_plotter=sub_plotter, 
                                    )



aplt.Inversion.individuals(inversion=fit.inversion, 
                           plotter=plotter,
                           include=include,
                           plot_interpolated_reconstruction=True, 
                           plot_errors=True,
                           plot_reconstruction=True, 
                           plot_residual_map=True, 
                           plot_chi_squared_map=True,
                           plot_normalized_residual_map=True,
                           plot_regularization_weight_map=True,
                               )


# In[12]:


aplt.FitImaging.subplot_fit_imaging(fit=fit, 
                                    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
                                    )

aplt.Inversion.individuals(inversion=fit.inversion,
                           include=aplt.Include(inversion_pixelization_grid=False),
                           plot_reconstruction=True, 
                           plot_regularization_weight_map=True,
                           plotter=aplt.Plotter(figure=aplt.Figure(figsize=(7,7)))
                               
                           )

print("Evidence using adaptive _Regularization_= ", fit.log_evidence)
frist_log = fit.log_evidence


# #################################################################################################################

# __Inversion with Adaptative Voronoi__

# In[ ]:


hyper_image = fit.model_image.in_1d_binned

source_adaptive_regularization = al.Galaxy(
    redshift=2.1,
    pixelization=al.pix.VoronoiMagnification(shape=(50, 50)),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.05, outer_coefficient=4, signal_scale=3.0
    ),
    hyper_galaxy_image=hyper_image,
)

fit = fit_masked_imaging_with_source_galaxy(
    masked_imaging=masked_imaging, source_galaxy=source_adaptive_regularization
)


# In[ ]:


sub_plotter = aplt.SubPlotter(output=aplt.Output(path='/home/carlos/autolens_workspace/howtolens/Testes with MGE/Spherical Power-Law/Image Output/AdaptativeVoronoi/',
                                          filename='Adaptative Voronoi Plots',
                                          format='png'),        
                              )


plotter = aplt.Plotter(output=aplt.Output(path='/home/carlos/autolens_workspace/howtolens/Testes with MGE/Spherical Power-Law/Image Output/AdaptativeVoronoi/',
                                          format='png'),         
                       )


include = aplt.Include(inversion_grid=False,
                       inversion_pixelization_grid=False,
                       inversion_border=True,
                       inversion_image_pixelization_grid=False,
                      )


# In[ ]:


aplt.FitImaging.subplot_fit_imaging(fit=fit, 
                                    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
                                    sub_plotter=sub_plotter, 
                                    )



aplt.Inversion.individuals(inversion=fit.inversion, 
                           plotter=plotter,
                           include=include,
                           plot_interpolated_reconstruction=True, 
                           plot_errors=True,
                           plot_reconstruction=True, 
                           plot_residual_map=True, 
                           plot_chi_squared_map=True,
                           plot_normalized_residual_map=True,
                           plot_regularization_weight_map=True,
                               )


# In[ ]:


aplt.FitImaging.subplot_fit_imaging(fit=fit, 
                                    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
                                    )

aplt.Inversion.individuals(inversion=fit.inversion,
                           include=aplt.Include(inversion_pixelization_grid=False),
                           plot_reconstruction=True, 
                           plot_regularization_weight_map=True,
                           plotter=aplt.Plotter(figure=aplt.Figure(figsize=(7,7)))
                               
                           )

print("Evidence using adaptive _Regularization_= ", fit.log_evidence)


# In[ ]:


print("Evidence using constant _Regularization_= ", frist_log)
print("Evidence using adaptive _Regularization_= ", fit.log_evidence)


# #################################################################################################################

# __Adaptative Voronoi with different regularizations__

# In[ ]:


source_adaptive_regularization = al.Galaxy(
    redshift=2.1,
    pixelization=al.pix.VoronoiMagnification(shape=(40, 40)),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy_image=hyper_image,
)

fit = fit_masked_imaging_with_source_galaxy(
    masked_imaging=masked_imaging, source_galaxy=source_adaptive_regularization
)


# In[ ]:


sub_plotter = aplt.SubPlotter(output=aplt.Output(path='/home/carlos/autolens_workspace/howtolens/Testes with MGE/Spherical Power-Law/Image Output/AdaptativeVoronoi2/',
                                          filename='Adaptative Voronoi Plots',
                                          format='png'),        
                              )


plotter = aplt.Plotter(output=aplt.Output(path='/home/carlos/autolens_workspace/howtolens/Testes with MGE/Spherical Power-Law/Image Output/AdaptativeVoronoi2/',
                                          format='png'),         
                       )


include = aplt.Include(inversion_grid=False,
                       inversion_pixelization_grid=False,
                       inversion_border=True,
                       inversion_image_pixelization_grid=False,
                      )


# In[ ]:


aplt.FitImaging.subplot_fit_imaging(fit=fit, 
                                    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
                                    sub_plotter=sub_plotter, 
                                    )



aplt.Inversion.individuals(inversion=fit.inversion, 
                           plotter=plotter,
                           include=include,
                           plot_interpolated_reconstruction=True, 
                           plot_errors=True,
                           plot_reconstruction=True, 
                           plot_residual_map=True, 
                           plot_chi_squared_map=True,
                           plot_normalized_residual_map=True,
                           plot_regularization_weight_map=True,
                               )


# In[ ]:


aplt.FitImaging.subplot_fit_imaging(fit=fit, 
                                    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
                                    )

aplt.Inversion.individuals(inversion=fit.inversion,
                           include=aplt.Include(inversion_pixelization_grid=False),
                           plot_reconstruction=True, 
                           plot_regularization_weight_map=True,
                           plotter=aplt.Plotter(figure=aplt.Figure(figsize=(7,7)))
                               
                           )

print("Evidence using adaptive _Regularization_= ", fit.log_evidence)

