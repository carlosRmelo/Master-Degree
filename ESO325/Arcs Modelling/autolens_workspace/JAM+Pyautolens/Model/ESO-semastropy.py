#!/usr/bin/env python
# coding: utf-8

# In[1]:


import autolens as al
import autolens.plot as aplt
import time
import numpy as np


# In[2]:


surf_star_dat, sigma_star_dat, qstar_dat = np.loadtxt('JAM Input.txt', unpack=True) #Star
surf_DM_dat, sigma_DM_dat, qDM_dat = np.loadtxt('gProfile-DM Input.txt', unpack=True) #DM


# In[5]:




z_lens = 0.035
z_source = 2.1
D_l=127
                       


# In[6]:


inc = 90.                                                                    
distance = 127                                               
mbh =  1e8                                          
beta = np.zeros(surf_star_dat.shape)                          
ML = 3.19          


surf_DM_dat = surf_DM_dat                                   
sigma_DM_dat_ARC = sigma_DM_dat                                         
sigma_DM_dat_PC = (sigma_DM_dat_ARC*D_l)      
qDM_dat = qDM_dat                                                               

surf_star_dat = surf_star_dat                                
sigma_star_dat_ARC = sigma_star_dat                                    
sigma_star_dat_PC = (sigma_star_dat_ARC*D_l) 
qstar_dat = qstar_dat                                                          


# In[7]:


Mass_DM_dat = 2*np.pi*surf_DM_dat*(sigma_DM_dat_PC**2)*qDM_dat
Lum_star_dat = 2*np.pi*surf_star_dat*(sigma_star_dat_PC**2)*qstar_dat
Mass_star_dat = Lum_star_dat*ML

sigmaBH_ARC = 0.01
sigmaBH_PC = (sigmaBH_ARC*D_l)
surfBH_PC = mbh/(2*np.pi*sigmaBH_PC**2)
qSMBH = 1.                                                               
Mass_SMBH_dat = 2*np.pi*surfBH_PC*(sigmaBH_PC**2)*qSMBH 

i = np.deg2rad(inc)                                                             
Total_Mass = np.concatenate((Mass_star_dat, Mass_DM_dat, Mass_SMBH_dat), axis=None)   
Total_q = np.concatenate((qstar_dat, qDM_dat, qSMBH), axis=None) 

Total_q_proj = np.sqrt(Total_q**2 - np.cos(i)**2)/np.sin(i)                                       
Total_sigma_ARC = np.concatenate((sigma_star_dat_ARC, sigma_DM_dat_ARC, sigmaBH_ARC), axis=None)  
Total_sigma_RAD = Total_sigma_ARC                  


# In[8]:


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


# In[11]:


#aplt.Imaging.subplot_imaging(
#    imaging=imaging, mask=mask_custom, include=aplt.Include(border=True)
#)


# In[9]:


mass_profile = al.mp.MGE(centre=(0.0, 0.0))                         
mass_profile.MGE_comps(M=Total_Mass, sigma=Total_sigma_RAD,
                       q=Total_q_proj, z_l=z_lens, z_s=z_source)        

mass_profile.MGE_Grid_parameters(masked_imaging.grid)              
                                                                         

lens_galaxy = al.Galaxy(                                            
        redshift=0.035,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=(0,0)),
    )


# In[10]:


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




