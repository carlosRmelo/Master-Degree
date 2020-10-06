#Author: Carlos Roberto de Melo
#Date: 10/05/2020
#Obj:
'''
    Here we have classes to initialize each model (Jumpy and Auto Lens) individually.
'''

#Packages

#General packages
import numpy as np

#Own packages
import My_Jampy
import model_data
from model_data import Global_Parameters as GP 
from model_data import Jampy_data as JP_data
from model_data import Autolens_data as AL_data 

#Autolens Model packages
import autolens as al
import autolens.plot as aplt
from pyprojroot import here

#print("Pyautolens version:", al.__version__)
workspace_path = str(here())
#print("Workspace Path: ", workspace_path)


class Jampy():
    # JAMPY Model
    """
      This class contain only the initialization of Jampy model. To perform the fit, just run:
            Jampy_Model.run()  
    """
    #Initializing the dynamic model
    Jampy_Model = My_Jampy.Jam_axi_rms(ybin=JP_data.ybin, xbin=JP_data.xbin, beta=JP_data.beta, mbh=GP.mbh.value, distance=GP.distance.value, surf_lum=GP.surf_star_dat.value, sigma_lum=GP.sigma_star_dat_ARC.value, qobs_lum=GP.qstar_dat, surf_DM=GP.surf_DM_dat.value, sigma_DM=GP.sigma_DM_dat_ARC.value, qobs_DM=GP.qDM_dat, ml=GP.ML.value, goodBins=JP_data.goodBins, sigmapsf=JP_data.sigmapsf, rms=JP_data.rms, erms=JP_data.erms, pixsize=JP_data.pixsize, inc=GP.inc, quiet=True)

class Autolens():
    """
        This class iniziale the lens model, and also read the arcs data and mask. Beside that, we can choose plot the arcs image, for that just set plot keyword as "True".
    """

    def imaging_data(plot=False):
        ### __Reading image data__

        #Paths
        dataset_type = "JAM+Pyautolens"
        dataset_name = "Data"
        dataset_path = f"/home/carlos/Documents/GitHub/Master-Degree/ESO325/Arcs Modelling/autolens_workspace/{dataset_type}/{dataset_name}"

        #Load data
        imaging = al.Imaging.from_fits(
            image_path=f"{dataset_path}/arcs_resized.fits",
            noise_map_path=f"{dataset_path}/noise_map_resized.fits",
            psf_path=f"{dataset_path}/psf.fits",
            pixel_scales=0.04,
        )

        #Load mask
        mask_custom = al.Mask.from_fits(
            file_path=f"{dataset_path}/mask gui.fits", hdu=0, pixel_scales=imaging.pixel_scales
        )

        masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask_custom)

        #Plot
        if plot is True:
            aplt.Imaging.subplot_imaging(imaging=imaging,mask=mask_custom,include=aplt.Include(border=True))

        return masked_imaging  

        
    def mass_model():
        ### __Defining the MGE mass model__

        #Initializing the MGE  model for lens

        mass_profile = al.mp.MGE(centre=(0.0, 0.0))                               #Defining the mass model
        mass_profile.MGE_comps(M=AL_data.Total_Mass.value, sigma=AL_data.Total_sigma_RAD.value, q=AL_data.Total_q_proj.value, z_l=GP.z_lens, z_s=GP.z_source)        #Input data

        mass_profile.MGE_Grid_parameters(self.imaging_data().grid)               #Creating the parameter grid for the parallel calculation

        return mass_profile