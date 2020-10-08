

#General packages
import numpy as np
import emcee
import matplotlib.pyplot as plt

#Own packages
import My_Jampy
import model_data
from model_data import Global_Parameters as GP 
from model_data import Jampy_data as JP_data
from model_data import Autolens_data as AL_data
import emcee_probabilities
from emcee_probabilities import Probability
from emcee_probabilities import prior 

#MPI packages and control paralelization
from time import perf_counter as clock
from multiprocessing import Pool
from schwimmbad import MPIPool
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"

#Autolens Model packages

import autolens as al
import autolens.plot as aplt
#print("Pyautolens version:", al.__version__)

from pyprojroot import here

from time import perf_counter as clock

from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u
workspace_path = str(here())
#print("Workspace Path: ", workspace_path)

#--------------------------------------------------------------------------------------------------------#
# JAMPY MODEL

#Initializing the dynamic model
Jampy_Model = My_Jampy.Jam_axi_rms(ybin=JP_data.ybin, xbin=JP_data.xbin, beta=JP_data.beta, mbh=GP.mbh.value, distance=GP.distance.value, surf_lum=GP.surf_star_dat.value, sigma_lum=GP.sigma_star_dat_ARC.value, qobs_lum=GP.qstar_dat, surf_DM=GP.surf_DM_dat.value, sigma_DM=GP.sigma_DM_dat_ARC.value, qobs_DM=GP.qDM_dat, ml=GP.ML.value, goodBins=JP_data.goodBins, sigmapsf=JP_data.sigmapsf, rms=JP_data.rms, erms=JP_data.erms, pixsize=JP_data.pixsize, inc=GP.inc, quiet=False)
                                
#--------------------------------------------------------------------------------------------------------------#

# Pyautolens Model

### __Reading simulated data__

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
'''
aplt.Imaging.subplot_imaging(
    imaging=imaging, mask=mask_custom, include=aplt.Include(border=True)
)
'''

### __Defining the MGE mass model__

#Initializing the MGE  model for lens

mass_profile = al.mp.MGE(centre=(0.0, 0.0))                               #Defining the mass model
mass_profile.MGE_comps(M=AL_data.Total_Mass.value, sigma=AL_data.Total_sigma_RAD.value, q=AL_data.Total_q_proj.value, z_l=GP.z_lens, z_s=GP.z_source)        #Input data

mass_profile.MGE_Grid_parameters(masked_imaging.grid, quiet=False)               #Creating the parameter grid for the parallel calculation

#Lens Model
lens_galaxy = al.Galaxy(                                            
        redshift=GP.z_lens,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=(0,0)),
    )

#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ EMCEE -----------------------------------------------------#

##### For the initial guesses we will use the Collett's best fit with an addition of a random noise between 0 and 1. This allow us probe faster our code.


np.savetxt('Output LogFile.txt', np.column_stack([0, 0, 0]),
                            fmt=b'	%i	 %e			 %e	 ', 
                            header="Output table for the combined model: Lens + Dynamic.\n Iteration	 Mean acceptance fraction	 Processing Time")

np.savetxt("LogFile_LastFit.txt", np.column_stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]),
                            fmt=b'%e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	', 
                            header="Iteration	 ML1/2	 ML3	 ML4	 ML5	 ML6	 ML7	 b1	 b2	 b3	 b4	 b5	 b6	 b7	 Inc	 qDM	 Logrho_s	 LogMBH	 MagShear	 PhiShear	 gamma")

#Initializing the probabilities
model = Probability(Jampy_Model=Jampy_Model, mass_profile=mass_profile, masked_imaging=masked_imaging)


with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
        
    np.random.seed(42)
    
    #Defining initial guesses
    nwalkers = 100
    ml = np.zeros((nwalkers,6))
    ml[:] = np.array([9.5,8.5,3.8,3.4,3.2,2.8])
    ml_noise = np.random.rand(nwalkers,6)

    beta = np.zeros((nwalkers,7))
    beta[:] = np.array([-0.6, -1.0, 0.34, -3.4, 0.39, -0.31, 0.36])
    beta_noise = np.random.rand(nwalkers,7)

    ml = ml + ml_noise                                                               #Between [2.8, 10.5]
    beta = beta + beta_noise                                                         #Between [-3.4, 1.39]
    inc = prior ['inc'][0] + (np.random.rand(nwalkers,1) -0.5)*10                          #Between [85, 95]
    qDM = np.random.rand(nwalkers,1)*0.74+0.26                                             #Between [0.26, 1]
    log_rho_s = np.random.rand(nwalkers,1)*prior['log_rho_s'][0]                           #Between [0, 10]
    log_mbh =  np.random.rand(nwalkers,1)+9.0                                              #Between [9.0, 10]
    mag_shear = (np.random.rand(nwalkers,1) - 0.5)*0.2                                     #Between [-0.1, 0.1]
    phi_shear = (np.random.rand(nwalkers,1) - 0.5)*0.2                                     #Between [-0.1, 0.1]
    gamma = (np.random.rand(nwalkers,1) - 0.5)*2                                           #Between [-1, 1]
    #50 walkers in a 21-D space
    pos = np.append(ml, beta, axis=1)
    pos = np.append(pos, inc, axis=1)
    pos = np.append(pos, qDM, axis=1)
    pos = np.append(pos, log_rho_s, axis=1)
    pos = np.append(pos, log_mbh, axis=1)
    pos = np.append(pos, mag_shear, axis=1)
    pos = np.append(pos, phi_shear, axis=1)
    pos = np.append(pos, gamma, axis=1)

    nwalkers, ndim = pos.shape
    
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "JAMPY+AUTOLENS-CLUSTER.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.log_probability, backend=backend, pool=pool)
    
    max_n = 5
    
    global_time = clock()
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    
    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 2 steps
        if sampler.iteration % 2:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1



        #Update a table output with acceptance
        table = np.loadtxt("Output LogFile.txt")


        iteration = sampler.iteration
        accept = np.mean(sampler.acceptance_fraction)
        total_time = clock() - global_time
        upt = np.column_stack([iteration, accept, total_time])

        np.savetxt('Output LogFile.txt', np.vstack([table, upt]),
                                fmt=b'	%i	 %e			 %e	 ', 
                                header="Iteration	 Mean acceptance fraction	 Processing Time")

        #Update table output with last best fit
        last_fit_table = np.loadtxt("LogFile_LastFit.txt")
        flat_samples = sampler.get_chain()
        values = []
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            values.append(mcmc[1])

        values = np.array(values)
        upt = np.append(iteration, values)

        np.savetxt("LogFile_LastFit.txt", np.vstack([last_fit_table, upt]),
                            fmt=b'%e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	', 
                            header="Iteration	 ML1/2	 ML3	 ML4	 ML5	 ML6	 ML7	 b1	 b2	 b3	 b4	 b5	 b6	 b7	 Inc	 qDM	 Logrho_s	 LogMBH	 MagShear	 PhiShear	 gamma")
 

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
tau = sampler.get_autocorr_time()
print(tau)
    
    

