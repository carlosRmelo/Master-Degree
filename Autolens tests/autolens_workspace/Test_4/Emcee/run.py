#!/usr/bin/env python
# coding: utf-8

# 
# ## Attention!
# ### This code runs in MPI mode.
# 
# 
# Trying to recover the input values of the simulation. The free parameters are:
#    - One ML, One beta, qinc, mbh, kappa_s, qDm, mag_shear, phi_shear and gamma.
# 

# In[1]:


#Control time packages
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"

#MPI
from schwimmbad import MPIPool

#General packages
import numpy as np
from My_Jampy import JAM
import emcee
import matplotlib.pyplot as plt

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.constants import G, M_sun, c
import astropy.units as u

#Autolens Model packages
import autolens as al
import autolens.plot as aplt

#Combined Model package
import CombinedModel

data_folder = "/home/carlos/Documents/GitHub/Master-Degree/Autolens tests/autolens_workspace/Test_4/Simulation_Data/"


# In[ ]:


#here we use the MPI
with MPIPool() as pool:

    if not pool.is_master():
        pool.wait()
        sys.exit(0)
   
        #Reading MGE inputs
    surf_lum, sigma_lum, qobs_lum = np.loadtxt("Input/JAM_Input.txt", unpack=True)      #MGE decomposition
    surf_dm, sigma_dm , qobs_dm   = np.loadtxt("Input/eNFW.txt", unpack=True)           #DM component
    norm_psf, sigma_psf           = np.loadtxt("Input/MUSE_Psf_model.txt", unpack=True) #PSF
    x, y, vrms, erms              = np.loadtxt("Input/vrms_data.txt", unpack=True)      #vrms data

    #Only for lensing modelling 
    z_l    = 0.299                                                         #Lens Redshift
    z_s    = 4.100                                                         #Source Redshift 
    D_l    = cosmo.angular_diameter_distance(z_l).value                    #Distance to lens [Mpc] 
    mbh    = 1e9                                                           #mass of black hole [log10(M_sun)]
    kappa_ = 0.075                                                         #kappa_s of DM profile
    r_s    = 11.5
    ml     = 7.00                                                          #mass to light ratio
    phi_shear = 88                                                         #Inclination of external shear [deg]
    mag_shear = 0.02                                                       #magnitude of shear
    shear_comp = al.convert.shear_elliptical_comps_from(magnitude=mag_shear, phi=phi_shear) #external shear

    beta    = np.full_like(surf_lum, -0.15)                                 #anisotropy [ad]
    inc     = 65                                                            #inclination [deg]
    inc_rad = np.radians(inc)
    qinc    = np.sqrt(np.min(qobs_lum)**2 - 
                        (1 - np.min(qobs_lum)**2)/np.tan(inc_rad)**2)       #Deprojected axial ratio for inclination
    qDM     = np.sqrt( qobs_dm[0]**2 - np.cos(inc_rad)**2)/np.sin(inc_rad)  #Deprojected DM axial ratio
    pixsize = 0.2                                                           #MUSE pixel size
    
    #Autolens Data
    imaging = al.Imaging.from_fits(
        image_path=f"{data_folder}/arcs_simulation.fits",
        noise_map_path=f"{data_folder}/noise_simulation.fits",
        psf_path=f"{data_folder}/psf_simulation.fits",
        pixel_scales=0.1,
    )

    mask        = al.Mask.from_fits( file_path=f"{data_folder}/new_mask.fits", hdu=1, 
                                        pixel_scales=imaging.pixel_scales)#You should check Mask_Maker folder before continue

    masked_image = al.MaskedImaging(imaging=imaging, mask=mask, inversion_uses_border=True)     #Masked image


    #aplt.Imaging.subplot_imaging(
    #    imaging=imaging, mask=mask, include=aplt.Include(border=True),                         #Plot
    #)
    #--------------------------------------------------------------------------------------------------#
    # JAMPY MODEL
    #Now we start our Jampy class
    Jam_model = JAM(ybin=y*pixsize, xbin=x*pixsize, inc=inc, distance=D_l, mbh=mbh, beta=beta, rms=vrms, erms=erms,
                   normpsf=norm_psf, sigmapsf=sigma_psf*pixsize, pixsize=pixsize)

    #Add Luminosity component
    Jam_model.luminosity_component(surf_lum=surf_lum, sigma_lum=sigma_lum,
                                        qobs_lum=qobs_lum, ml=ml)

    #Add DM component
    Jam_model.DM_component(surf_dm=kappa_ * surf_dm, sigma_dm=sigma_dm, qobs_dm=qobs_dm)
    #--------------------------------------------------------------------------------------------------#
    # PYAUTOLENS MODEL
    
    ell_comps = al.convert.elliptical_comps_from(axis_ratio=qobs_dm[0], phi=0.0) #Elliptical components in Pyautolens units
    eNFW      = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=kappa_, elliptical_comps=ell_comps, scale_radius=r_s) #elliptical NFW

    mass_profile = al.mp.MGE()                            #MGE mass model
    mass_profile.Analytic_Model(analytic_profile=eNFW)   #Include the analytical pENFW

    #Components
    mass_profile.MGE_comps(z_l=z_l, z_s=z_s, 
                        surf_lum=surf_lum, sigma_lum=sigma_lum, qobs_lum=qobs_lum, ml=ml,
                        mbh=mbh) #DON'T INCLUDE THE MGE PARAMETRIZATION OF DM

 
    #Lens galaxy
    lens_galaxy = al.Galaxy(
        redshift=z_l,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=shear_comp)
    )
    #--------------------------------------------------------------------------------------------------#
    # COMBINED MODEL
    #Starting Model
    model_emcee = CombinedModel.Models(Jampy_model=Jam_model, mass_profile=mass_profile,
                                    masked_imaging=masked_image, quiet=True)

    #Setup Configurations

    model_emcee.mass_to_light(ml_kind='scalar')                                          #Setting scalar ML
    model_emcee.beta(beta_kind='scalar')                                                 #Seting vector anisotropy
    model_emcee.has_MGE_DM(a=True, filename="Input/eNFW.txt", include_MGE_DM="Dynamical")#Setting Dark matter component
    model_emcee.include_DM_analytical(analytical_DM=eNFW)    
    #--------------------------------------------------------------------------------------------------#
    #  EMCEE
    """
        Pay close attention to the order in which the components are added. 
        They must follow the log_probability unpacking order.
    """

    #In order: ML, beta, qinc, log_mbh, kappa_s, qDM, mag_shear, phi_shear, gamma 
    p0 = np.array([ml, beta[0], qinc, np.log10(mbh), kappa_, qDM, mag_shear, phi_shear, 1.00]) #Best fit
    nwalkers = 120                                                  #Number of walkers
    sigma    = np.ones_like(p0) * 0.2                               #Assuming a 0.2 sigma around the best fit
    initial  = np.random.normal(p0, sigma, size=(120, p0.size))     #Initial walkers around best fit
    nwalkers, ndim = initial.shape                                      #Number of walkers/dimensions
    
    """
        We save the results in a table.
        This table marks the number of iterations, the mean acceptance fraction,the running time, and the mean accep. fraction of last 100 its. 
        
    """

    np.savetxt('Output_LogFile.txt', np.column_stack([0, 0, 0, 0]),
                                fmt=b'	%i	 %e			 %e     %e', 
                                header="Output table for the combined model: Dynamic.\n Iteration	 Mean acceptance fraction	 Processing Time    Last 100 Mean Accp.")
    #Print the number os cores/workers
    print("Workers nesse job:", pool.workers)
    print("Start")
    
    #This lines only check if the inputs are ok
    model_emcee(p0)


    #Backup
    filename = "Simulation4.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    moves=[(emcee.moves.DEMove(), 0.80), (emcee.moves.DEMove(gamma0=0.80), 0.20)]

    #Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model_emcee, pool=pool,
                                     backend=backend, moves=moves)
    
    #Burn in fase
    burnin = 1                           #Number of burn in steps
    print("Burn in with %i steps"%burnin)
    start = time.time()
    state = sampler.run_mcmc(initial, nsteps=burnin, progress=True)
    print("\n")
    print("Burn in elapsed time:", time.time() - start)
    sampler.reset()
    print("\n")
    print("End of burn-in fase")
    
    #End of burn in fase
    
    nsteps = 50000                          #Number of walkes 

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)
    # This will be useful to testing convergence
    old_tau = np.inf
    # This saves how many walkers have been accepted in the last 100 steps
    old_accp = np.zeros(nwalkers,)

    # Now we'll sample for up to max_n steps
    start = time.time()
    global_time = time.time()
    
    for sample in sampler.sample(initial, iterations=nsteps, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        print("\n")
        print("##########################")

        #Compute how many walkes have been accepted during the last 100 steps

        new_accp = sampler.backend.accepted             #Total number of accepted
        old_accp = new_accp - old_accp                  #Number of accepted in the last 100 steps
        mean_accp_100 = np.mean(old_accp/float(100))    #Mean accp fraction of last 100 steps

        #Update a table output with acceptance
        table = np.loadtxt("Output_LogFile.txt")

        iteration = sampler.iteration
        accept = np.mean(sampler.acceptance_fraction)
        total_time = time.time() - global_time
        upt = np.column_stack([iteration, accept, total_time, mean_accp_100])

        np.savetxt('Output_LogFile.txt', np.vstack([table, upt]),
                                fmt=b'	%i	 %e			 %e             %e', 
                            header="Output table for the combined model: Dynamic.\n Iteration	 Mean acceptance fraction	 Processing Time    Last 100 Mean Accp. Fraction")

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            if 0.2 < accept < 0.35:
                break
        old_tau = tau



    end = time.time()
    print('\n')
    print("Final")
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

