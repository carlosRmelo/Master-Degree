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

#My Emcee for Pyautolens
import My_Autolens

data_folder = "/home/carlos/Documents/GitHub/Master-Degree/Autolens tests/autolens_workspace/Test_5/Simulation_Data/"

# In[ ]:


#here we use the MPI
with MPIPool() as pool:

    if not pool.is_master():
        pool.wait()
        sys.exit(0)
   
        #Reading MGE inputs
    surf_lum, sigma_lum, qobs_lum = np.loadtxt("Input/JAM_Input.txt", unpack=True)      #MGE decomposition
    surf_dm, sigma_dm , qobs_dm   = np.loadtxt("Input/eNFW.txt", unpack=True)            #DM component

    ## Models inicialization

    """
        To inicialize the model, we set some random values for the parameters. But it's only necessary for initialize the model. During the non-linear search, this values will be updated constantly until the best fit.
    """  
    #Only for lensing modelling 
    z_l    = 0.299                                                         #Lens Redshift
    z_s    = 4.100                                                         #Source Redshift 
    D_l    = cosmo.angular_diameter_distance(z_l).value                    #Distance to lens [Mpc] 
    mbh    = 1e9                                                           #mass of black hole [log10(M_sun)]
    kappa_ = 0.075                                                         #kappa_s of DM profile
    ml     = 7.00                                                          #mass to light ratio
    r_s    = 11.5                                                          #scale radius [arcsec]
    shear_comp = al.convert.shear_elliptical_comps_from(magnitude=0.02, phi=88) #external shear
    
    
    #Autolens Data
    imaging = al.Imaging.from_fits(
            image_path=f"{data_folder}/arcs_simulation.fits",
            noise_map_path=f"{data_folder}/noise_simulation.fits",
            psf_path=f"{data_folder}/psf_simulation.fits",
            pixel_scales=0.1,
        )

    mask        = al.Mask.from_fits( file_path=f"{data_folder}/new_mask.fits", hdu=1, 
                                    pixel_scales=imaging.pixel_scales)

    masked_image = al.MaskedImaging(imaging=imaging, mask=mask, inversion_uses_border=True)   #Masked image
    
    #--------------------------------------------------------------------------------------------------#
    # PYAUTOLENS MODEL
    #MGE mass profile
    mass_profile = al.mp.MGE()    #Mass class

    ell_comps    = al.convert.elliptical_comps_from(axis_ratio=qobs_dm[0], phi=0.0) #Elliptical components in Pyautolens units
    eNFW         = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=kappa_, elliptical_comps=ell_comps ,scale_radius=r_s) #Analytical eNFW profile


    #Components
    #Do not include MGE DM component here
    mass_profile.MGE_comps(z_l=z_l, z_s=z_s, 
                           surf_lum=surf_lum, sigma_lum=sigma_lum, qobs_lum=qobs_lum, ml=ml, mbh=mbh) 
    mass_profile.Analytic_Model(eNFW)  #Include Analytical NFW

    #--------------------------------------------------------------------------------------------------#
    #Emcee Model
    emcee_model = My_Autolens.Models(mass_profile=mass_profile, masked_imaging=masked_image, quiet=True)
    emcee_model.include_DM_analytical(eNFW)
    
    #  EMCEE
    """
        Pay close attention to the order in which the components are added. 
        They must follow the log_probability unpacking order.
    """

    #In order: ml, kappa_s, qDM, log_mbh, mag_shear, phi_shear, gamma 
    nwalkers = 120                                                  #Number of walkers
    #We distribute the initial position of walkers using a uniform distribution over all the possible values.
    pos = np.random.uniform(low=[1, 0, 0.3, 7, 0, 0, 0.9], high=[10, 1, 1, 10, 0.1, 179, 1.1], size=[nwalkers, 7])
    nwalkers, ndim = pos.shape                                      #Number of walkers/dimensions
    
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
    """
    #This lines only check if the inputs are ok
    p0     = np.array(([7.0, 0.075, qobs_dm[0], 9.0, 0.02, 88, 1.0]))            #All parameters
    emcee_model(p0)
    """

    #Backup
    filename = "Lens_Simulation.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    moves=[(emcee.moves.DEMove(), 0.80), (emcee.moves.DEMove(gamma0=1.0), 0.20)]

    #Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_model, pool=pool,
                                     backend=backend, moves=moves)
    
    #Burn in fase
    burnin = 1                           #Number of burn in steps
    print("Burn in with %i steps"%burnin)
    start = time.time()
    state = sampler.run_mcmc(pos, nsteps=burnin, progress=True)
    print("\n")
    print("Burn in elapsed time:", time.time() - start)
    sampler.reset()
    print("\n")
    print("End of burn-in fase")
    
    print("\n")
    print("Testing velocity of 1 step with %i walkers."%nwalkers)
    start = time.time()
    state = sampler.run_mcmc(pos, nsteps=burnin, progress=True)
    print("\n")
    print("Test elapsed time:", time.time() - start)
    sampler.reset()
    print("\n")
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
    
    for sample in sampler.sample(state, iterations=nsteps, progress=True):
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


