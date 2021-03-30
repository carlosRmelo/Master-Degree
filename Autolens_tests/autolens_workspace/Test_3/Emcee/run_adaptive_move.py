"""
Attention!
This code runs in MPI mode.

Beside that, this script is designed to be scale tuned, i.n, the walkers scale parameter could be updated every 100 walks. The reason behind that  is avoid very low (or very high) acceptance fractions.
This "tuned technique" is partially motivated by, https://github.com/pymc-devs/pymc3/blob/master/pymc3/step_methods/metropolis.py, but we find references for this in https://arxiv.org/abs/1311.5229 and https://link.springer.com/article/10.1007/s11222-008-9104-9.
"""



""""
    With this code, using Emcee, we want to find the input values used for simulation. To knhow:
    One ML, One beta, inclination, mbh, rho_s, r_s, mag_shear, phi_shear and gamma.
"""

#Control time packages
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"


#General packages
import numpy as np
from My_Jampy import JAM
import emcee

#MPI
from schwimmbad import MPIPool

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u

#Autolens Model packages
import autolens as al
import autolens.plot as aplt

#Combined Model package
import CombinedModel_transform_upd as CombinedModel

#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s


#Dataset path
data_folder = "/home/carlos/Documents/GitHub/Master-Degree/Autolens_tests/autolens_workspace/Test_3/Simulation_data/"

def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:
    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """
    if acc_rate < 0.001:
        # reduce by 90 percent
        return scale * 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        return scale * 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        return scale * 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        return scale * 10.0
    elif acc_rate > 0.75:
        # increase by double
        return scale * 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        return scale * 1.1

    return scale

#here we use the MPI
with MPIPool() as pool:

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

        #Reading MGE inputs
            #attention to units
    surf_lum, sigma_lum, qobs_lum = np.loadtxt("JAM_Input.txt", unpack=True)          #MGE decomposition
    surf_dm, sigma_dm , qobs_dm   = np.loadtxt("SDP81_pseudo-DM.txt", unpack=True)    #DM component
    norm_psf, sigma_psf           = np.loadtxt("MUSE_Psf_model.txt", unpack=True)     #PSF
    xbin, ybin, vrms              = np.loadtxt("vrms_data.txt", unpack=True)          #Vrms data
    
    muse_pixsize = 0.2                            #Muse pixel size [arcsec/px]

    z_lens   = 0.299                                    #Lens redshifth
    z_source = 3.100                                    #Source redshift

    #Angular diameter distances
    D_l = cosmo.angular_diameter_distance(z_lens)                   #Lens              
    D_s = cosmo.angular_diameter_distance(z_source)                 #Source
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)   #Lens to source

    ## Models inicialization

    """
        To inicialize the model, we set some random values for the parameters. But it's only necessary for initialize the model. During the non-linear search, this values will be updated constantly until the best fit.
    """  


       #This quantities are our unknown parameters
    inc       = 75                              #Inclination [deg]
    mbh       = 1e10                            #Mass of black hole [M_sun]
    beta      = np.full_like(surf_lum, 0.3)     #Anisotropy
    ml        = 10                              #Mass to light ratio [M_sun/L_sun]
    mag_shear = 0.01                            #Shear magnitude
    phi_shear = 100.0                             #Shear angle
    rho_s     = 1e10                            #dark matter intensity
    qdm       = np.full_like(qobs_dm, 0.5)      #dark matter axial ratio
    gamma     = 1.0                             #Gamma

    #--------------------------------------------------------------------------------------------------#
    # JAMPY MODEL

    Jam_model = JAM(ybin=ybin, xbin=xbin, inc=inc, distance=D_l.value, mbh=mbh, beta=beta, rms=vrms,
                   normpsf=norm_psf, sigmapsf=sigma_psf*muse_pixsize, pixsize=muse_pixsize)

        #Add Luminosity component
    Jam_model.luminosity_component(surf_lum=surf_lum, sigma_lum=sigma_lum,
                                    qobs_lum=qobs_lum, ml=ml)
        #Add DM component
    Jam_model.DM_component(surf_dm=rho_s * surf_dm, sigma_dm=sigma_dm, qobs_dm=qdm)


    #--------------------------------------------------------------------------------------------------#
    # PYAUTOLENS MODEL

    imaging = al.Imaging.from_fits(
        image_path=f"{data_folder}/arcs_simulation.fits",
        noise_map_path=f"{data_folder}/noise_simulation.fits",
        psf_path=f"{data_folder}/psf_simulation.fits",
        pixel_scales=0.1,
    )

    mask = al.Mask.circular_annular(centre=(0.0, -0.2), inner_radius=1., outer_radius=2.3,
                              pixel_scales=imaging.pixel_scales, shape_2d=imaging.shape_2d) #Create a mask

    masked_image = al.MaskedImaging(imaging=imaging, mask=mask, inversion_uses_border=True) #Masked image

    mass_profile = al.mp.MGE()

        #Components
    mass_profile.MGE_comps(z_l=z_lens, z_s=z_source, 
                       surf_lum=surf_lum, sigma_lum=sigma_lum, qobs_lum=qobs_lum, ml=ml,
                       mbh=mbh, surf_dm =rho_s * surf_dm, sigma_dm=sigma_dm, qobs_dm=qdm)

    #--------------------------------------------------------------------------------------------------#
    # COMBINED MODEL

        #Just remembering, by default the model does not include dark matter.
    model = CombinedModel.Models(Jampy_model=Jam_model, mass_profile=mass_profile,
                                 masked_imaging=masked_image, quiet=True)

    model.mass_to_light(ml_kind='scalar')               #Setting gradient ML
    model.beta(beta_kind='scalar')                      #Seting vector anisotropy
    model.has_DM(a=True,filename="SDP81_pseudo-DM.txt") #Setting Dark matter component
    #--------------------------------------------------------------------------------------------------#
    #  EMCEE
    """
        Pay close attention to the order in which the components are added. 
        They must follow the log_probability unpacking order.
    """

    #In order: ML, beta, inc, log_mbh, log_rho_s, qDM, mag_shear, phi_shear, gamma
   
    #Lets try other type of initial position, spreading the the walkers uniformly over the range of parameters.
    nwalkers = 40                                                   #Number of walkers
    pos = np.random.uniform(low=[-100, -100, -100, -100, -100, -100, -100, -100, -100], high=[100, 100, 100, 100, 100, 100, 100, 100, 100], size=[nwalkers, 9])
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

    #Backup
    filename = "simulation.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DEMove(gamma0=1.0), 0.20)] 

    #Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model, pool=pool,
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
    #End of burn in fase


    nsteps = 500000                          #Number of walkes 

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
        if sampler.iteration % 2:
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

        #If Not converged, change the scale factor of proposal walk, if needed
        sampler._moves[0].gamma0 = tune( 2.38 / np.sqrt(2 * ndim), mean_accp_100)
                    #Proposal value recommend by reference  2.38 / np.sqrt(2 * ndim)
        


    end = time.time()
    print('\n')
    print("Final")
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    


    
