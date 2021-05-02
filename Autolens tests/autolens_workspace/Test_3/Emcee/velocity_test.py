"""
Attention!
This code runs in MPI mode.
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
import CombinedModel

#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s


#Dataset path
data_folder = "/home/carlos/Documents/GitHub/Master-Degree/Autolens_tests/autolens_workspace/Test_3/Simulation_data/"

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
    """
    p0 = np.array([ml, beta[0], inc, np.log10(mbh), np.log10(rho_s), qdm[0], mag_shear, phi_shear, gamma])
    model(p0)

    #Finally we initialize the walkers arround these positions above.
    nwalkers = 120                                                   #Number of walkers
    pos = p0 +  (p0 * (0.5)) * np.random.randn(nwalkers, p0.size)                  #Initial guess of walkers
    print(pos)
    nwalkers, ndim = pos.shape                                      #Number of walkers/dimensions
    """
    #Lets try other type of initial position, spreading the the walkers uniformly over the range of parameters.
    nwalkers = 20                                                   #Number of walkers
    pos = np.random.uniform(low=[0.5, -3, 50, 6, 6, 0.2, 0, 0, 0.9], high=[15, 3, 90, 10, 12, 1, 0.1, 170, 1.1], size=[nwalkers, 9])
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



    it_time = np.zeros(2)
    burnin = 1
    for i in range(2):
        start = time.time()
        state = sampler.run_mcmc(pos, nsteps=burnin, progress=True)
        it_time[i] = time.time() - start
        sampler.reset()

    





np.savetxt("Saida.txt", it_time, fmt="%.10e", header="Time(s)")
    









    


    
