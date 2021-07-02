#Control time packages
import time
import os
import sys

from numpy.core import machar
os.environ["OMP_NUM_THREADS"] = "1"

import autolens as al
import autolens.plot as aplt
import numpy as np
import matplotlib.pyplot as plt

from plotbin.plot_velfield import plot_velfield


from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from schwimmbad import MPIPool



from My_Jampy import JAM                          #My class for jampy
import dark_mge                                   #My mge1d_fit for DM 
from scipy.special import ndtri

data_folder = "/home/carlos/Documents/GitHub/Master-Degree/SDP81/Autolens/Combined_Excluded_Spaxels/Data"#Reading MGE inputs

boundary = {"ml0": [0.5, 15], "delta": [0.1, 2], "lower": [0, 1],  "qinc": [0.051, 0.468808],
             "beta0": [-2, 2],"beta1": [-2, 2],"beta2": [-2, 2],"beta3": [-2, 2],"beta4": [-2, 2],"beta5": [-2, 2],"beta6": [-2, 2],"beta7": [-2, 2], "log_mbh": [7, 11],
             "kappa_s": [0, 2.0], "rs": [2.0, 50], "qDM": [0.1, 1], 
             "mag_shear": [0, 0.1],"phi_shear": [0, 179], "gamma": [0.95, 1.05] }

def gaussian_ml(sigma, delta, ml0=1.0, lower=0.4):
    """
    Create a M*L gradient
    Input:
    -----------
        sigma: Gaussian sigma                           [arcsec]
        delta: Gradient value
        ml0: Central stellar mass to light ratio        [M_sun/L_sun]    
        lower: the ratio between the central and the outer most M/L
    Output:
    ----------
        ML: gaussian mass to light ratio. One component per gaussian in surf_lum.
    """

    sigma = np.atleast_1d(sigma)
    sigma = sigma - sigma[0]
    ML = ml0 * (lower + (1-lower)*np.exp(-0.5 * (sigma * delta)**2))
    
    return ML

class Model(object):

    def __init__(self, mass_model, masked_image, Jam_model, mgeNFW):
        self.mass_profile  = mass_model
        self.masked_image  = masked_image
        self.Jam_model     = Jam_model
        self.mgeNFW        = mgeNFW

    def prior_transform(self, theta):
        (ml0, delta, lower, qinc, beta0,beta1,beta2,
                    beta3,beta4,beta5,beta6,beta7, log_mbh,
                    kappa_s, rs, qDM, mag_shear, phi_shear, gamma) = theta
        
        parsDic = {"ml0":ml0, "delta": delta, "lower": lower, "qinc":qinc, 
                    "beta0":beta0,"beta1":beta1,"beta2":beta2,"beta3":beta3,
                    "beta4":beta4,  "beta5":beta5,"beta6":beta6,"beta7":beta7,
                    "log_mbh":log_mbh, "kappa_s": kappa_s, "rs": rs, "qDM": qDM, 
                    "mag_shear": mag_shear, "phi_shear": phi_shear, "gamma": gamma}


        keys = set(parsDic.keys())                                      #All parameters
    
        #Frist the prior on gamma
        gamma_0      = 1.00       #mean
        sigma_gamma  = 0.05       #sigma/std
    
        parsDic['gamma'] = gamma_0 + sigma_gamma*ndtri(parsDic['gamma']) #Convert back to physical gamma
        excludes = set(['gamma'])  #Exclude gamma, because we already verify it above
        
        for key in keys.difference(excludes): #Loop over the remains parameters
            parsDic[key] = boundary[key][0] + parsDic[key]*(boundary[key][1] - boundary[key][0])
            
        return np.array(list(parsDic.values()))



    def log_likelihood(self, theta):
        (ml0, delta, lower, qinc, beta0,beta1,beta2,
                    beta3,beta4,beta5,beta6,beta7, log_mbh,
                    kappa_s, rs, qDM, mag_shear, phi_shear, gamma) = theta

        # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
        qmin = np.min(self.Jam_model.qobs_lum)
        inc_model = np.degrees(np.arctan(np.sqrt((1 - qmin**2)/(qmin**2 -qinc**2))))

        qdm_proj      = np.sqrt( (np.sin(np.radians(inc_model)) * qDM )**2  + np.cos( np.radians(inc_model))**2     )     #Projected DM axial ratio
        qDM_model     = np.ones_like(self.Jam_model.qobs_dm)*qdm_proj

        surf_dm_model, sigma_dm_model = self.mgeNFW.fit(kappa_s,rs,ngauss=15,outer_slope=2, quiet=True)

        beta_model  = np.array([beta0,beta0,beta1,beta2,beta3,beta4,beta5,beta6,beta7])
        ml_model    = gaussian_ml(sigma=self.Jam_model.sigma_lum, ml0=ml0, delta=delta, lower=lower)
        mbh_model   = 10**log_mbh

        
        # Pyautolens
        ell_comps = al.convert.elliptical_comps_from(axis_ratio=qdm_proj, phi=0.0) #Elliptical components in Pyautolens units
        eNFW = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=kappa_s,elliptical_comps=ell_comps, scale_radius=rs) #Set the analytical model
        self.mass_profile.Analytic_Model(eNFW)        #Include analytical model
        self.mass_profile.MGE_Updt_parameters(ml=ml_model, mbh=10**log_mbh, gamma=gamma)
        shear_comp_model = al.convert.shear_elliptical_comps_from(magnitude=mag_shear, phi=phi_shear)
        #New lens model
        lens_galaxy = al.Galaxy(                                            
                redshift=self.mass_profile.z_l,
                mass=self.mass_profile,
                shear=al.mp.ExternalShear(elliptical_comps=shear_comp_model),
            )

        source_galaxy = al.Galaxy(
                redshift=self.mass_profile.z_s,
                pixelization=al.pix.Rectangular(shape=(40, 40)),
                regularization=al.reg.Constant(coefficient=2.0),
            )
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
        
        try:
            fit = al.FitImaging(masked_imaging=self.masked_image, tracer=tracer)

            log_evidence = fit.log_evidence
        except:
            print("An exception ocurres in Pyautolens_log_likelihood().")
            return -np.inf

        
        # Jampy
        self.Jam_model.upt(inc=inc_model, ml=ml_model, beta=beta_model, mbh=mbh_model,
                            surf_dm=surf_dm_model, sigma_dm=sigma_dm_model, qobs_dm=qDM_model)
        
        rmsModel, ml, chi2, chi2T = self.Jam_model.run(quiet=True, plot=False)


        return log_evidence -0.5 * chi2T
    
    def __call__(self, pars):
        return self.log_likelihood(pars)

def resume_dlogz(sampler):
        results = sampler.results
        logz_remain = np.max(sampler.live_logl) + results.logvol[-1]
        delta_logz = np.logaddexp(results.logz[-1], logz_remain) - results.logz[-1]
        
        return delta_logz
        

with MPIPool() as pool:

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Reading Data
    x, y, vrms, erms              = np.loadtxt(data_folder+"/Input/Vrms_map_rot_BadAss_boots.txt", unpack=True)           #vrms data
    surf_lum, sigma_lum, qobs_lum = np.loadtxt(data_folder+"/Input/JAM_Input.txt", unpack=True)              #MGE decomposition
    norm_psf, sigma_psf           = np.loadtxt(data_folder+"/Input/MUSE_Psf_model.txt", unpack=True)         #PSF


    #Only for lensing modelling 
    z_l    = 0.299                                                         #Lens Redshift
    z_s    = 3.042                                                         #Source Redshift 
    D_l    = cosmo.angular_diameter_distance(z_l).value                    #Distance to lens [Mpc] 
    mbh    = 1e9                                                           #mass of black hole [log10(M_sun)]
    kappa_ = 0.075                                                         #kappa_s of DM profile
    r_s    = 11.5
    ml     = gaussian_ml(sigma_lum, delta=0.5, ml0=7.0, lower=0.4)         #mass to light ratio
    beta   = np.full_like(surf_lum, -0.15)                                 #anisotropy [ad]
    phi_shear = 88                                                         #Inclination of external shear [deg]
    mag_shear = 0.02                                                       #magnitude of shear
    shear_comp = al.convert.shear_elliptical_comps_from(magnitude=mag_shear, phi=phi_shear) #external shear
    inc     = 65                                                            #inclination [deg]
    inc_rad = np.radians(inc)
    qinc    = np.sqrt(np.min(qobs_lum)**2 - 
                            (1 - np.min(qobs_lum)**2)/np.tan(inc_rad)**2)   #Deprojected axial ratio for inclination

    pixsize = 0.2                                                          #MUSE pixel size

    #Autolens Data
    imaging = al.Imaging.from_fits(
            image_path=f"{data_folder}/Alma_with_lens_center.fits",
            noise_map_path=f"{data_folder}/rms_noise_map.fits",
            psf_path=f"{data_folder}/Alma_psf_rot.fits",
            pixel_scales=0.01,
            image_hdu=1, noise_map_hdu=1, psf_hdu=1,
        )

    mask        = al.Mask.from_fits( file_path=f"{data_folder}/mask.fits", 
                                    pixel_scales=imaging.pixel_scales)

    masked_image = al.MaskedImaging(imaging=imaging, mask=mask, inversion_uses_border=True)   #Masked image
    #aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

    #--------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------#
    # PYAUTOLENS MODEL
    #MGE mass profile

    #Initializing
    mass_profile = al.mp.MGE()
    ell_comps = al.convert.elliptical_comps_from(axis_ratio=0.85, phi=0.0) #Elliptical components in Pyautolens units
    eNFW      = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=kappa_, elliptical_comps=ell_comps, scale_radius=r_s) #pseudo elliptical NFW


    #Components

    mass_profile.MGE_comps(z_l=z_l, z_s=z_s, 
                        surf_lum=surf_lum, sigma_lum=sigma_lum, qobs_lum=qobs_lum, ml=ml,
                        mbh=mbh)
    mass_profile.Analytic_Model(analytic_profile=eNFW)



    #Lens galaxy
    lens_galaxy = al.Galaxy(
        redshift=z_l,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=shear_comp)
    )

    source_galaxy = al.Galaxy(
        redshift=z_s,
        pixelization=al.pix.Rectangular(shape=(40, 40)),
        regularization=al.reg.Constant(coefficient=2.00),
    )

    # JAMPY MODEL

    # Now we start our Jampy class
    Jam_model = JAM(ybin=y*pixsize, xbin=x*pixsize, inc=inc, distance=D_l, mbh=mbh, beta=beta, rms=vrms, erms=erms, normpsf=norm_psf, sigmapsf=sigma_psf*pixsize, pixsize=pixsize)

    #Add Luminosity component
    Jam_model.luminosity_component(surf_lum=surf_lum, sigma_lum=sigma_lum,
                                        qobs_lum=qobs_lum, ml=ml)
    #Add DM component
        #First initialize the NFW class. We are giving random numbers, because they will be updates during the fit. 
    mgeNFW = dark_mge.NFW(z_l=z_l, z_s=z_s, r_min=0.1, r_max=6.0, nsample=200, quiet=False)
    surf_dm, sigma_dm = mgeNFW.fit(kappa_s=0.075,rs=1.0,ngauss=15,outer_slope=2, quiet=True)
    
    Jam_model.DM_component(surf_dm=surf_dm, sigma_dm=sigma_dm, qobs_dm=np.ones_like(sigma_dm))

    print("Starting functions... \n")
    start = time.time()
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
    fit = al.FitImaging(masked_imaging=masked_image, tracer=tracer)

    #aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True,critical_curves=False,caustics=False))
    print("Log Likelihood with Regularization:", fit.log_likelihood_with_regularization)
    print("Log Evidence:", fit.log_evidence)
    print("Log Likelihood :", fit.log_likelihood)
    print("Elapsed Time [s]:", (time.time() - start))

    #Defing Model
    model = Model(mass_model=mass_profile, masked_image=masked_image,Jam_model=Jam_model, mgeNFW=mgeNFW)

    

    # (ml0, delta, lower, qinc, beta0,beta1,beta2,
    #                beta3,beta4,beta5,beta6,beta7, log_mbh,
    #                kappa_s, rs, qDM, mag_shear, phi_shear, gamma) = theta

    print("\n Testing Likelihood call...")
    start = time.time()
    p0 = np.array([7.0, 0.5, 0.4, qinc, beta[0], beta[0], 
                    beta[0], beta[0], beta[0], beta[0], beta[0], 
                    beta[0], np.log10(mbh), 1.0, 18, 0.85, 0.02, 100, 1.0])
    value = model(p0)
    print("Likelihood Call Value:", value)
    print("Likelihood Call Time:", (time.time() - start))

    from dynesty import NestedSampler
    import _pickle  as pickle
    import shutil


    original = r"sdp.pickle"
    beckup   = r"beckup/sdp_beckup.pickle"
    log_table = np.savetxt("Log.txt", np.column_stack([0.0,0.0,0.0,0.0]), 
                        header="Maxiter \t Maxcall \t dlogz \t Time[s]", 
                        fmt="%d \t\t %d \t\t %f \t %f")
    

    print("\n Number of CPUS:", pool.size)
    print("\n")
    nlive = 400             # number of (initial) live points
    ndim  = p0.size         # number of dimensions



    # Now run with the static sampler
    sampler = NestedSampler(model, model.prior_transform, ndim,
                                pool=pool,
                                nlive=nlive,walks=15,
                            )
    # These hacks are necessary to be able to pickle the sampler.
    sampler.rstate = np.random
    sampler.pool   = pool
    sampler.M      = pool.map

    print(sampler.sampling, sampler.walks)


    delta_logz = 1e200
    while delta_logz > 0.1:
        maxiter = 200
        start = time.time()

        sampler.run_nested(
                            maxiter=maxiter, 
                            dlogz=0.1,add_live=False,
                            print_progress=False
        )

        
        delta_logz = resume_dlogz(sampler)
        sampler.add_final_live(print_progress=False)
        sampler_pickle = sampler
        sampler_pickle.loglikelihood = None

        with open(f"sdp.pickle", "wb") as f:
            pickle.dump(sampler_pickle, f, -1)
            f.close()
        
        sampler_pickle.loglikelihood = model.log_likelihood

        
        # Performing Update
        original = r"sdp.pickle"
        beckup   = r"beckup/sdp_beckup.pickle" 

        beckup = shutil.copyfile(original, beckup)
        log_table = np.loadtxt("Log.txt")
        run_time = time.time() - start
        np.savetxt("Log.txt",
                    np.vstack([log_table,np.array([sampler.results.niter, sampler.results.ncall.sum(), delta_logz, run_time])]),
                    header="Maxiter \t Maxcall \t dlogz \t Time[s]", 
                    fmt="%d \t\t %d \t\t %f \t %f")
        
        print(f"\nniter: %d. ncall: %d. dlogz: %f." %(sampler.it, sampler.ncall,delta_logz))

    

    print(f"\nSaving Final Sample:")
    with open(f"final_dynesty_lens.pickle", "wb") as f:
        pickle.dump(sampler, f, -1) 
    print(f"\nSaved!!")     
