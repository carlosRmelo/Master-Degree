#Control time packages
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np
import matplotlib.pyplot as plt
from plotbin.plot_velfield import plot_velfield
from My_Jampy import JAM                          #My class for jampy


from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from schwimmbad import MPIPool

boundary = {"ml0": [0.5, 15], "delta": [0.1, 2], "lower": [0, 1],  
                "qinc": [0.051, 0.468808], "beta": [-3, 3],
                "log_mbh": [7, 11]}

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
    def __init__(self, Jam_model):
        self.Jam_model = Jam_model

    def prior_transform(self, theta):
        ml0, delta, lower, qinc, beta, log_mbh = theta
        parsDic = {"ml0":ml0, "delta": delta, "lower": lower, "qinc":qinc, 
                "beta":beta, "log_mbh":log_mbh}
        for key in parsDic:
            parsDic[key] = boundary[key][0] + parsDic[key]*(boundary[key][1] - boundary[key][0])
        
        return np.array(list(parsDic.values()))

    def log_likelihood(self, theta):
        ml0, delta, lower, qinc, beta, log_mbh = theta = theta  
        # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
        qmin = np.min(self.Jam_model.qobs_lum)
        inc_model = np.degrees(np.arctan(np.sqrt((1 - qmin**2)/(qmin**2 -qinc**2))))
        

        beta_model  = np.ones_like(self.Jam_model.qobs_lum)*beta
        ml_model    = gaussian_ml(sigma=self.Jam_model.sigma_lum, ml0=ml0, delta=delta, lower=lower)
        mbh_model   = 10**log_mbh
        
        self.Jam_model.upt(inc=inc_model, ml=ml_model, beta=beta_model, mbh=mbh_model)
        
        rmsModel, ml, chi2, chi2T = self.Jam_model.run(quiet=True, plot=False)
        
        
        return -0.5 * chi2T

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
    x, y, vrms, erms              = np.loadtxt("Input/Vrms_map_rot.txt", unpack=True)           #vrms data
    surf_lum, sigma_lum, qobs_lum = np.loadtxt("Input/JAM_Input.txt", unpack=True)              #MGE decomposition
    norm_psf, sigma_psf           = np.loadtxt("Input/MUSE_Psf_model.txt", unpack=True)         #PSF

    # Defing some inputs
    z_l     = 0.299                                                         #Redshift
    D_l     = cosmo.angular_diameter_distance(z_l).value                    #Distance to Lens [Mpc] 
    mbh     = 1e9                                                           #mass of black hole [log10(M_sun)]
    beta    = np.full_like(surf_lum, -0.15)                                 #anisotropy [ad]
    inc     = 65                                                            #inclination [deg]
    ml      = gaussian_ml(sigma_lum, delta=0.5, ml0=7.0, lower=0.4)         #mass to light ratio
    inc_rad = np.radians(inc)
    qinc    = np.sqrt(np.min(qobs_lum)**2 - 
                            (1 - np.min(qobs_lum)**2)/np.tan(inc_rad)**2)   #Deprojected axial ratio for inclination

    pixsize = 0.2    #MUSE pixel size

    # Now we start our Jampy class
    Jam_model = JAM(ybin=y*pixsize, xbin=x*pixsize, inc=inc, distance=D_l, mbh=mbh, beta=beta, rms=vrms, erms=erms, normpsf=norm_psf, sigmapsf=sigma_psf*pixsize, pixsize=pixsize)

    #Add Luminosity component
    Jam_model.luminosity_component(surf_lum=surf_lum, sigma_lum=sigma_lum,
                                        qobs_lum=qobs_lum, ml=ml)

    model = Model(Jam_model=Jam_model)

    print("Testing Inputs...")
    #ML0, delta, lower, qinc, beta, log_mbh
    p0 = np.array([7.0, 0.5, 0.4, qinc, beta[0], np.log10(mbh)])
    print("Likelihood Value:", model(p0))

    from dynesty import NestedSampler
    import _pickle  as pickle
    import time

    original = r"dynesty_jam.pickle"
    log_table = np.savetxt("Log.txt", np.column_stack([0.0,0.0,0.0,0.0]), 
                        header="Maxiter \t Maxcall \t dlogz \t Time[s]", 
                        fmt="%d \t\t %d \t\t %f \t %f")
    
    print("\n Number of CPUS:", pool.size)
    print("\n")
    nlive = 150             # number of (initial) live points
    ndim  = p0.size         # number of dimensions
    sampling = "rwalk"      # sampling method



    # Now run with the static sampler
    sampler = NestedSampler(model, model.prior_transform, ndim,
                                pool=pool,walks=10,
                                nlive=nlive,
                            )
    print(sampler.sampling)
    # These hacks are necessary to be able to pickle the sampler.
    sampler.rstate = np.random
    sampler.pool   = pool
    sampler.M      = pool.map


    delta_logz = 1e200
    while delta_logz > 0.1:
        maxcall = 1500
        start = time.time()

        sampler.run_nested(
                            maxcall=maxcall, 
                            dlogz=0.1,add_live=False,
                            print_progress=False
        )

        
        delta_logz = resume_dlogz(sampler)
        sampler.add_final_live()
        sampler_pickle = sampler
        sampler_pickle.loglikelihood = None
        print("__________________________________________________________")
        with open(f"jam_gaussianML_noDM.pickle", "wb") as f:
            pickle.dump(sampler_pickle, f, -1)
            f.close()
        
        sampler_pickle.loglikelihood = model.log_likelihood

        
        # Performing Update)
        log_table = np.loadtxt("Log.txt")
        run_time = time.time() - start
        np.savetxt("Log.txt",
                    np.vstack([log_table,np.array([sampler.results.niter, sampler.results.ncall.sum(), delta_logz, run_time])]),
                    header="Maxiter \t Maxcall \t dlogz \t Time[s]", 
                    fmt="%d \t\t %d \t\t %f \t %f")
        
        print(f"\nniter: %d. ncall: %d. dlogz: %f." %(sampler.it, sampler.ncall,delta_logz))

    

    print(f"\nSaving Final Sample:")
    with open(f"final_jam_gaussianML_noDM.pickle", "wb") as f:
        pickle.dump(sampler, f, -1) 
    print(f"\nSaved!!")   
