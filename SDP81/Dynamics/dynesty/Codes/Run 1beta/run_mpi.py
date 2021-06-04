#Control time packages
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np
import matplotlib.pyplot as plt
from plotbin.plot_velfield import plot_velfield
from My_Jampy import JAM                          #My class for jampy


from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

from schwimmbad import MPIPool

boundary = {"ml": [0.5, 15], "qinc": [0.051, 0.468808], "beta": [-3, 3],
                "log_mbh": [7, 11], "kappa_s": [0, 1], "qDM":[0.1, 1]}

class Model(object):
    def __init__(self, Jam_model):
        self.Jam_model = Jam_model

    def prior_transform(self, theta):
        ml, qinc, beta, log_mbh, kappa_s, qDM = theta
        parsDic = {"ml":ml, "qinc":qinc, "beta":beta, "log_mbh":log_mbh, "kappa_s":kappa_s, "qDM":qDM }
        for key in parsDic:
            parsDic[key] = boundary[key][0] + parsDic[key]*(boundary[key][1] - boundary[key][0])
        
        return np.array(list(parsDic.values()))

    def log_likelihood(self, theta):
        ml, qinc, beta, log_mbh, kappa_s, qDM = theta    
        # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
        qmin = np.min(self.Jam_model.qobs_lum)
        inc_model = np.degrees(np.arctan(np.sqrt((1 - qmin**2)/(qmin**2 -qinc**2))))
        
        surf_dm_model = (kappa_s) * self.Jam_model.surf_dm
        qdm_proj      = np.sqrt( (np.sin(np.radians(inc_model)) * qDM )**2  + np.cos( np.radians(inc_model))**2     )     #Projected DM axial ratio
        qDM_model     = np.ones_like(self.Jam_model.qobs_dm)*qdm_proj

        beta_model  = np.ones_like(self.Jam_model.surf_lum)*beta
        ml_model    = ml
        mbh_model   = 10**log_mbh
        
        self.Jam_model.upt(surf_dm=surf_dm_model, qobs_dm=qDM_model, inc=inc_model,
                            ml=ml_model, beta=beta_model, mbh=mbh_model)
        
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
    x, y, vrms, erms              = np.loadtxt("Input/Vrms_map_rot_BadAss_boots.txt", unpack=True)          #vrms data
    surf_lum, sigma_lum, qobs_lum = np.loadtxt("Input/JAM_Input.txt", unpack=True)              #MGE decomposition
    surf_dm, sigma_dm , qobs_dm   = np.loadtxt("Input/eNFW.txt", unpack=True)                   #DM component
    norm_psf, sigma_psf           = np.loadtxt("Input/MUSE_Psf_model.txt", unpack=True)         #PSF

    # Defing some inputs
    z_l     = 0.299                                                         #Redshift
    D_l     = cosmo.angular_diameter_distance(z_l).value                    #Distance to Lens [Mpc] 
    mbh     = 1e9                                                           #mass of black hole [log10(M_sun)]
    beta    = np.full_like(surf_lum, -0.15)                                 #anisotropy [ad]
    inc     = 65                                                            #inclination [deg]
    kappa_s = 0.075                                                         #kappa_s of DM profile
    ml      = 7.00                                                          #mass to light ratio
    inc_rad = np.radians(inc)
    qinc    = np.sqrt(np.min(qobs_lum)**2 - 
                            (1 - np.min(qobs_lum)**2)/np.tan(inc_rad)**2)   #Deprojected axial ratio for inclination
    qDM     = np.sqrt( qobs_dm[0]**2 - np.cos(inc_rad)**2)/np.sin(inc_rad)  #Deprojected DM axial ratio

    pixsize = 0.2    #MUSE pixel size

    # Now we start our Jampy class
    Jam_model = JAM(ybin=y*pixsize, xbin=x*pixsize, inc=inc, distance=D_l, mbh=mbh, beta=beta, rms=vrms, erms=erms,
                    normpsf=norm_psf, sigmapsf=sigma_psf*pixsize, pixsize=pixsize)

    #Add Luminosity component
    Jam_model.luminosity_component(surf_lum=surf_lum, sigma_lum=sigma_lum,
                                        qobs_lum=qobs_lum, ml=ml)
    #Add DM component
    Jam_model.DM_component(surf_dm=kappa_s * surf_dm, sigma_dm=sigma_dm, qobs_dm=qobs_dm)

    model = Model(Jam_model=Jam_model)

    print("Testing Inputs...")
    p0 = np.array([ml, qinc, beta[0], np.log10(mbh), kappa_s, qDM])
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
    nlive = 40              # number of (initial) live points
    ndim  = p0.size         # number of dimensions
    sampling = "rwalk"      # sampling method



    # Now run with the static sampler
    sampler = NestedSampler(model, model.prior_transform, ndim,
                                pool=pool,
                                nlive=nlive,
                            )
    # These hacks are necessary to be able to pickle the sampler.
    sampler.rstate = np.random
    sampler.pool   = pool
    sampler.M      = pool.map


    delta_logz = 1e200
    while delta_logz > 0.1:
        maxcall = 500
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
        with open(f"dynesty_jam.pickle", "wb") as f:
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
    with open(f"final_dynesty_jam.pickle", "wb") as f:
        pickle.dump(sampler, f, -1) 
    print(f"\nSaved!!")   
