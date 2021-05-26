import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from plotbin.plot_velfield import plot_velfield
from My_Jampy import JAM                          #My class for jampy


from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

from multiprocessing import Pool,cpu_count


# Reading Data
y, x, vrms, erms              = np.loadtxt("Input/pPXF_rot_data.txt", unpack=True)          #vrms data
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


boundary = {"ml": [0.5, 15], "qinc": [0.051, 0.55], "beta": [-3, 3],
                "log_mbh": [7, 11], "kappa_s": [0, 1], "qDM":[0.1, 1]}

def prior_transform(theta):
    ml, qinc, beta, log_mbh, kappa_s, qDM = theta
    parsDic = {"ml":ml, "qinc":qinc, "beta":beta, "log_mbh":log_mbh, "kappa_s":kappa_s, "qDM":qDM }
    for key in parsDic:
        parsDic[key] = boundary[key][0] + parsDic[key]*(boundary[key][1] - boundary[key][0])
    
    return np.array(list(parsDic.values()))

def log_likelihood(theta):
    ml, qinc, beta, log_mbh, kappa_s, qDM = theta    
    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc_model = np.degrees(np.arctan(np.sqrt((1 - qmin**2)/(qmin**2 -qinc**2))))
    
    surf_dm_model = (kappa_s) * surf_dm
    qdm_proj      = np.sqrt( (np.sin(np.radians(inc_model)) * qDM )**2  + np.cos( np.radians(inc_model))**2     )     #Projected DM axial ratio
    qDM_model     = np.ones_like(qobs_dm)*qdm_proj

    beta_model  = np.ones_like(surf_lum)*beta
    ml_model    = ml
    mbh_model   = 10**log_mbh
    
    Jam_model.upt(surf_dm=surf_dm_model, qobs_dm=qDM_model, inc=inc_model,
                        ml=ml_model, beta=beta_model, mbh=mbh_model)
    
    rmsModel, ml, chi2, chi2T = Jam_model.run(quiet=True, plot=False)
    
    
    return -0.5 * chi2T

print("Testing Inputs...")
p0 = np.array([ml, qinc, beta[0], np.log10(mbh), kappa_s, qDM])
log_likelihood(p0)

from dynesty import NestedSampler
import _pickle  as pickle
import time

def run_dynesty():
    niter=1
    ncall=0
    with Pool(cpu_count()-1) as executor:
        print("Number of CPUS:", cpu_count())
        nlive = 20             # number of (initial) live points
        ndim  = p0.size         # number of dimensions


        # Now run with the static sampler
        sampler = NestedSampler(log_likelihood, prior_transform, ndim, 
                                    pool=executor, queue_size=cpu_count(),
                                    nlive=nlive
                                )
        print_func      = None
        print_progress  = True
        pbar,print_func = sampler._get_print_func(print_func,print_progress)                        
        #fit_fun         = sampler.loglikelihood
        
        start = time.time()
        print("\n")
        print("Job started!")
        for it, res in enumerate(sampler.sample(dlogz=0.01)):
            
            (worst, ustar, vstar, loglstar, logvol,
                    logwt, logz, logzvar, h, nc, worst_it,
                    boundidx, bounditer, eff, delta_logz) = res

            niter+=1
            ncall+=nc

            print_func(res,niter,ncall,nbatch=0,dlogz=0.01,logl_max=np.inf)

            
            if it%50:
                continue
            print("\n")
            print("Saving...")
            with open(f"jam_run.pickle", "wb") as f:
                pickle.dump(sampler, f, -1)
                f.close()
            print("File Save!\n")
            print("Cumulative Time [s]:", (time.time() - start))
            print("\n ########################################## \n")
        
    print("Elapse Time:", (time.time() - start))  
    print("Saving Final Sample:")
    # Adding the final set of live points.
    for it_final, res in enumerate(sampler.add_live_points()):
        pass

    with open(f"final_jam_run.pickle", "wb") as f:
        pickle.dump(sampler, f) 
    print("Saved!!")
    return print("Final")  

if __name__ == '__main__':

    run_dynesty() 


