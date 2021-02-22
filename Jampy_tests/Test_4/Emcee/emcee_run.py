"""
Attention!
This code makes use of MPI
"""
""""
The goal is reproduce the simulation maded in the folder _Simulation_
"""

#Control time packages
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"


#General packages
import numpy as np
from My_Jampy import JAM
import emcee
import matplotlib.pyplot as plt

#MPI
from schwimmbad import MPIPool

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u


#Reading MGE inputs
surf_lum, sigma_lum, qobs_lum = np.loadtxt("JAM_Input.txt", unpack=True)               #MGE decomposition
norm_psf, sigma_psf           = np.loadtxt("MUSE_Psf_model.txt", unpack=True)          #PSF
xbin, ybin, vrms, goodbins    = np.loadtxt("vrms_data.txt", unpack=True)               #Vrms data

goodbins = goodbins == 1

z_l    = 0.299                                            #Redshift
D_l    = cosmo.angular_diameter_distance(z_l).value       #Distance to Lens [Mpc] 
pixsize = 0.2                                             #pixel size [arcsec/px]

#JAMPY MODEL

#Just for inicialization
mbh  = 1         
inc  = 1
beta = np.full_like(surf_lum, 0)

#Create model
Jampy_model = JAM(ybin=ybin, xbin=xbin,inc=inc, distance=D_l, mbh=mbh, goodbins=goodbins,
                  rms=vrms, beta=beta, sigmapsf=sigma_psf, normpsf=norm_psf, pixsize=pixsize)

#Add Luminosity component
Jampy_model.luminosity_component(surf_lum=surf_lum, sigma_lum=sigma_lum, qobs_lum=qobs_lum)


#--------------------------------------- EMCEE ------------------------------------------------------#
### Priors

###boundaries. [lower, upper]
boundary = {'inc': [40, 90], 'beta': [-5, 5], 'ml': [0.5, 15], 'log_mbh':[5, 11]}



def check_Deprojected_axial(parsDic):
    inc = np.radians(parsDic['inc'])
    #Stellar
    qintr_star = qobs_lum**2 - np.cos(inc)**2
    if np.any(qintr_star <= 0):
        return -np.inf
    
    qintr_star = np.sqrt(qintr_star)/np.sin(inc)
    if np.any(qintr_star <= 0.05):
        return -np.inf
    
    return 0.0

def check_boundary(parsDic):
    """
        Check whether parameters are within the boundary limits
        input
            parsDic: parameter dictionary {'paraName', value}
        output
            -np.inf or 0.0
    """   

    #Check if beta is ok

    #Avoid beta[i] == 1, because this could cause problems
    for i in range(len(parsDic['beta'])):
        if parsDic['beta'][i] != 1:
            pass
        else:
            return -np.inf

    #Check beta boundary
    for i in range(len(parsDic['beta'])):
        if boundary['beta'][0] < parsDic['beta'][i] < boundary['beta'][1] :
            pass
        else:
            return -np.inf


    #Check if deprojected axial ratio is ok  (q' <=0 or q' <= 0.05) for the dynamical model.
    if not np.isfinite(check_Deprojected_axial(parsDic)):
        return -np.inf

    #Check if the others parameters are within the boundary limits
    keys = set(parsDic.keys())
    excludes = set(['beta'])  #Exclude beta and ml, because we already verify above


    for keys in keys.difference(excludes):
        if boundary[keys][0] < parsDic[keys] < boundary[keys][1]:
            pass
        else:
            return -np.inf
    return 0.0

def log_prior(parsDic):
    '''
    Calculate the prior lnprob
    input
      parsDic: parameter dictionary {'paraName', value}
    output
      lnprob
    '''
    
    rst = 0
    """
    The lines above is only for doble check, because we are assuming flat prior for all parameters, Once we got here, all values ​​have already been accepted, so just return 0.0 for one of them
    """

    rst += 0.0     #mass-to-light
    rst += 0.0     #beta
    rst += 0.0     #inc
    rst += 0.0     #log_mbh

    
    return rst


def Updt_JAM(parsDic):
    '''
       Update the dynamical mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''
    

    ml_model = parsDic['ml']                                  #mass-to-light update 
    mbh_model = 10**parsDic['log_mbh']              #BH mass

    
    #Model Updt
    Jampy_model.upt(inc=parsDic['inc'], ml=ml_model, beta=parsDic['beta'], mbh=mbh_model)


def JAM_log_likelihood(parsDic):
    """
        Perform JAM modeling and return the chi2
    """
    
    Updt_JAM(parsDic)               #Updt values for each iteration
    
    rmsModel, ml, chi2, chi2T = Jampy_model.run()
    return -0.5 * chi2T


def log_probability(pars):
    """
        Log-probability function for whole model.
        input:
            pars: current values in the Emcee sample.
        output:
            log probability for the combined model.
    """

    (ml, beta, inc, log_mbh) = pars
    
    beta =  np.full_like(surf_lum, beta)


    parsDic = {'ml': ml, 'beta': beta, 'inc': inc, 'log_mbh': log_mbh}

    
    #Checking boundaries
    if not np.isfinite(check_boundary(parsDic)):
        return -np.inf
    #calculating the log_priors
    lp = log_prior(parsDic)

    return lp + JAM_log_likelihood(parsDic)


#Initial Positions of walkers
"""
    Pay close attention to the order in which the components are added. 
    They must follow the log_probability unpacking order.
"""

p0 = np.array([5.0, 0.0, 80, 7.0])    #In order: ML, beta, inclination, log_mbh

#Finally we initialize the walkers arround these positions above.
nwalkers = 120                                                   #Number of walkers
pos = p0 +  np.random.randn(nwalkers, p0.size)                  #Initial guess of walkers
nwalkers, ndim = pos.shape                                      #Number of walkers/dimensions

"""
    We save the results in a table.
    This tabke marks the number of iterations, the mean acceptance fraction,the running time, and the mean accep. fraction of last 100 its. 
    
"""
np.savetxt('Output_LogFile.txt', np.column_stack([0, 0, 0, 0]),
                            fmt=b'	%i	 %e			 %e     %e', 
                            header="Output table for the combined model: Dynamic.\n Iteration	 Mean acceptance fraction	 Processing Time    Last 100 Mean Accp.")


#here we use the MPI
with MPIPool() as pool:

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    #Print the number os cores/workers
    print("Workers nesse job:", pool.workers)
    print("Start")

    #Backup
    filename = "simulation.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)


    #Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,
                                     backend=backend)
    
    #Burn in fase
    burnin = 1                           #Number os burn in steps
    print("Burn in with %i steps"%burnin)
    state = sampler.run_mcmc(pos, nsteps=burnin, progress=True)
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


        
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        #Update a table output with acceptance
        table = np.loadtxt("Output_LogFile.txt")

        iteration = sampler.iteration
        accept = np.mean(sampler.acceptance_fraction)
        total_time = time.time() - global_time
        upt = np.column_stack([iteration, accept, total_time, mean_accp_100])

        np.savetxt('Output_LogFile.txt', np.vstack([table, upt]),
                                fmt=b'	%i	 %e			 %e             %e', 
                            header="Output table for the combined model: Dynamic.\n Iteration	 Mean acceptance fraction	 Processing Time    Last 100 Mean Accp. Fraction")


        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            if 0.21 < accept < 0.35:
                break
        old_tau = tau



    end = time.time()
    print('\n')
    print("Final")
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

