"""
Attention!
This code makes use of MPI
"""
""""
The goal is reproduce Cappellari's jam_axi_rms_example.py making use of Emcee.
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

#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s


#Dataset path
data_folder = "/home/carlos/Documents/GitHub/Master-Degree/SDP81/Dynamics/Emcee/Data/"

#Reading MGE inputs
    #attention to units
surf_lum, sigma_lum, qobs_lum = np.loadtxt(data_folder+"JAM_Input.txt", unpack=True)          #MGE decomposition
surf_dm, sigma_dm , qobs_dm   = np.loadtxt(data_folder+"NFW.txt", unpack=True)             #DM component
norm_psf, sigma_psf           = np.loadtxt(data_folder+"MUSE_Psf_model.txt", unpack=True)     #PSF
ybin, xbin, vrms, erms        = np.loadtxt(data_folder+"pPXF_rot_data.txt", unpack=True)          #Vrms data

muse_pixsize = 0.2                            #Muse pixel size [arcsec/px]

z_lens   = 0.299                                    #Lens redshifth
z_source = 3.042                                    #Source redshift

#Angular diameter distances
D_l = cosmo.angular_diameter_distance(z_lens)                   #Lens              
D_s = cosmo.angular_diameter_distance(z_source)                 #Source
D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)   #Lens to source


#This quantities are our unknown parameters
inc       = 75                              #Inclination [deg]
mbh       = 1e10                            #Mass of black hole [M_sun]
beta      = np.full_like(surf_lum, 0.3)     #Anisotropy
ml        = 10                              #Mass to light ratio [M_sun/L_sun]
rho_s     = 1e10                            #dark matter intensity
qdm       = np.full_like(qobs_dm, 0.5)      #dark matter axial ratio
    

#--------------------------------------------------------------------------------------------------#
# JAMPY MODEL

Jam_model = JAM(ybin=ybin*muse_pixsize, xbin=xbin*muse_pixsize, inc=inc, distance=D_l.value, mbh=mbh, beta=beta, rms=vrms,
                erms=erms,normpsf=norm_psf, sigmapsf=sigma_psf*muse_pixsize, pixsize=muse_pixsize)

    #Add Luminosity component
Jam_model.luminosity_component(surf_lum=surf_lum, sigma_lum=sigma_lum,
                                qobs_lum=qobs_lum, ml=ml)
    #Add DM component
Jam_model.DM_component(surf_dm=rho_s * surf_dm, sigma_dm=sigma_dm, qobs_dm=qdm)


#--------------------------------------- EMCEE ------------------------------------------------------#

### Priors

###boundaries. [lower, upper]
boundary = {'qinc': [0.0501, np.min(qobs_lum)], 'beta': [-5, 5], 'ml': [0.5, 15], 'log_mbh':[5, 11],
                'kappa_s': [0, 1.0], 'qDM': [0.4, 1]}





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




    #Check if the others parameters are within the boundary limits
    keys = set(parsDic.keys())
    excludes = set(['beta'])  #Exclude beta and ml, because we already verify above

    
    for keys in keys.difference(excludes):
        if boundary[keys][0] <= parsDic[keys] <= boundary[keys][1]:
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
    rst += 0.0     #qinc
    rst += 0.0     #log_mbh
    rst += 0.0     #kappa_s
    rst += 0.0     #qDM

    
    return rst


def Updt_JAM(parsDic):
    '''
       Update the dynamical mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''

        # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc_model = np.degrees(np.arctan(np.sqrt((1 - qmin**2)/(qmin**2 - parsDic['qinc']**2))))
   

    ml_model  = parsDic['ml']                        #mass-to-light update 
    mbh_model = 10**parsDic['log_mbh']               #BH mass
    kappa_model = parsDic['kappa_s']               #rho DM
    qdm_proj  = np.sqrt( (np.sin(np.radians(inc_model)) * parsDic['qDM'] )**2  + np.cos( np.radians(inc_model))**2     )                            #Projected DM axial ratio
    qdm_model = np.full_like(surf_dm, qdm_proj)      #qDM

    #We project the DM axial ratio because during the JAM fit it will be deprojected.

    #Model Updt
    Jam_model.upt(inc=inc_model, ml=ml_model, beta=parsDic['beta'], mbh=mbh_model,
                        surf_dm= kappa_model * surf_dm, qobs_dm=qdm_model)


def JAM_log_likelihood(parsDic):
    """
        Perform JAM modeling and return the chi2
    """
    Updt_JAM(parsDic)               #Updt values for each iteration
    
    rmsModel, ml, chi2, chi2T = Jam_model.run()
    plt.show()
    return -0.5 * chi2T


def log_probability(pars):
    """
        Log-probability function for whole model.
        input:
            pars: current values in the Emcee sample.
        output:
            log probability for the combined model.
    """

    (ml, b1, b2, b3, b4, b5, b6, b7, b8, qinc, log_mbh, kappa_s, qdm) = pars
    
    beta =  np.array([b1, b2, b3, b4, b5, b6, b7, b8])


    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                        'kappa_s': kappa_s, 'qDM': qdm}

    
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

nwalkers = 120                                                   #Number of walkers
pos = np.random.uniform(low=[1, -3,-3,-3,-3,-3,-3,-3,-3, 0.1, 6, 0, 0.5], high=[10, 3, 3, 3, 3, 3, 3, 3, 3, np.min(qobs_lum), 10, 1, 1], size=[nwalkers, 13])
nwalkers, ndim = pos.shape                                      #Number of walkers/dimensions

log_probability(pos[0])
"""
    We save the results in a table.
    This tabke marks the number of iterations, the mean acceptance fraction,the running time, and the mean accep. fraction of last 100 its. 
    
"""
np.savetxt('Output_LogFile.txt', np.column_stack([0, 0, 0, 0]),
                            fmt=b'	%i	 %e			 %e     %e', 
                            header="Output table for the combined model: Dynamic.\n Iteration	 Mean acceptance fraction	 Processing Time    Last 100 Mean Accp.")

moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DEMove(gamma0=1.0), 0.20)]
#here we use the MPI
with MPIPool() as pool:

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    #Print the number os cores/workers
    print("Workers nesse job:", pool.workers)
    print("Start")

    #Backup
    filename = "SDP_jam.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)


    #Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,
                                     backend=backend, moves=moves)
    
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

    for sample in sampler.sample(pos, iterations=nsteps, progress=True):
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
            break
        old_tau = tau



    end = time.time()
    print('\n')
    print("Final")
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

