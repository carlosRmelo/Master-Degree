"""
Attention!
This code runs in MPI mode.
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

#MPI Multiprocessing
from schwimmbad import MPIPool


#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u


#Gaussian ML function
def gaussian_ml(sigma, delta, ml0=1.0, lower=0.0):
    '''
    Create a M*L gradient
    sigma: Gaussian sigma [arcsec]
    delta: Gradient value
    ml0: Central stellar mass to light ratio
    lower: the ratio between the central and the outer most M*/L
    '''

    sigma = np.atleast_1d(sigma)
    sigma = sigma - sigma[0]
    ML = ml0 * (lower + (1-lower)*np.exp(-0.5 * (sigma * delta)**2))
    
    return ML
#------------------------------------------------------------------------------------#

#Reading data
y_px, x_px, vrms, erms = np.loadtxt('pPXF_rot_data.txt', unpack=True)                  #pPXF
surf_star_dat, sigma_star_dat, qstar_dat = np.loadtxt('JAM_Input.txt', unpack=True)    #photometry
surf_DM_dat, sigma_DM_dat, qDM_dat  = np.loadtxt('pseudo-DM Input.txt', unpack=True)   #DM


muse_normpsf, muse_sigmapsf = np.loadtxt("MUSE_Psf_model.txt", unpack=True)             #Muse PSF

### Global Constantes

#readshift

z_galaxy = 0.299                                #galaxy redshifth


#Angular diameter distances
D_l = cosmo.angular_diameter_distance(z_galaxy)                       


#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s


### Global Parameters
"""
    To inicialize the model, we set some random values for the parameters. But it's only necessary for initialize the model. During the non-linear search, this values will be updated constantly until the best fit.
"""   

#Galaxy
distance = D_l    #Angular diameter distance [Mpc]
inc = 85                                                    #Inclination [deg]
mbh =  1e5*u.solMass                                        #Mass of black hole [M_sun]
beta0 = np.ones_like(surf_star_dat)                         #Anisotropy parameter, one for each gaussian component
ML0 = gaussian_ml(sigma=sigma_star_dat, delta=1,
                     ml0=8, lower=0.4)*(u.solMass/u.solLum) #Gaussian Mass-to-light ratio [M_sun/L_sun]



#DM
surf_DM_dat = surf_DM_dat*(u.solMass/u.pc**2)                          #Surface Density in M_sun/pc²
sigma_DM_dat_ARC = sigma_DM_dat*u.arcsec                               #Sigma in arcsec
sigma_DM_dat_PC = (sigma_DM_dat_ARC*D_l).to(u.pc, u.dimensionless_angles())    #Convert sigma in arcsec to sigma in pc
qDM_dat = qDM_dat                                                              #axial ratio of DM halo


#Stars
surf_star_dat = surf_star_dat*(u.solLum/u.pc**2)               #Surface luminosity Density in L_sun/pc²
sigma_star_dat_ARC = sigma_star_dat*u.arcsec                   #Sigma in arcsec
sigma_star_dat_PC = (sigma_star_dat_ARC*D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc
qstar_dat = qstar_dat                                          #axial ratio of star photometry


#----------------------------------------------------------------------------------------------------#


# JAMPY MODEL

#Defining some instrumental quantities and galaxy characteristics

muse_pixsize=0.2                                            #pixscale of IFU [arcsec/px]
muse_normpsf=muse_normpsf                                   #normalized intensity of IFU PSF
muse_sigmapsf=muse_sigmapsf                                 #sigma of each gaussian IFU PSF [arcsec]

#Create model
Jampy_model = JAM(ybin=y_px, xbin=x_px,inc=inc, distance=distance.value, mbh=mbh.value,
                  rms=vrms, erms=erms, beta=beta0, normpsf=muse_normpsf, sigmapsf=muse_sigmapsf, pixsize=muse_pixsize)

#Add Luminosity component
Jampy_model.luminosity_component(surf_lum=surf_star_dat.value, sigma_lum=sigma_star_dat_ARC.value,
                                    qobs_lum=qstar_dat, ml=ML0.value)

#Add Dark Matter component
Jampy_model.DM_component(surf_dm=surf_DM_dat.value, sigma_dm=sigma_DM_dat_ARC.value, qobs_dm=qDM_dat)



#--------------------------------------- EMCEE -----------------------------------------------------#


### Priors

###boundaries. [lower, upper]
boundary = {'inc': [60, 120], 'beta': [-5, 5], 'ml0': [0.5, 15], 'delta': [0.5, 2], 'lower': [0, 1],
            'log_mbh':[7, 11], 'qDM': [0.15, 1], 'log_rho_s':[6, 13]}







"""
    For all  parameters we assume flat priors in log space, i.e, if the value is accepted its return 0 (log(1)), otherwise return -np.inf (log(0)). This choice is because we are assuming no previous knowledge about any parameters.
"""



def check_Deprojected_axial(parsDic):
    inc = np.radians(parsDic['inc'])
    #Stellar
    qintr_star = qstar_dat**2 - np.cos(inc)**2
    if np.any(qintr_star <= 0):
        return -np.inf
    
    qintr_star = np.sqrt(qintr_star)/np.sin(inc)
    if np.any(qintr_star <= 0.05):
        return -np.inf
    
        #DM
    qintr_DM = parsDic['qDM']**2 - np.cos(inc)**2
    if qintr_DM <= 0:
        return -np.inf
    
    qintr_DM = np.sqrt(qintr_DM)/np.sin(inc)
    if qintr_DM <= 0.05:
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
    if any(parsDic['beta'] == 1):
        return -np.inf
    else:
        pass

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
    The lines above is only for doble check, because we are assuming flat prior for all parameters. Once we got here, all values ​​have already been accepted, so just return 0.0 for each of them. 
    """

    rst += 0.0     #ml0
    rst += 0.0     #beta
    rst += 0.0     #inc
    rst += 0.0     #log_mbh
    rst += 0.0     #qDM
    rst += 0.0     #log_rho_s
    rst += 0.0     #delta
    rst += 0.0     #lower
    
    return rst




def Updt_JAM(parsDic):
    '''
       Update the dynamical mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''
    surf_DM_model = surf_DM_dat*(10**parsDic['log_rho_s'])       #DM mass surface density
    qDM_model = np.ones(qDM_dat.shape)*parsDic['qDM']            #DM axial ratio
    beta_model = np.array(parsDic['beta'])                       #anisotropy parameter
    mbh_model = 10**parsDic['log_mbh']                           #BH mass
    
    #mass-to-light update 
    ml_model = gaussian_ml(sigma=sigma_star_dat, delta=parsDic['delta'],
                           ml0=parsDic['ml0'], lower=parsDic['lower'])
    


    #Model Updt
    Jampy_model.upt(surf_dm=surf_DM_model, qobs_dm=qDM_model, inc=parsDic['inc'],
                     ml=ml_model, beta=parsDic['beta'], mbh=mbh_model)


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

    (ml0, delta, lower, b1, b2, b3, b4, b5, b6, b7, b8, inc, log_mbh,
             qDM, log_rho_s) = pars
    


    
    beta =  np.array([b1, b2, b3, b4, b5, b6, b7, b8])
    parsDic = {'ml0': ml0, 'delta': delta, 'lower':lower,'beta': beta, 'inc': inc,
                'log_mbh': log_mbh, 'qDM': qDM, 'log_rho_s': log_rho_s}
    
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

ml_init = np.array([10, 0.7, 0.4])                #Parameters of gaussian ML [ml0, delta, lower]
beta_init = beta0*(3 - 4*np.random.random())    #Anisotropy parameters 
inc_init = np.array([85])                         #Inclination in deg
log_mbh_init = np.array([8])                      #Log mass of SMBH
qDM_init = np.array([0.5])                        #Scalar describing the axial ratio of DM component
log_rho_s_init = np.array([8])                    #Log intensity of pseudo-NFW profile



##Here we append all the variables in asingle array.
p0 = np.append(ml_init, beta_init)
p0 = np.append(p0,[inc_init, log_mbh_init, qDM_init, log_rho_s_init])

"""
  We will start all walkers in a large gaussian ball around the values above. 
  There is no reason for that, once we do not have any prior information  
"""
p0_std = np.abs(p0*0.5)                                  #0.5 is the sigma of the Gaussian ball. 
                                                                #We are using 50% of the initial guesses


#Finally we initialize the walkers with a gaussian ball around the best Collet's fit.
nwalkers = 200                                                  #Number of walkers
pos = emcee.utils.sample_ball(p0, p0_std, nwalkers)             #Initial position of all walkers


nwalkers, ndim = pos.shape




"""
    We save the results in two tables. 
    The first marks the number of iterations, the mean acceptance fraction an the running time. 
    The second marks the last fit values for each parameter.
"""
np.savetxt('Output_LogFile.txt', np.column_stack([0, 0, 0]),
                            fmt=b'	%i	 %e			 %e	 ', 
                            header="Output table for the Dynamic model.\n Iteration	 Mean acceptance fraction	 Processing Time")


with MPIPool() as pool:

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    #Print the number os cores/workers
    print("Workers nesse job:", pool.workers)
    print("Start")


    #Backup
    filename = "SDP81_Jampy.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    #Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)
    
    nsteps = 50000

     # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)

     # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    start = time.time()
    global_time = time.time()
    for sample in sampler.sample(pos, iterations=nsteps, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        print("\n")
        print("##########################")
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
        upt = np.column_stack([iteration, accept, total_time])

        np.savetxt('Output_LogFile.txt', np.vstack([table, upt]),
                                fmt=b'	%i	 %e			 %e	 ', 
                                header="Iteration	 Mean acceptance fraction	 Processing Time")


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


