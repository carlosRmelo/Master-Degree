"""
Attention!
This code makes use of MPI
"""


"""
The proposal of this test is implement a walk as was proposed by Nelson et al (2014).
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
    implemented following `Nelson et al. (2013)
    <https://arxiv.org/abs/1311.5229>`_.
Roughly speaking, we want update the walkers parameters for each 100 steps, with the objective of obtaining a better acceptance fraction.


In his words:
"The value of γ can be updated after every generation through-out a RUN DMC simulation. We aim for an acceptance fractionof 0.25. If too few states are being accepted (<0.2), γ is scaled by 0.9 in the hope that smaller jumps will lead to a higher accep-tance fraction. If the acceptance fraction exceeds 0.31, then γ is scaled by 1.1 to allow for larger jumps. For intermediate acceptance fractions, γ is scaled by sqrt(Acceptance Fraction/0.25). In DEMCMC, this procedure references information from only one previous generation of states, so our algorithm is still Markov foreach generation. The mathematical conditions for RUN DMC converging to the target distribution are still satisfied (ter Braak2006), and thus, adjustments in the proposal vector size will not change the shape or scale of the target distribution."
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

from os import path

#Reading data
y_px, x_px, vrms, erms = np.loadtxt('pPXF_rot_data.txt', unpack=True)                  #pPXF
surf_star_dat, sigma_star_dat, qstar_dat = np.loadtxt('JAM_Input.txt', unpack=True)    #photometry
surf_DM_dat, sigma_DM_dat, qDM_dat  = np.loadtxt('pseudo-DM Input.txt', unpack=True)   #DM



### Global Constantes

#Redshifth

z_galaxy = 0.035                                 #galaxy redshifth

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
distance = D_l                                              #Angular diameter distance [Mpc]
inc = 85                                                    #Inclination [deg]
mbh =  1e5*u.solMass                                        #Mass of black hole [M_sun]
beta0 = beta = np.zeros(surf_star_dat.shape)                #Anisotropy parameter, one for each gaussian component
ML0 = np.ones(surf_star_dat.shape)*u.solMass/u.solLum       #Mass-to-light ratio per gaussian [M_sun/L_sun]


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

muse_pixsize=0.6                                            #pixscale of IFU [arcsec/px]
muse_sigmapsf= 0.2420                                       ##Sigma of psf from MUSE [arcsec]

#Create model
Jampy_model = JAM(ybin=y_px, xbin=x_px,inc=inc, distance=distance.value, mbh=mbh.value,
                  rms=vrms, erms=erms, beta=beta0, sigmapsf=muse_sigmapsf, pixsize=muse_pixsize)

#Add Luminosity component
Jampy_model.luminosity_component(surf_lum=surf_star_dat.value, sigma_lum=sigma_star_dat_ARC.value,
                                    qobs_lum=qstar_dat, ml=ML0.value)

#Add Dark Matter component
Jampy_model.DM_component(surf_dm=surf_DM_dat.value, sigma_dm=sigma_DM_dat_ARC.value, qobs_dm=qDM_dat)



#----------------------------------------------------------------------------------------------------#

#--------------------------------------- EMCEE ------------------------------------------------------#


### Priors

###boundaries. [lower, upper]
boundary = {'inc': [60, 120], 'beta': [-5, 5], 'ml': [0.5, 15],
                 'log_mbh':[7, 11], 'qDM': [0.15, 1], 'log_rho_s':[6, 13]}


""" 
    For all  parameters we assume flat priors in log space, i.e, if the value is accepted its return 0 (log(1)), otherwise return -np.inf (log(0)). This choice is because we are assuming no previous knowledge about any parameters.
"""



def check_ML_Grad(ml):
    """
        Check if the mass-to-light ratio is descending
    """

    for i in range(len(ml) -1):
        if ml[i] >= ml[i+1]:
            pass
        else:
            return -np.inf

    return 0.0


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

    #Check if ml is ok

    #Check the gradient
    if not np.isfinite(check_ML_Grad(parsDic['ml'])):
        return -np.inf

    #Check ml boundary
    for i in range(len(parsDic['ml'])):
        if boundary['ml'][0] < parsDic['ml'][i] < boundary['ml'][1] :
            pass
        else:
            return -np.inf


    #Check if deprojected axial ratio is ok  (q' <=0 or q' <= 0.05) for the dynamical model.
    if not np.isfinite(check_Deprojected_axial(parsDic)):
        return -np.inf

    #Check if the others parameters are within the boundary limits
    keys = set(parsDic.keys())
    excludes = set(['ml', 'beta'])  #Exclude beta and ml, because we already verify above


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
    The lines above is only for doble check, because we are assuming flat prior for all parameters, except for the gamma. Once we got here, all values ​​have already been accepted, so just return 0.0 for one of them
    """

    rst += 0.0     #mass-to-light
    rst += 0.0     #beta
    rst += 0.0     #inc
    rst += 0.0     #log_mbh
    rst += 0.0     #qDM
    rst += 0.0     #log_rho_s
    
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
    ml_model = np.array([parsDic['ml'][0], parsDic['ml'][0], parsDic['ml'][1], 
                            parsDic['ml'][2], parsDic['ml'][3], parsDic['ml'][4], parsDic['ml'][5]])
    
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
    (m1, m2, m3, m4, m5, m6, b1, b2, b3, b4, b5, b6, b7,
         inc, qDM, log_rho_s, log_mbh) = pars
    

    ml =  np.array([m1, m2, m3, m4, m5, m6])  #we have 7-gaussians, but we assume ml1=ml2.
                                                #in the codes above we updt the parameters in that sense
    beta =  np.array([b1, b2, b3, b4, b5, b6, b7])
    
    parsDic = {'ml': ml, 'inc': inc, 'qDM': qDM, 'log_rho_s': log_rho_s,
                     'log_mbh': log_mbh, 'beta': beta}
    
    
    #Checking boundaries
    if not np.isfinite(check_boundary(parsDic)):
        return -np.inf
    #calculating the log_priors
    lp = log_prior(parsDic)

    return lp + JAM_log_likelihood(parsDic)


#Initial Positions of walkers


"""
    For the initial guesses we will use the Collett's best fit with a gaussian ball error around it variables tagged with <name>_std are the standard deviation of the parameter <name>.

    Pay close attention to the order in which the components are added. 
    They must follow the log_probability unpacking order.
"""


np.random.seed(42)   
 #Defining initial guesses
'''
ml = np.array([9.5,8.5,3.8,3.4,3.2,2.8])
ml_std = ml*np.array(0.15)

beta = np.array([-0.6, -1.0, 0.34, -3.4, 0.39, -0.31, 0.36])
beta_std = np.abs(beta*np.array(0.15))

inc = np.array([90])
inc_std = inc*np.array(0.15)

qDM = np.array([0.74])
qDM_std = qDM*np.array(0.15)

log_rho_s = np.array([6])
log_rho_s_std = np.ones(log_rho_s.shape)*4

log_mbh = np.array([10])
log_mbh_std = np.ones(log_mbh.shape)*3

'''
ml = np.array([9.5,8.5,3.8,3.4,3.2,2.8])
beta = np.array([-0.6, -1.0, 0.34, -3.4, 0.39, -0.31, 0.36])
inc = np.array([90])
qDM = np.array([0.74])
log_rho_s = np.array([6])
log_mbh = np.array([10])

##Here we append all the variables and stds in a single array.
p0 = np.append(ml, beta)
p0 = np.append(p0,[inc, qDM, log_rho_s, log_mbh])

#p0_std = np.append(ml_std, beta_std)
#p0_std = np.append(p0_std, [inc_std, qDM_std, log_rho_s_std, log_mbh_std])

#Finally we initialize the walkers with a gaussian ball around the best Collet's fit.
nwalkers = 400                                                  #Number of walkers

pos = p0 + np.random.randn(nwalkers, p0.size)

#pos = emcee.utils.sample_ball(p0, p0_std, nwalkers)             #Initial position of all walkers

nwalkers, ndim = pos.shape                                      #Number of walkers/dimensions
#print(pos.shape)
#print("\n")
#print(pos)


"""
    We save the results in two tables. 
    The first marks the number of iterations, the mean acceptance fraction an the running time. 
    The second marks the last fit values for each parameter.
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
    filename = "Emcee-JAM-ESO325.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    #Defining moves
    moves =  [ (emcee.moves.DEMove())]
    g0 = 2.38 / np.sqrt(2 * ndim)             #gamma0 recommended by emcee
    #Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,
                                     backend=backend, moves=moves)
    
    #Burn in fase
    burnin = 1000                           #Number os burn in steps
    print("Burn in with %i steps"%burnin)
    state = sampler.run_mcmc(pos, nsteps=burnin, progress=True)
    sampler.reset()
    print("\n")
    print("End of bur in fase")
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

        #Check accp. fraction of last 100 steps and updt the moves
        print(sampler._moves[0].gamma0)
        if mean_accp_100 < 0.2:
            sampler._moves[0].gamma0 = 0.9 * g0
            
        
        elif mean_accp_100 > 0.31:
            sampler._moves[0].gamma0 = 1.1 * g0
            
        else:
            sampler._moves[0].gamma0 = np.sqrt(mean_accp_100/0.25) * g0
                       
        
        print("\n")
        print(mean_accp_100, sampler._moves[0].gamma0)
        





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
