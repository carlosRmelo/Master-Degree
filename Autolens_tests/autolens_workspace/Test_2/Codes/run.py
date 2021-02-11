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

#MPI
from schwimmbad import MPIPool

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.constants import G, M_sun, c
import astropy.units as u

#Autolens Model packages
import autolens as al
import autolens.plot as aplt

#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s



#Dataset path
dataset_path = "/home/carlos/Documents/GitHub/Master-Degree/Autolens_tests/autolens_workspace/Test_2/Data"

#Reading data of MGE and velocity dispersion maps
surf_lum, sigma_lum, qobs_lum = np.loadtxt("mge.txt", unpack=True)   #MGE data
xbin, ybin, goodbins, vrms  = np.loadtxt("vrms.txt", unpack=True)       #velocity dispersion map data

## Global informations and parameters
distance = 16.5 * u.Mpc                         #Lens galaxy distance [Mpc]

z = z_at_value(cosmo.angular_diameter_distance, distance, zmax=1.0) #Convert distance to redshifth 
z_lens = z                                    #Lens redshifth
z_source = 2.1                                #Source redshift

#Angular diameter distances
D_l = cosmo.angular_diameter_distance(z_lens)                   #Lens              
D_s = cosmo.angular_diameter_distance(z_source)                 #Source
D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)   #Lens to source

## Models inicialization

"""
    To inicialize the model, we set some random values for the parameters. But it's only necessary for initialize the model. During the non-linear search, this values will be updated constantly until the best fit.
"""  
#This quantities are our unknown parameters
inc = np.random.random()                    #Inclination [deg]
mbh = np.random.random()                    #Mass of black hole [M_sun]
beta = np.full_like(surf_lum, np.random.random()) #Anisotropy
ml = np.random.random()                     #Mass to light ratio [M_sun/L_sun]
mag_shear = np.random.random()              #Shear magnitude
phi_shear = np.random.random()              #Shear angle
gamma = np.random.random()                  #Gamma


#Now we define some quantities base on theses parameters

#Stars component for Jampy and Autolens
surf_star_dat = surf_lum                            #Surface luminosity Density in L_sun/pc²
sigma_star_dat_ARC = sigma_lum * u.arcsec           #Sigma in arcsec
sigma_star_dat_PC = (sigma_star_dat_ARC*D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc

     #After convertion, get only the values.   
sigma_star_dat_PC = sigma_star_dat_PC.value                 #Sigma of each gaussian [arcsec]
sigma_star_dat_ARC = sigma_star_dat_ARC.value               #Sigma of each gaussian [pc]
qobs_star_dat = qobs_lum                                    #Axial ratio of star photometry 

#Convert  surf_lum_sim to total mass per Guassian
Lum_star_dat = 2*np.pi*surf_star_dat*(sigma_star_dat_PC**2)*qobs_star_dat    #Total luminosity per gaussian component in L_sun

#Update the stellar mass based on M/L.
Mass_star = Lum_star_dat * ml                                  #Total star mass per gaussian in M_sun

#Inserting a Gaussian to represent SMBH at the center of the galaxy
sigmaBH_ARC = 0.01*u.arcsec
"""
        This scalar gives the sigma in arcsec of the Gaussian representing the
        central black hole of mass MBH (See Section 3.1.2 of `Cappellari 2008.
        <http://adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_)
        The gravitational potential is indistinguishable from a point source
        for ``radii > 2*RBH``, so the default ``RBH=0.01`` arcsec is appropriate
        in most current situations.

        ``RBH`` should not be decreased unless actually needed!
"""


sigmaBH_PC = (sigmaBH_ARC*D_l).to(u.pc, u.dimensionless_angles())        #Sigma of the SMBH in pc

    #After convertion, get only the values
sigmaBH_ARC = sigmaBH_ARC.value         #Sigma of gaussian BH [arcsec] 
sigmaBH_PC = sigmaBH_PC.value           #Sigma of gaussian BH [pc] 


surfBH_PC = mbh/(2*np.pi*sigmaBH_PC**2)                       #Mass surface density of SMBH [M_sun]
qSMBH = 1.                                                    #Assuming a circular gaussian
Mass_mbh = 2*np.pi*surfBH_PC*(sigmaBH_PC**2)*qSMBH            #SMBH Total mass 


Total_Mass = np.concatenate((Mass_star, Mass_mbh), axis=None)    #Mass per gaussian component in M_sun
Total_q = np.concatenate((qobs_star_dat, qSMBH), axis=None)      #Total axial ratio per gaussian


Total_sigma_ARC = np.concatenate((sigma_star_dat_ARC, sigmaBH_ARC), axis=None)  #Total sigma per gaussian in arcsec
Total_sigma_RAD = (Total_sigma_ARC * u.arcsec).to(u.rad)    #Total sigma per gaussian in radians
Total_sigma_RAD = Total_sigma_RAD.value                     #Only the value


#----------------------------------------------------------------------------------------------------#
# JAMPY MODEL

#Defining some instrumental quantities and galaxy characteristics

pixsize=0.8                                            #pixscale of IFU [arcsec/px]
normpsf=np.array([0.7, 0.3])                           #normalized intensity of IFU PSF
sigmapsf=np.array([0.6, 1.2])                          #sigma of each gaussian IFU PSF [arcsec]

#Create model
Jampy_model = JAM(ybin=ybin, xbin=xbin, inc=inc, distance=D_l.value, mbh=Mass_mbh,
                  rms=vrms, beta=beta, normpsf=normpsf, sigmapsf=sigmapsf, pixsize=pixsize)

#Add Luminosity component
Jampy_model.luminosity_component(surf_lum=surf_star_dat, sigma_lum=sigma_star_dat_ARC,
                                    qobs_lum=qobs_star_dat, ml=ml)

#----------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------#
# Pyautolens Model
#Reading fits file with the arcs data

#Imaging
pixel_scales = 0.1
imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        noise_map_path=f"{dataset_path}/noise.fits",
        psf_path=f"{dataset_path}/psf.fits",
        pixel_scales=pixel_scales,
    )

#Load mask
mask_custom = al.Mask.from_fits(
    file_path=f"{dataset_path}/mask.fits", hdu=0, pixel_scales=pixel_scales)
masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask_custom, inversion_uses_border=True)

#Plot
#aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask_custom, include=aplt.Include(border=False))

#Initializing the MGE model for the lens

mass_profile = al.mp.MGE(centre=(0.0, 0.0))                                #Mass model
mass_profile.MGE_comps(M=Total_Mass, sigma=Total_sigma_RAD,
                       q=Total_q, z_l=z_lens, z_s=z_source)               #Input parameters

mass_profile.MGE_Grid_parameters(masked_imaging.grid)
shear_comp = al.convert.shear_elliptical_comps_from(magnitude=mag_shear, phi=phi_shear) #external shear

lens_galaxy = al.Galaxy(
    redshift=z_lens,
    mass=mass_profile,
    shear=al.mp.ExternalShear(elliptical_comps=shear_comp)
)
#----------------------------------------------------------------------------------------------------#


#---------------------------------------- EMCEE -----------------------------------------------------#

### Priors

###boundaries. [lower, upper]
boundary = {'inc': [50, 140], 'beta': [-5, 5], 'ml': [0.5, 15], 'log_mbh':[7, 11],
                     'mag_shear': [0, 0.5], 'phi_shear': [0, 180], 'gamma': [-1, 2]}


"""
    Except for the gamma parameter, all other parameters has flat priors in log space, i.e, if the value is accepted its return 0 (log(1)), otherwise return -np.inf (log(0)). This choice is because we are assuming no previous knowledge about any parameters, except for the General Relativity, quantified in terms of gamma. 

    For gamma we assume a gaussian prior, centered around the General Relativity (gamma=1), because we expect that in case there is a deviation from GR, this deviation should be small.
"""


# parameter gaussian priors. [mean, sigma]
prior = {'gamma': [1.0, 0.5] }


def check_Deprojected_axial(parsDic):
    inc = np.radians(parsDic['inc'])
    #Stellar
    qintr_star = qobs_star_dat**2 - np.cos(inc)**2
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

    #Avoid beta = 1, because this could cause problems
    if parsDic['beta'] == 1:
        return -np.inf
    else:
        pass


    #Check if deprojected axial ratio is ok  (q' <=0 or q' <= 0.05) for the dynamical model.
    if not np.isfinite(check_Deprojected_axial(parsDic)):
        return -np.inf

    for keys in parsDic:
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
    The lines above is only for doble check, because we are assuming flat prior for all parameters. Once we got here, all values ​​have already been accepted, so just return 0.0 for each of them. Except for gamma, which has a gaussian prior.
    """

    rst += 0.0     #ml
    rst += 0.0     #beta
    rst += 0.0     #inc
    rst += 0.0     #log_mbh
    rst += 0.0     #mag_shear
    rst += 0.0     #phi_shear

    
    #Finaly gaussian prior for gamma
    rst += -0.5 * (parsDic['gamma'] - prior['gamma'][0])**2/prior['gamma'][1]**2     #gamma
    
    return rst



def Updt_Pyautolens(parsDic):
    '''
    Update the Lens mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''
    #Inclination
    inc_model = np.deg2rad(parsDic['inc'])                     #Get new inclination in radians
    
    #Stellar parameters
    ml_model = parsDic['ml'] #New Gaussian Mass-to-light ratio [M_sun/L_sun]
    
    Stellar_Mass_model = Lum_star_dat*ml_model         #Updt the stellar mass 
        
    #Total mass and new projected axis here we add the new SMBH mass
    Total_Mass_model = np.concatenate((Stellar_Mass_model, 10**parsDic['log_mbh']), axis=None)  #New total mass
    
    Total_q_model = np.concatenate((qobs_star_dat, qSMBH), axis=None)  #New axial  ratio

    #Model Updt
    mass_profile.MGE_Updt_parameters(Total_Mass_model,Total_sigma_RAD,
                                             Total_q_model, parsDic['gamma']) #Update the model



def Updt_JAM(parsDic):
    '''
       Update the dynamical mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''
    beta_model = np.full_like(surf_star_dat, parsDic['beta'])    #anisotropy parameter
    mbh_model = 10**parsDic['log_mbh']                           #BH mass
    
    #mass-to-light update 
    ml_model = parsDic['ml']
    
    #Model Updt
    Jampy_model.upt(inc=parsDic['inc'],ml=ml_model, beta=beta_model, mbh=mbh_model)


def JAM_log_likelihood(parsDic):
    """
        Perform JAM modeling and return the chi2
    """
    
    Updt_JAM(parsDic)               #Updt values for each iteration
    
    rmsModel, ml, chi2, chi2T = Jampy_model.run()
    return -0.5 * chi2T


def Pyautolens_log_likelihood(parsDic):
    """
        Perform Pyautolens modeling and return the chi2
    """
    
    Updt_Pyautolens(parsDic)        #Updt values for each iteration
    shear_comp = al.convert.shear_elliptical_comps_from(magnitude=parsDic['mag_shear'], phi=parsDic['phi_shear']
    ) #external shear
    #New lens model
    lens_galaxy = al.Galaxy(                                            
        redshift=z_lens,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=shear_comp),
    )
    
    
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=z_source)])
    source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[1]
    
    #Check if the model has converged. If not, return -inf
    try:
        rectangular = al.pix.Rectangular(shape=(40, 40))
        mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)
    
        inversion = al.Inversion(
            masked_dataset=masked_imaging,
            mapper=mapper,
            regularization=al.reg.Constant(coefficient=3.5),
    )
        chi2T = inversion.chi_squared_map.sum()
        return -0.5 * chi2T
    except:
        return -np.inf


def log_probability(pars):
    """
        Log-probability function for whole model.
        input:
            pars: current values in the Emcee sample.
        output:
            log probability for the combined model.
    """

    (ml, beta, inc, log_mbh, mag_shear, phi_shear, gamma) = pars
    
    parsDic = {'ml': ml, 'beta': beta, 'inc': inc,
                'log_mbh': log_mbh, 'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
    
    #Checking boundaries
    if not np.isfinite(check_boundary(parsDic)):
        return -np.inf
    #calculating the log_priors
    lp = log_prior(parsDic)

    return lp + JAM_log_likelihood(parsDic) + Pyautolens_log_likelihood(parsDic)



#Initial Positions of walkers
"""
    Pay close attention to the order in which the components are added. 
    They must follow the log_probability unpacking order.
"""
#In order: ML, beta, inclination, log_mbh, mag_shear, phi_shear, gamma
p0 = np.array([5.0, 0.0, 90, 9.0, 0.01, 90, 1.0])  

#Finally we initialize the walkers arround these positions above.
nwalkers = 32                                                   #Number of walkers
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
    burnin = 100                           #Number os burn in steps
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
            break
        old_tau = tau



    end = time.time()
    print('\n')
    print("Final")
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))