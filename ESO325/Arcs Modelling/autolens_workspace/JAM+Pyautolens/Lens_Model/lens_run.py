""""
This code runs in MPI mode
"""

"""
This model makes use of best fit of Jampy-model.
In this case, we are modelling only the lens, and we will left the following free parameters:
Mass per gaussian, shear, gamma.

All other parameter are fixed: beta, mbh and DM component.

The intent is find some region in the space parameter to explore afterwards with a complete modelling (Jampy+Pyautolens).
"""

#Frist we load the dynamical  model

#General packages
import numpy as np
from My_Jampy import JAM
import matplotlib.pyplot as plt
import emcee

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u

from os import path
path = "/home/carlos/Documents/GitHub/Master-Degree/ESO325/Arcs Modelling/autolens_workspace/"

#Reading data
y_px, x_px, vrms, erms = np.loadtxt('pPXF_rot_data.txt', unpack=True)                  #pPXF
surf_star_dat, sigma_star_dat, qstar_dat = np.loadtxt('JAM_Input.txt', unpack=True)    #photometry
surf_DM_dat, sigma_DM_dat, qDM_dat  = np.loadtxt('pseudo-DM Input.txt', unpack=True)   #DM

### Global Constantes

#Redshifth

z_lens = 0.035                                #Lens redshifth
z_source = 2.1                                #Source redshift

#Angular diameter distances
D_l = cosmo.angular_diameter_distance(z_lens)                       
D_s = cosmo.angular_diameter_distance(z_source)
D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)                      


#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s


#Dynamic os lens
"""
These values are based on the best fit of Jampy-Emcee.
"""
distance = D_l                                                #Angular diameter distance [Mpc]
inc = 89.74                                                   #Inclination [deg]
mbh =  (10**(7.77))*u.solMass                                 #Mass of black hole [M_sun]
beta0 = np.array([-0.46, -0.64, 0.72, -3.35, 0.37, 0.09, 0.05])      #Anisotropy parameter, one for each gaussian component

ML0 = np.array([6.75, 6.75, 5.48, 5.36, 5.32, 5.30, 5.29])*u.solMass/u.solLum     #Mass-to-light ratio per gaussian [M_sun/L_sun]


#DM
surf_DM_dat = (10**(9.21))*surf_DM_dat*(u.solMass/u.pc**2)          #Surface Density in M_sun/pc²
sigma_DM_dat_ARC = sigma_DM_dat*u.arcsec                            #Sigma in arcsec
sigma_DM_dat_PC = (sigma_DM_dat_ARC*D_l).to(u.pc, u.dimensionless_angles())    #Convert sigma in arcsec to sigma in pc
qDM_dat = np.ones_like(qDM_dat)*0.61                                                              #axial ratio of DM halo

#Stars
surf_star_dat = surf_star_dat*(u.solLum/u.pc**2)               #Surface luminosity Density in L_sun/pc²
sigma_star_dat_ARC = sigma_star_dat*u.arcsec                   #Sigma in arcsec
sigma_star_dat_PC = (sigma_star_dat_ARC*D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc
qstar_dat = qstar_dat                                          #axial ratio of star photometry

#----------------------------------------------------------------------------------------------------#


# JAMPY MODEL

#Defining some instrumental quantities and galaxy characteristics

muse_pixsize=0.6                                            #pixscale of IFU [arcsec/px]
muse_sigmapsf= 0.2420                                       #Sigma of psf from MUSE [arcsec]


#Create model
Jampy_model = JAM(ybin=y_px, xbin=x_px,inc=inc, distance=distance.value, mbh=mbh.value,
                  rms=vrms, erms=erms, beta=beta0, sigmapsf=muse_sigmapsf, pixsize=muse_pixsize)

#Add Luminosity component
Jampy_model.luminosity_component(surf_lum=surf_star_dat.value, sigma_lum=sigma_star_dat_ARC.value,
                                    qobs_lum=qstar_dat, ml=ML0.value)

#Add Dark Matter component
Jampy_model.DM_component(surf_dm=surf_DM_dat.value, sigma_dm=sigma_DM_dat_ARC.value, qobs_dm=qDM_dat)
plt.figure(figsize=(12,12))
Jampy_model.run(plot=True, quiet=False, vmax=400, vmin=300)

plt.show()


#Now the lens model

#Autolens Model packages

import autolens as al
import autolens.plot as aplt

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u


# Pyautolens Model

#Convert  surf_DM_dat to total mass per Guassian
Mass_DM_dat = 2*np.pi*surf_DM_dat*(sigma_DM_dat_PC**2)*qDM_dat     #Total mass per gaussian component in M_sun


#Convert surf_star_dat to total Luminosity per Guassian and then to total mass per gaussian
Lum_star_dat = 2*np.pi*surf_star_dat*(sigma_star_dat_PC**2)*qstar_dat    #Total luminosity per gaussian component in L_sun



#Update the stellar mass based on M/L.
Mass_star_dat = Lum_star_dat*ML0                          #Total star mass per gaussian in M_sun

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
surfBH_PC = mbh/(2*np.pi*sigmaBH_PC**2)                                  #Mass surface density of SMBH
qSMBH = 1.                                                               #Assuming a circular gaussian
Mass_SMBH_dat = 2*np.pi*surfBH_PC*(sigmaBH_PC**2)*qSMBH                  #SMBH Total mass 


#Defining the general inputs for the model

i = np.deg2rad(inc)*u.rad                                         #Inclination angle in rad
Total_Mass = np.concatenate((Mass_star_dat, Mass_DM_dat,
                                 Mass_SMBH_dat), axis=None)       #Mass per gaussian component in M_sun
Total_q = np.concatenate((qstar_dat, qDM_dat, qSMBH), axis=None)  #Total axial ratio per gaussian


#Total_q_proj = np.sqrt(Total_q**2 - np.cos(i)**2)/np.sin(i)    #Total projected axial ratio per gaussian
Total_sigma_ARC = np.concatenate((sigma_star_dat_ARC, sigma_DM_dat_ARC, sigmaBH_ARC), axis=None)  #Total sigma per gaussian in arcsec
Total_sigma_RAD = Total_sigma_ARC.to(u.rad)                    #Total sigma per gaussian in radians


#Reading fits file with the arcs data

dataset_type = "JAM+Pyautolens"
dataset_name = "Data"
dataset_path = path+f"{dataset_type}/{dataset_name}"

#Load data
imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/arcs_resized.fits",
        noise_map_path=f"{dataset_path}/noise_map_resized.fits",
        psf_path=f"{dataset_path}/psf.fits",
        pixel_scales=0.04,
    )

#Load mask
mask_custom = al.Mask.from_fits(
    file_path=f"{dataset_path}/mask_gui_2.fits", hdu=0, pixel_scales=imaging.pixel_scales
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask_custom, inversion_uses_border=False)

#Plot image+mask
#aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask_custom, include=aplt.Include(border=False))


# __Defining the MGE mass model__

#Initializing the MGE model for the lens

mass_profile = al.mp.MGE(centre=(0.0, 0.0))                             #Mass model
mass_profile.MGE_comps(M=Total_Mass.value, sigma=Total_sigma_RAD.value,
                       q=Total_q, z_l=z_lens, z_s=z_source)  #Input parameters

mass_profile.MGE_Grid_parameters(masked_imaging.grid)               #Grid with the data

#Lens Model
lens_galaxy = al.Galaxy(                                            
        redshift=z_lens,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=(0,0)),
    )


#--------------------------------------- EMCEE -----------------------------------------------------#


### Priors

# parameter boundaries. [lower, upper]
boundary = {'ml': [0.5, 15], 'qDM': [0.15, 1], 'mag_shear': [0, 2], 'phi_shear': [0, 180],
                     'gamma': [-2, 2] }

"""
    Except for the gamma parameter, all other parameters has flat priors in log space, i.e, if the value is accepted its return 0 (log(1)), otherwise return -np.inf (log(0)). This choice is because we are assuming no previous knowledge about any parameters, except for the General Relativity, quantified in terms of gamma. 

    For gamma we assume a gaussian prior, centered around the General Relativity (gamma =1), because we expect that in case there is a deviation from GR, this deviation should be small.
"""

# parameter gaussian priors. [mean, sigma]
prior = {'gamma': [1.0, 0.5] }


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



def check_boundary(parsDic):
    """
        Check whether parameters are within the boundary limits
        input
            parsDic: parameter dictionary {'paraName', value}
        output
            -np.inf or 0.0
    """   
    
    
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


    #Check if the others parameters are within the boundary limits
    keys = set(parsDic.keys())
    excludes = set(['ml'])  #Exclude beta and ml, because we already verify above


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
    rst += 0.0     #mag_shear
    rst += 0.0     #phi_shear

    """
    Finaly gaussian prior for gamma
    """
    rst += -0.5 * (parsDic['gamma'] - prior['gamma'][0])**2/prior['gamma'][1]**2         #gamma
    
    return rst


def Updt_Pyautolens(parsDic):
    '''
    Update the Lens mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''
    
    #Stellar parameters
    ml_model = np.array([parsDic['ml'][0], parsDic['ml'][0], parsDic['ml'][1], 
                            parsDic['ml'][2], parsDic['ml'][3], parsDic['ml'][4], parsDic['ml'][5]])
    
    Stellar_Mass_model = (Lum_star_dat*ml_model).value         #Updt the stellar mass 

    
    #Total mass
    Total_Mass_model = np.concatenate((Stellar_Mass_model, Mass_DM_dat, 
                                            Mass_SMBH_dat), axis=None)  #New total mass
    
    Total_q_model = np.concatenate((qstar_dat, qDM_dat, qSMBH), axis=None)     #New axial  ratio                       

    #Model Updt
    mass_profile.MGE_Updt_parameters(Total_Mass_model,Total_sigma_RAD.value,
                                             Total_q_model, parsDic['gamma']) #Update the model

            

def Pyautolens_log_likelihood(parsDic):
    """
        Perform Pyautolens modeling and return the chi2
    """
    
    Updt_Pyautolens(parsDic)        #Updt values for each iteration
    shear_comp = al.convert.shear_elliptical_comps_from(magnitude=parsDic['mag_shear'],
                                                             phi=parsDic['phi_shear'])
    #New lens model
    lens_galaxy = al.Galaxy(                                            
        redshift=z_lens,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=shear_comp))
    
    
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=z_source)])
    source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[1]
    
    #check if the integral converge. If not, return -np.inf
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

    (m1, m2, m3, m4, m5, m6, mag_shear, phi_shear, gamma) = pars
    
    ml =  np.array([m1, m2, m3, m4, m5, m6])  #we have 7-gaussians, but we assume ml1=ml2.
                                                #in the codes above we updt the parameters in that sense
    
    parsDic = {'ml': ml, 'mag_shear': mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
    
    #Checking boundaries
    if not np.isfinite(check_boundary(parsDic)):
        return -np.inf
    #calculating the log_priors
    lp = log_prior(parsDic)

    return lp + Pyautolens_log_likelihood(parsDic)


"""
    We save the results in a tables. 
    It marks the number of iterations, the mean acceptance fraction an the running time. 
"""
np.savetxt('Output_LogFile.txt', np.column_stack([0, 0, 0]),
                            fmt=b'	%i	 %e			 %e	 ', 
                            header="Output table for the combined model: Lens + Dynamic.\n Iteration	 Mean acceptance fraction	 Processing Time")



""""
 For the initial guesses of ML we will use the best fit  of Jampy-Emcee with a gaussian ball error around it variables tagged with <name>_std are the standard deviation of the parameter <name>.
 For the shear we will use the best fit of Collett's paper. 
"""

#Defining initial guesses

ml = ML0.value
ml_std = ml*0.2                                 #30% of each value

mag_shear = np.array([0.02])
mag_shear_std = mag_shear*0.1                   #10% of the value

phi_shear = np.array([119])
phi_shear_std = phi_shear*0.1                   #10% of the value

gamma = np.array([1.0])
gamma_std = gamma*0.2                           #20% of the value

"""
    Pay close attention to the order in which the components are added. 
    They must follow the log_probability unpacking order.
"""

#Initial Positions of walkers
##Here we append all the variables and stds in a single array.
p0 = np.append(ml, mag_shear)
p0 = np.append(p0,[phi_shear, gamma])

p0_std = np.append(ml_std, mag_shear_std)
p0_std = np.append(p0_std, [phi_shear_std, gamma_std])

#Finally we initialize the walkers with a gaussian ball around the best Collet's fit.
nwalkers = 200                                                  #Number of walkers
pos = emcee.utils.sample_ball(p0, p0_std, nwalkers)             #Initial position of all walkers


nwalkers, ndim = pos.shape

with MPIPool() as pool:

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    #Print the number os cores/workers
    print("Workers nesse job:", pool.workers)
    print("Start")

    #Backup
    filename = "ESO325_Lens_Model.h5"
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