"""
This code makes use of python multiprocesing. The  number of workers are define in the following line.
"""
workers = 3

#Control time packages
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"

#General packages
import numpy as np
import My_Jampy
import emcee
import matplotlib.pyplot as plt


#Multiprocessing
from multiprocessing import Pool


#Autolens Model packages

import autolens as al
import autolens.plot as aplt

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u
#------------------------------------------------------------------------------------#


## DATA

#Rewading data from: photometry, dark matter and dynamics
surf_star_dat, sigma_star_dat, qstar_dat = np.loadtxt('JAM Input.txt', unpack=True)   #Star
surf_DM_dat, sigma_DM_dat, qDM_dat = np.loadtxt('pseudo-DM Input.txt', unpack=True)   #DM
y_px, x_px, vel,  disp, chi, dV, dsigma = np.loadtxt('pPXF DATA.txt', unpack=True)    #pPXF  


### Global Constantes

#Lens parameters

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


### Global Parameters
"""
    To inicialize the model, we set a ML igual to 1 for every component in Star MGE. But it's only necessary for initialize the model. During the non-linear search, this ML will be updated constantly until the best fit.
    Same as above for the Anisotropy parameter.
"""   

inc = 90.                                    #Assumed galaxy inclination [deg]                  
distance = D_l                               #Distance in Mpc
mbh =  1e8*u.solMass                         #Mass of SMBH in solar masses
beta = np.zeros(surf_star_dat.shape)         #Anisotropy parameter. One for each gaussian component 
ML = np.ones(surf_star_dat.shape)*u.solMass/u.solLum          #Mass to light ratio per gaussian in M_sun/L_sun


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

sigmapsf = 0.2420                                   #Sigma of psf from MUSE [arcsec]
pixsize = 0.6                                       #pixel scale [px/arcsec] from data (binned here)
e = 0.24                                            #galaxy ellipticity. Value from find_my_galaxy

#Selecting pixels where we want to compute the model

x_good = []
y_good = []
disp_good = []
vel_good = []
dV_good = []
dsigma_good = []

for i in range(len(disp)):
    r = np.sqrt((x_px[i]*pixsize)**2 + ((y_px[i])*pixsize/(1-e))**2)
    if r < 5:
        x_good.append(x_px[i])
        y_good.append(y_px[i])
        disp_good.append(disp[i])
        vel_good.append(vel[i])
        dV_good.append(dV[i])
        dsigma_good.append(dsigma[i])

#Calculando a Velocidade Vrms
"""
    Note that you first identify the px with the greatest velocity dispersion, in order to identify the center of the galaxy. After that, we calculate the rotation velocity with respect to that center. Only then can we calculate the Vrms velocity and the associated propagated erms error.
"""

idx_max = np.where(np.array(disp_good) == max(disp_good))

vel_good = vel_good - vel_good[idx_max[0][0]]
vrms = np.sqrt(np.array(vel_good)**2 + np.array(disp_good)**2) #Vrms velocity
erms = np.sqrt((np.array(dV_good)*np.array(vel_good))**2 + (np.array(dsigma_good)*np.array(disp_good))**2)/vrms #error in vrms



#Defining the input data from dynamical modeling

    #Position [arcsec], where the model will be computed
xbin = np.array(x_good)*pixsize
ybin = np.array(y_good)*pixsize

r = np.sqrt(xbin**2 + (ybin/(1-e))**2)              #Radius in the plane of the disk
rms = vrms                                          #Vrms field in km/s
erms = erms                                         #1-sigma erro in velocity dispersion
goodBins =    (r > 0)                               #Good bins for model

#Initializing the dynamic model
Jampy_Model = My_Jampy.Jam_axi_rms(ybin=ybin, xbin=xbin,beta=beta, mbh=mbh.value, distance=distance.value,
                                surf_lum=surf_star_dat.value, sigma_lum=sigma_star_dat_ARC.value, qobs_lum=qstar_dat,
                                surf_DM=surf_DM_dat.value, sigma_DM=sigma_DM_dat_ARC.value, qobs_DM=qDM_dat,
                                ml=ML.value, goodBins=goodBins, sigmapsf=sigmapsf, rms=rms, erms=erms,
                                pixsize=pixsize, inc=inc)
                                
#----------------------------------------------------------------------------------------------------#


# Pyautolens Model

#Convert  surf_DM_dat to total mass per Guassian
Mass_DM_dat = 2*np.pi*surf_DM_dat*(sigma_DM_dat_PC**2)*qDM_dat     #Total mass per gaussian component in M_sun


#Convert surf_star_dat to total Luminosity per Guassian and then to total mass per gaussian
Lum_star_dat = 2*np.pi*surf_star_dat*(sigma_star_dat_PC**2)*qstar_dat    #Total luminosity per gaussian component in L_sun



#Update the stellar mass based on M/L.
Mass_star_dat = Lum_star_dat*ML                          #Total star mass per gaussian in M_sun

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


Total_q_proj = np.sqrt(Total_q**2 - np.cos(i)**2)/np.sin(i)    #Total projected axial ratio per gaussian
Total_sigma_ARC = np.concatenate((sigma_star_dat_ARC, sigma_DM_dat_ARC, sigmaBH_ARC), axis=None)  #Total sigma per gaussian in arcsec
Total_sigma_RAD = Total_sigma_ARC.to(u.rad)                    #Total sigma per gaussian in radians



#Reading fits file with the arcs data

dataset_type = "JAM+Pyautolens"
dataset_name = "Data"
dataset_path = f"/home/carlos/Documents/GitHub/Master-Degree/ESO325/Arcs Modelling/autolens_workspace/{dataset_type}/{dataset_name}"

#Load data
imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/arcs_resized.fits",
        noise_map_path=f"{dataset_path}/noise_map_resized.fits",
        psf_path=f"{dataset_path}/psf.fits",
        pixel_scales=0.04,
    )

#Load mask
mask_custom = al.Mask.from_fits(
    file_path=f"{dataset_path}/mask gui.fits", hdu=0, pixel_scales=imaging.pixel_scales
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask_custom)


# __Defining the MGE mass model__

#Initializing the MGE model for the lens

mass_profile = al.mp.MGE(centre=(0.0, 0.0))                             #Mass model
mass_profile.MGE_comps(M=Total_Mass.value, sigma=Total_sigma_RAD.value,
                       q=Total_q_proj.value, z_l=z_lens, z_s=z_source)  #Input parameters

mass_profile.MGE_Grid_parameters(masked_imaging.grid)               #Grid with the data

#Lens Model
lens_galaxy = al.Galaxy(                                            
        redshift=0.035,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=(0,0)),
    )





#--------------------------------------- EMCEE -----------------------------------------------------#


### Priors

# parameter boundaries. [lower, upper]
boundary = {'inc': [70, 120], 'beta': [-5, 5], 'ml': [0.5, 15], 'log_mbh':[7, 11],
             'qDM': [0.15, 1], 'log_rho_s':[6, 13], 'mag_shear': [-0.2, 0.2], 
             'phi_shear': [-0.2, 0.2], 'gamma': [-2, 2] }

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
    #Inclination
    inc_model = np.deg2rad(parsDic['inc'])                     #Get new inclination in radians
    
    #Stellar parameters
    ml_model = np.array([parsDic['ml'][0], parsDic['ml'][0], parsDic['ml'][1], 
                            parsDic['ml'][2], parsDic['ml'][3], parsDic['ml'][4], parsDic['ml'][5]])
    
    Stellar_Mass_model = (Lum_star_dat*ml_model).value         #Updt the stellar mass 
    
    #DM parameters
    qDM_model = np.ones(qDM_dat.shape)*parsDic['qDM']          #Updt DM axial ratio 
    Mass_DM_model = (10**parsDic['log_rho_s'])*(2*np.pi*surf_DM_dat*(sigma_DM_dat_PC**2)*qDM_model).value    #Updt DM Mass
    
    
    #Total mass and new projected axis here we add the new SMBH mass
    Total_Mass_model = np.concatenate((Stellar_Mass_model, Mass_DM_model, 
                                            10**parsDic['log_mbh']), axis=None)  #New total mass
    
    Total_q_model = np.concatenate((qstar_dat, qDM_model, qSMBH), axis=None)     #New axial  ratio                       
    Total_q_proj_model = (np.sqrt(Total_q_model**2 - np.cos(inc_model)**2)/np.sin(inc_model))          #New projected axial ratio

    #Model Updt
    mass_profile.MGE_Updt_parameters(Total_Mass_model,Total_sigma_RAD.value,
                                             Total_q_proj_model, parsDic['gamma']) #Update the model

                
def Updt_JAM(parsDic):
    '''
       Update the dynamical mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''
    surf_DM_model = surf_DM_dat.value*(10**parsDic['log_rho_s']) #DM mass surface density
    qDM_model = np.ones(qDM_dat.shape)*parsDic['qDM']            #DM axial ratio
    beta_model = np.array(parsDic['beta'])                       #anisotropy parameter
    mbh_model = 10**parsDic['log_mbh']                           #BH mass
    
    #mass-to-light update 
    ml_model = np.array([parsDic['ml'][0], parsDic['ml'][0], parsDic['ml'][1], 
                            parsDic['ml'][2], parsDic['ml'][3], parsDic['ml'][4], parsDic['ml'][5]])

    
    #Model Updt
    Jampy_Model.Updt_parameters(surf_DM=surf_DM_model, qobs_DM=qDM_model,
                                 beta=beta_model, ml=ml_model, inc=parsDic['inc'],
                                    mbh=10**parsDic['log_mbh'])



def JAM_log_likelihood(parsDic):
    """
        Perform JAM modeling and return the chi2
    """
    
    Updt_JAM(parsDic)               #Updt values for each iteration
    
    rmsModel, ml, chi2, chi2T = Jampy_Model.run()
    return -0.5 * chi2T



def Pyautolens_log_likelihood(parsDic):
    """
        Perform Pyautolens modeling and return the chi2
    """
    
    Updt_Pyautolens(parsDic)        #Updt values for each iteration
    
    #New lens model
    lens_galaxy = al.Galaxy(                                            
        redshift=0.035,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=(parsDic['mag_shear'], parsDic['phi_shear'])),
    )
    
    
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=2.1)])
    source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[1]
    
    #check if the integral converge. If not, return -np.inf
    if np.isnan(source_plane_grid[0,0]):
        return -np.inf
    
    rectangular = al.pix.Rectangular(shape=(40, 40))
    mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)
    
    inversion = al.Inversion(
        masked_dataset=masked_imaging,
        mapper=mapper,
        regularization=al.reg.Constant(coefficient=3.5),
    )
    chi2T = inversion.chi_squared_map.sum()
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
         inc, qDM, log_rho_s, log_mbh, mag_shear, phi_shear, gamma) = pars
    
    ml =  np.array([m1, m2, m3, m4, m5, m6])  #we have 7-gaussians, but we assume ml1=ml2.
                                                #in the codes above we updt the parameters in that sense
    
    beta =  np.array([b1, b2, b3, b4, b5, b6, b7])
    parsDic = {'ml': ml, 'inc': inc, 'qDM': qDM, 'log_rho_s': log_rho_s, 'log_mbh': log_mbh,
                  'mag_shear': mag_shear, 'phi_shear': phi_shear, 'gamma': gamma, 'beta': beta}
    
    #Checking boundaries
    if not np.isfinite(check_boundary(parsDic)):
        return -np.inf
    #calculating the log_priors
    lp = log_prior(parsDic)

    return lp + Pyautolens_log_likelihood(parsDic) + JAM_log_likelihood(parsDic)

"""
    We save the results in two tables. 
    The first marks the number of iterations, the mean acceptance fraction an the running time. 
    The second marks the last fit values for each parameter.
"""
np.savetxt('Output_LogFile.txt', np.column_stack([0, 0, 0]),
                            fmt=b'	%i	 %e			 %e	 ', 
                            header="Output table for the combined model: Lens + Dynamic.\n Iteration	 Mean acceptance fraction	 Processing Time")

np.savetxt("LastFit.txt", np.column_stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]), 
fmt=b'%e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	', 
header="Iteration	 ML1/2	 ML3	 ML4	 ML5	 ML6	 ML7	 b1	 b2	 b3	 b4	 b5	 b6	 b7	 Inc	 qDM	 Logrho_s	 LogMBH	 MagShear	 PhiShear	 gamma")



#Where we use the multiprocesing avaible in python.
with Pool(processes=workers) as pool:

#Frist we read the last sample
    filename = "save.h5"
    read = emcee.backends.HDFBackend("save.h5")
    nwalkers, ndim = read.shape

        #Defining the moves
    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    


    print("Workers nesse job:", pool.workers)
    print("Início")
    

    
    
      #Initialize the new sampler
    new_sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=read, moves=moves)
        #and get the last position
    state = new_sampler.get_last_sample()
    
    nsteps = 50000

     # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)

     # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    start = time.time()
    global_time = time.time()
    for sample in new_sampler.sample(state, iterations=nsteps, progress=True):
        # Only check convergence every 100 steps
        if new_sampler.iteration % 100:
            continue
        print("\n")
        print("##########################")
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = new_sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1



        #Update a table output with acceptance
        table = np.loadtxt("Output_LogFile.txt")


        iteration = new_sampler.iteration
        accept = np.mean(new_sampler.acceptance_fraction)
        total_time = time.time() - global_time
        upt = np.column_stack([iteration, accept, total_time])

        np.savetxt('Output_LogFile.txt', np.vstack([table, upt]),
                                fmt=b'	%i	 %e			 %e	 ', 
                                header="Iteration	 Mean acceptance fraction	 Processing Time")

        #Update table output with last best fit
        last_fit_table = np.loadtxt("LastFit.txt")
        flat_samples = new_sampler.get_chain()
        values = []
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            values.append(mcmc[1])

        values = np.array(values)
        upt = np.append(iteration, values)

        np.savetxt("LastFit.txt",np.vstack([last_fit_table, upt]),
                    fmt=b'%e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	', 
                    header="Iteration	 ML1/2	 ML3	 ML4	 ML5	 ML6	 ML7	 b1	 b2	 b3	 b4	 b5	 b6	 b7	 Inc	 qDM	 Logrho_s	 LogMBH	 MagShear	 PhiShear	 gamma")
 

        # Check convergence
        converged = np.all(tau * 100 < new_sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau



    end = time.time()
    print('\n')
    print("Final")
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))