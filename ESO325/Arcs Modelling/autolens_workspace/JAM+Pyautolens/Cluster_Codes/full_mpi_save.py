

#General packages
import numpy as np
import My_Jampy
import emcee
import matplotlib.pyplot as plt

from time import perf_counter as clock
from schwimmbad import MPIPool
import time
import os


os.environ["OMP_NUM_THREADS"] = "1"

#Autolens Model packages

import autolens as al
import autolens.plot as aplt
#print("Pyautolens version:", al.__version__)

#from pyprojroot import here
import numpy as np

from time import perf_counter as clock

from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u

#workspace_path = str(here())
#print("Workspace Path: ", workspace_path)
#------------------------------------------------------------------------------------#
## DATA

#Lendo os dados de fotometria, DM halo e cinemática
surf_star_dat, sigma_star_dat, qstar_dat = np.loadtxt('JAM Input.txt', unpack=True) #Star
surf_DM_dat, sigma_DM_dat, qDM_dat = np.loadtxt('pseudo-DM Input.txt', unpack=True) #DM
y_px, x_px, vel,  disp, chi, dV, dsigma = np.loadtxt('pPXF DATA.txt', unpack=True)  #pPXF  

### Global Constantes

#Lens parameters

z_lens = 0.035
z_source = 2.1

D_l = cosmo.angular_diameter_distance(z_lens)
D_s = cosmo.angular_diameter_distance(z_source)
D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s

### Global Parameters

#To inicialize the model, we set an ML igual to 1 for every component in Star MGE.
    #But it's only necessary for initialize the model. 
    #During the non-linear search, this ML will be updated constantly until the best fit.
    #Same as above for the Anisotropy parameter.
    
inc = 120.                                                    #Assumed galaxy inclination                  
distance = D_l                                                #Distance in Mpc
mbh =  1e8*u.solMass                                          #Mass of SMBH in solar masses
beta = np.zeros(surf_star_dat.shape)                          #Anisotropy parameter. One for each gaussian component 
ML = np.ones(surf_star_dat.shape)*u.solMass/u.solLum          #Mass to light ratio per gaussian in M_sun/L_sun

#DM
surf_DM_dat = surf_DM_dat*(u.solMass/u.pc**2)                                    #Surface Density in M_sun/pc²
sigma_DM_dat_ARC = sigma_DM_dat*u.arcsec                                         #Sigma in arcsec
sigma_DM_dat_PC = (sigma_DM_dat_ARC*D_l).to(u.pc, u.dimensionless_angles())      #Convert sigma in arcsec to sigma in pc
qDM_dat = qDM_dat                                                                #axial ratio of DM halo

#Stars
surf_star_dat = surf_star_dat*(u.solLum/u.pc**2)                                #Surface luminosity Density in L_sun/pc²
sigma_star_dat_ARC = sigma_star_dat*u.arcsec                                    #Sigma in arcsec
sigma_star_dat_PC = (sigma_star_dat_ARC*D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc
qstar_dat = qstar_dat                                                           #axial ratio of star photometry

#--------------------------------------------------------------------------------------------------------#

# JAMPY MODEL

#Definindo algumas quantidades dos instrumentos e características da galáxia

sigmapsf = 0.2420                                   #Sigma psf de onde foram coletados os dados de cinemática, em arcsec
pixsize = 0.6                                       #pixel scale, em px/arcsec, dos dados de cinemática
e = 0.24                                            #elipticidade da galáxia. Valor encontrado pelo find_my_galaxy


#Selecionando os pixels onde queremos calcular o modelo

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
    #Note que primeiro identificamos o px com a maior dispersão de vlocidades, de modo a identificar o centro da
    #galáxia. Após isso, calculamos a velocidade de rotação com relação a esse centro. Somente então podemos
    #calcular a velocidade Vrms e o erro erms propagado associado.
idx_max = np.where(np.array(disp_good) == max(disp_good))

vel_good = vel_good - vel_good[idx_max[0][0]]
vrms = np.sqrt(np.array(vel_good)**2 + np.array(disp_good)**2) #Vrms velocity
erms = np.sqrt((np.array(dV_good)*np.array(vel_good))**2 + (np.array(dsigma_good)*np.array(disp_good))**2)/vrms #error in vrms

#Definindo os dados de entrada do modelo dinâmico

    #Posição, em arcsec, onde vamos calcular o modelo
xbin = np.array(x_good)*pixsize
ybin = np.array(y_good)*pixsize

r = np.sqrt(xbin**2 + (ybin/(1-e))**2)              #Radius in the plane of the disk
rms = vrms                                          #Vrms field in km/s
erms = erms                                         #1-sigma erro na dispersão
goodBins =    (r > 0)                               #Informa quais valores de r são bons para gerar o modelo.

#Inicializando o modelo dinâmico
Jampy_Model = My_Jampy.Jam_axi_rms(ybin=ybin, xbin=xbin,beta=beta, mbh=mbh.value, distance=distance.value,
                                surf_lum=surf_star_dat.value, sigma_lum=sigma_star_dat_ARC.value, qobs_lum=qstar_dat,
                                surf_DM=surf_DM_dat.value, sigma_DM=sigma_DM_dat_ARC.value, qobs_DM=qDM_dat,
                                ml=ML.value, goodBins=goodBins, sigmapsf=sigmapsf, rms=rms, erms=erms,
                                pixsize=pixsize, inc=inc)
                                
#--------------------------------------------------------------------------------------------------------------#

# Pyautolens Model

#Convert  surf_DM_dat to total mass per Guassian

Mass_DM_dat = 2*np.pi*surf_DM_dat*(sigma_DM_dat_PC**2)*qDM_dat      #Total mass per gaussian component in M_sun

#print("Total Mass per Gaussian component in DM profile:")
#print(Mass_DM_dat)

#Convert surf_star_dat to total Luminosity per Guassian and then to total mass per gaussian

Lum_star_dat = 2*np.pi*surf_star_dat*(sigma_star_dat_PC**2)*qstar_dat    #Total luminosity per gaussian component in L_sun

#print("Total Luminosity per Gaussian component of Stars:")
#print(Lum_star_dat)

#Update the stellar mass based on M/L.

Mass_star_dat = Lum_star_dat*ML                          #Total star mass per gaussian in M_sun

#print("Total Mass per Gaussian component of Star:")
#print(Mass_star_dat)

#Inserting a Gaussian to represent SMBH at the center of the galaxy

sigmaBH_ARC = 0.01*u.arcsec
'''
        This scalar gives the sigma in arcsec of the Gaussian representing the
        central black hole of mass MBH (See Section 3.1.2 of `Cappellari 2008.
        <http://adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_)
        The gravitational potential is indistinguishable from a point source
        for ``radii > 2*RBH``, so the default ``RBH=0.01`` arcsec is appropriate
        in most current situations.

        ``RBH`` should not be decreased unless actually needed!
    '''


sigmaBH_PC = (sigmaBH_ARC*D_l).to(u.pc, u.dimensionless_angles())        #Sigma of the SMBH in pc
surfBH_PC = mbh/(2*np.pi*sigmaBH_PC**2)                                  #Mass surface density of SMBH
qSMBH = 1.                                                               #Assuming a circular gaussian
Mass_SMBH_dat = 2*np.pi*surfBH_PC*(sigmaBH_PC**2)*qSMBH                  #SMBH Total mass 

#print("Total Mass of SMBH")
#print(Mass_SMBH_dat)

#Defining the general inputs for the model
i = np.deg2rad(inc)*u.rad                                                             #Inclination angle in rad
Total_Mass = np.concatenate((Mass_star_dat, Mass_DM_dat, Mass_SMBH_dat), axis=None)   #Mass per gaussian component in M_sun
Total_q = np.concatenate((qstar_dat, qDM_dat, qSMBH), axis=None)                      #Total axial ratio per gaussian


Total_q_proj = np.sqrt(Total_q**2 - np.cos(i)**2)/np.sin(i)                                       #Total projected axial ratio per gaussian
Total_sigma_ARC = np.concatenate((sigma_star_dat_ARC, sigma_DM_dat_ARC, sigmaBH_ARC), axis=None)  #Total sigma per gaussian in arcsec
Total_sigma_RAD = Total_sigma_ARC.to(u.rad)                                                       #Total sigma per gaussian in radians

#print("Total Mass per Gaussian of Model:")
#print(Total_Mass)


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

#Plot
'''
aplt.Imaging.subplot_imaging(
    imaging=imaging, mask=mask_custom, include=aplt.Include(border=True)
)
'''

### __Defining the MGE mass model__

#Iniciando o modelo MGE para a lente

mass_profile = al.mp.MGE(centre=(0.0, 0.0))                         #Definindo o modelo de massa
mass_profile.MGE_comps(M=Total_Mass.value, sigma=Total_sigma_RAD.value,
                       q=Total_q_proj.value, z_l=z_lens, z_s=z_source)        #Defindo os dados de entrada

mass_profile.MGE_Grid_parameters(masked_imaging.grid)               #Criando a grid de parâmetros para o cálculo
                                                                         #em paralelo
#Criando o modelo da lente
lens_galaxy = al.Galaxy(                                            
        redshift=0.035,
        mass=mass_profile,
        shear=al.mp.ExternalShear(elliptical_comps=(0,0)),
    )


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ EMCEE -----------------------------------------------------#

### Priors

# parameter boundaries. [lower, upper]
boundary = {'inc': [70, 120], 'beta': [-5, 5], 'ml': [0.5, 15], 'log_mbh':[7, 11], 'qDM': [0.15, 1],
                'log_rho_s':[6, 13], 'mag_shear': [-0.2, 0.2], 'phi_shear': [-0.2, 0.2],
                'gamma': [-2, 2]
               }

# parameter gaussian priors. [mean, sigma]
prior = {'inc': [90, 10], 'beta': [0.0, 2], 'ml': [1.0, 5],'log_mbh':[8, 3], 'qDM': [0.5, 3e-1],
            'log_rho_s':[10, 4], 'mag_shear': [0, 0.1], 'phi_shear': [0, 0.1], 'gamma': [1.0, 0.5]
            }

def check_ML_Grad(ml):
    
    #Check if the mass-to-light ratio is descending and inside boundaries.
    
    for i in range(len(ml)):
        if boundary['ml'][0] < ml[i] < boundary['ml'][1]:
            pass
        else:
            return -np.inf
        
    for i in range(len(ml) -1):
        if ml[i] >= ml[i+1]:
            pass
        else:
            return -np.inf
    
    return 0.0

def check_boundary(parsDic):
    '''
    Check whether parameters are within the boundary limits
    input
      parsDic: parameter dictionary {'paraName', value}
    output
      -np.inf or 0.0
    '''   
    #Check if any deprojected axial ratio is too low (q' <=0 or q' <= 0.05) for the dynamical model.
    inc = np.radians(parsDic['inc'])
        #Stellar
    qintr_star = qstar_dat**2 - np.cos(inc)**2
    if np.any(qintr_star <= 0):
        #print('erro no qstar1')
        return -np.inf
    
    qintr_star = np.sqrt(qintr_star)/np.sin(inc)
    if np.any(qintr_star <= 0.05):
        #print('erro no qstar')
        return -np.inf
    
        #DM
    qintr_DM = parsDic['qDM']**2 - np.cos(inc)**2
    if qintr_DM <= 0:
        #print('erro no qdm1')
        return -np.inf
    
    qintr_DM = np.sqrt(qintr_DM)/np.sin(inc)
    if qintr_DM <= 0.05:
        #print('erro no qdm')
        return -np.inf
    
    #Check if ml is gradient and inside boundaries
    if not np.isfinite(check_ML_Grad(parsDic['ml'])):
        #print('erro ml')
        return -np.inf
    
    #Check if beta is within boundary limits
    for i in range(len(parsDic['beta'])):
        if boundary['beta'][0] < parsDic['beta'][i] < boundary['beta'][1] :
            pass
        else:
            #print('erro beta')
            return -np.inf
        
    for i in range(len(parsDic['beta'])):
        if parsDic['beta'][i] !=1:
            pass
        else:
            return -np.inf

    #Check if the others parameters are within the boundary limits
    keys = set(parsDic.keys())
    excludes = set(['ml', 'beta'])
    
    
    for keys in keys.difference(excludes):
        if boundary[keys][0] < parsDic[keys] < boundary[keys][1]:
            pass
        else:
            #print('erro',keys)
            return -np.inf
    return 0.0
    


def log_prior(parsDic):
    '''
    Calculate the gaussian prior lnprob
    input
      parsDic: parameter dictionary {'paraName', value}
    output
      lnprob
    '''
    
    rst = 0
    
    #log_prior for mass-to-light
    for i in range(len(parsDic['ml'])):
        rst += -0.5 * (parsDic['ml'][i] - prior['ml'][0])**2/prior['ml'][1]**2
        
    #log_prior for beta
    for i in range(len(parsDic['beta'])):
        rst += -0.5 * (parsDic['beta'][i] - prior['beta'][0])**2/prior['beta'][1]**2
        
    #log_prior for others parameters
    keys = set(parsDic.keys())
    excludes = set(['ml', 'beta'])
    for keys in keys.difference(excludes):
        rst += -0.5 * (parsDic[keys] - prior[keys][0])**2/prior[keys][1]**2
    
    return rst
        


def Updt_Pyautolens(parsDic):
    '''
    Update the Lens mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''
    #Inclination
    inc_model = np.deg2rad(parsDic['inc'])                          #Get new inclination in radians
    
    #Stellar parameters
    Stellar_Mass_model = (Lum_star_dat*parsDic['ml']).value         #Updt the stellar mass 
    
    #DM parameters
    qDM_model = np.ones(qDM_dat.shape)*parsDic['qDM']                                              #Updt DM axial ratio 
    Mass_DM_model = (10**parsDic['log_rho_s'])*(2*np.pi*surf_DM_dat*(sigma_DM_dat_PC**2)*qDM_model).value    #Updt DM Mass
    
    
    #Total mass and new projected axis here we add the new SMBH mass
    Total_Mass_model = np.concatenate((Stellar_Mass_model, Mass_DM_model, 10**parsDic['log_mbh']), axis=None)  #New total mass
    Total_q_model = np.concatenate((qstar_dat, qDM_model, qSMBH), axis=None)                           
    Total_q_proj_model = (np.sqrt(Total_q_model**2 - np.cos(inc_model)**2)/np.sin(inc_model))          #New projected axial ratio
    mass_profile.MGE_Updt_parameters(Total_Mass_model,Total_sigma_RAD.value, Total_q_proj_model, parsDic['gamma'])       #Update the model
    
    
def Updt_JAM(parsDic):
    '''
       Update the dynamical mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''
    surf_DM_model = surf_DM_dat.value*(10**parsDic['log_rho_s'])
    qDM_model = np.ones(qDM_dat.shape)*parsDic['qDM']
    beta_model = np.array(parsDic['beta'])
    mbh_model = 10**parsDic['log_mbh']
    
    
    Jampy_Model.Updt_parameters(surf_DM=surf_DM_model, qobs_DM=qDM_model,
                                beta=beta_model, ml=np.array(parsDic['ml']),
                                inc=parsDic['inc'],mbh=10**parsDic['log_mbh'])

def JAM_log_likelihood(parsDic):
    
    Updt_JAM(parsDic)
    
    rmsModel, ml, chi2, chi2T = Jampy_Model.run()
    #print('Jampy', -0.5 * chi2T )
    return -0.5 * chi2T
    
    
    
    
def Pyautolens_log_likelihood(parsDic):
    
    Updt_Pyautolens(parsDic)
    
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
    
    rectangular = al.pix.Rectangular(shape=(50, 50))
    mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)
    
    inversion = al.Inversion(
        masked_dataset=masked_imaging,
        mapper=mapper,
        regularization=al.reg.Constant(coefficient=1),
    )
    chi2T = inversion.chi_squared_map.sum()
    #print('Autolens',-0.5 * chi2T )
    return -0.5 * chi2T

def log_probability(pars):
    (m1, m2, m3, m4, m5, m6, b1, b2, b3, b4, b5, b6, b7,
         inc, qDM, log_rho_s, log_mbh, mag_shear, phi_shear, gamma) = pars
    
    ml =  np.array([m1, m1, m2, m3, m4, m4, m6])
    beta =  np.array([b1, b2, b3, b4, b5, b6, b7])
    parsDic = {'ml': ml, 'inc': inc, 'qDM': qDM, 'log_rho_s': log_rho_s, 'log_mbh': log_mbh,
                  'mag_shear': mag_shear, 'phi_shear': phi_shear, 'gamma': gamma, 'beta': beta}
    
    #Checking boundaries
    if not np.isfinite(check_boundary(parsDic)):
        return -np.inf
    #calculating the log_priors
    lp = log_prior(parsDic)
    
    return lp + Pyautolens_log_likelihood(parsDic) + JAM_log_likelihood(parsDic) 
    



np.savetxt('Output_LogFile.txt', np.column_stack([0, 0, 0]),
                            fmt=b'	%i	 %e			 %e	 ', 
                            header="Output table for the combined model: Lens + Dynamic.\n Iteration	 Mean acceptance fraction	 Processing Time")

np.savetxt("LastFit.txt", np.column_stack([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]),
                            fmt=b'%e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	', 
                            header="Iteration	 ML1/2	 ML3	 ML4	 ML5	 ML6	 ML7	 b1	 b2	 b3	 b4	 b5	 b6	 b7	 Inc	 qDM	 Logrho_s	 LogMBH	 MagShear	 PhiShear	 gamma")
    



with MPIPool() as pool:
    
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

        ##### For the initial guesses we will use the Collett's best fit with a gaussian ball error around it
        ## variables tagged with <name>_std are the standard deviation of the parameter <name>.
    np.random.seed(42)   
    #Defining initial guesses

    ml = np.array([9.5,8.5,3.8,3.4,3.2,2.8])
    ml_std = np.ones(ml.shape)*np.array(prior['ml'][1])

    beta = np.array([-0.6, -1.0, 0.34, -3.4, 0.39, -0.31, 0.36])
    beta_std = np.ones(beta.shape)*np.array(prior['beta'][1])

    inc = np.array([90])
    inc_std = np.ones(inc.shape)*float(prior['inc'][1])

    qDM = np.array([0.74])
    qDM_std = np.ones(qDM.shape)*float(prior['qDM'][1])

    log_rho_s = np.array([2])
    log_rho_s_std = np.ones(log_rho_s.shape)*float(prior['log_rho_s'][1])

    log_mbh = np.array([10])
    log_mbh_std = np.ones(log_mbh.shape)*float(prior['log_mbh'][1])

    mag_shear = np.array([0.2])
    mag_shear_std = np.ones(mag_shear.shape)*float(prior['mag_shear'][1])

    phi_shear = np.array([0.1])
    phi_shear_std = np.ones(phi_shear.shape)*float(prior['phi_shear'][1])

    gamma = np.array([1])
    gamma_std = np.ones(gamma.shape)*float(prior['gamma'][1])

    ##Here we append all the variables and stds in a single array.
    p0 = np.append(ml, beta)
    p0 = np.append(p0,[inc, qDM, log_rho_s, log_mbh, mag_shear, phi_shear, gamma])

    p0_std = np.append(ml_std, beta_std)
    p0_std = np.append(p0_std, [inc_std, qDM_std, log_rho_s_std, log_mbh_std, mag_shear_std, phi_shear_std, gamma_std])

    #Finally we initialize the walkers with a gaussian ball around the best Collet's fit.
    nwalkers = 200                                                  #Number of walkers
    pos = emcee.utils.sample_ball(p0, p0_std, nwalkers)             #Initial position of all walkers


    nwalkers, ndim = pos.shape

    

    print("Workers nesse job:", pool.workers)
    print("Início")

        # Set up the backend
        # Don't forget to clear it in case the file already exists
    filename = "save.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
    
     # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend, a=1)
    
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

        #Update table output with last best fit
        last_fit_table = np.loadtxt("LastFit.txt")
        flat_samples = sampler.get_chain()
        values = []
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            values.append(mcmc[1])

        values = np.array(values)
        upt = np.append(iteration, values)

        np.savetxt("LastFit.txt", np.vstack([last_fit_table, upt]),
                            fmt=b'%e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	 %e	', 
                            header="Iteration	 ML1/2	 ML3	 ML4	 ML5	 ML6	 ML7	 b1	 b2	 b3	 b4	 b5	 b6	 b7	 Inc	 qDM	 Logrho_s	 LogMBH	 MagShear	 PhiShear	 gamma")
 

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
    
    

