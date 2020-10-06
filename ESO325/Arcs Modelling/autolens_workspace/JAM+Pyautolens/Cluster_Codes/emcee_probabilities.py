#Author: Carlos Roberto de Melo
#Date: 10/05/2020
#Obj: 
'''
    In this file we found all probabilities associated to combine (Jampy+Auto Lens) model. To know:
    log priors of each parameters, log log likelihood of each model, and finaly the total log probability.
'''
import numpy as np
from model_data import Global_Parameters as GP
from model_data import Autolens_data as AL_data 
import autolens as al


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



class Probability():
    """
        This class receive the models (Jampy and AutoLens), and compute the log_probability for each iteration of Emcee. For that, each new parameter combination is tested and the models build if the parameters  comply with the requirements.
    """

    def __init__(self, Jampy_Model, mass_profile, masked_imaging):
        """
        Parameters
        ----------
            Jampy_model: is the initialized dynamical model.
            mass_profile: is the initialized MGE mass profile, destined to build the lens model.
            masked_imaging: is the image of the arcs with the mask already applied. Comes from autolens.
        """

        #Making all variables global variables

        super(Probability, self).__init__(
        )

        self.Jampy_Model = Jampy_Model
        self.mass_profile = mass_profile
        self.masked_imaging = masked_imaging





    def check_ML_Grad(self, ml):
        
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

    def check_boundary(self, parsDic):
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
        qintr_star = GP.qstar_dat**2 - np.cos(inc)**2
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
        if not np.isfinite(self.check_ML_Grad(parsDic['ml'])):
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
        


    def log_prior(self, parsDic):
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
            


    def Updt_Pyautolens(self, parsDic):
        '''
        Update the Lens mass model
        input
        parsDic: parameter dictionary {'paraName', value}
        '''
        #Inclination
        inc_model = np.deg2rad(parsDic['inc'])                          #Get new inclination in radians
        
        #Stellar parameters
        Stellar_Mass_model = (AL_data.Lum_star_dat*parsDic['ml']).value         #Updt the stellar mass 
        
        #DM parameters
        qDM_model = np.ones(GP.qDM_dat.shape)*parsDic['qDM']                                              #Updt DM axial ratio 
        Mass_DM_model = (10**parsDic['log_rho_s'])*(2*np.pi*GP.surf_DM_dat*(GP.sigma_DM_dat_PC**2)*qDM_model).value    #Updt DM Mass
        
        
        #Total mass and new projected axis here we add the new SMBH mass
        Total_Mass_model = np.concatenate((Stellar_Mass_model, Mass_DM_model, 10**parsDic['log_mbh']), axis=None)  #New total mass
        Total_q_model = np.concatenate((GP.qstar_dat, qDM_model, AL_data.qSMBH), axis=None)                           
        Total_q_proj_model = (np.sqrt(Total_q_model**2 - np.cos(inc_model)**2)/np.sin(inc_model))          #New projected axial ratio
        self.mass_profile.MGE_Updt_parameters(Total_Mass_model,AL_data.Total_sigma_RAD.value, Total_q_proj_model, parsDic['gamma'])       #Update the model
        
        
    def Updt_JAM(self, parsDic):
        '''
        Update the dynamical mass model
        input
        parsDic: parameter dictionary {'paraName', value}
        '''
        surf_DM_model = GP.surf_DM_dat.value*(10**parsDic['log_rho_s'])
        qDM_model = np.ones(GP.qDM_dat.shape)*parsDic['qDM']
        beta_model = np.array(parsDic['beta'])
        mbh_model = 10**parsDic['log_mbh']
        
        
        self.Jampy_Model.Updt_parameters(surf_DM=surf_DM_model, qobs_DM=qDM_model,
                                    beta=beta_model, ml=np.array(parsDic['ml']),
                                    inc=parsDic['inc'],mbh=10**parsDic['log_mbh'])

    def JAM_log_likelihood(self, parsDic):
        
        self.Updt_JAM(parsDic)
        
        rmsModel, ml, chi2, chi2T = self.Jampy_Model.run()
        #print('Jampy', -0.5 * chi2T )
        return -0.5 * chi2T
        
        
        
        
    def Pyautolens_log_likelihood(self, parsDic):
        
        self.Updt_Pyautolens(parsDic)
        
        #New lens model
        lens_galaxy = al.Galaxy(                                            
            redshift=GP.z_lens,
            mass=self.mass_profile,
            shear=al.mp.ExternalShear(elliptical_comps=(parsDic['mag_shear'], parsDic['phi_shear'])),
        )
        
        
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=GP.z_source)])
        source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=self.masked_imaging.grid)[1]
        
        #check if the integral converge. If not, return -np.inf
        if np.isnan(source_plane_grid[0,0]):
            return -np.inf
        
        rectangular = al.pix.Rectangular(shape=(50, 50))
        mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)
        
        inversion = al.Inversion(
            masked_dataset=self.masked_imaging,
            mapper=mapper,
            regularization=al.reg.Constant(coefficient=1),
        )
        chi2T = inversion.chi_squared_map.sum()
        #print('Autolens',-0.5 * chi2T )
        return -0.5 * chi2T


    def log_probability(self, pars):
        (m1, m2, m3, m4, m5, m6, b1, b2, b3, b4, b5, b6, b7,
            inc, qDM, log_rho_s, log_mbh, mag_shear, phi_shear, gamma) = pars
        
        ml =  np.array([m1, m1, m2, m3, m4, m4, m6])
        beta =  np.array([b1, b2, b3, b4, b5, b6, b7])
        parsDic = {'ml': ml, 'inc': inc, 'qDM': qDM, 'log_rho_s': log_rho_s, 'log_mbh': log_mbh,
                    'mag_shear': mag_shear, 'phi_shear': phi_shear, 'gamma': gamma, 'beta': beta}
    
        #Checking boundaries
        if not np.isfinite(self.check_boundary(parsDic)):
            return -np.inf
        #calculating the log_priors
        lp = self.log_prior(parsDic)
        
        return lp + self.Pyautolens_log_likelihood(parsDic) + self.JAM_log_likelihood(parsDic) 
