
import numpy as np
import autolens as al
import autolens.plot as aplt

###boundaries. [lower, upper]
boundary = {'inc': [50, 140], 'beta': [-5, 5], 'ml': [0.5, 15], 'log_mbh':[7, 11],
                     'mag_shear': [0, 0.1], 'phi_shear': [0, 180], 'gamma': [0, 2]}


# parameter gaussian priors. [mean, sigma]
prior = {'gamma': [1.0, 0.5] }


class Model(object):

    def __init__(self, z_lens, z_source,
                    surf_star_dat, sigma_star_dat_ARC, sigma_star_dat_PC, qobs_star_dat, Lum_star_dat,
                    sigmaBH_ARC, sigmaBH_PC, qSMBH, Total_sigma_RAD,
                    Jampy_model, masked_imaging, mass_profile,  
                    ):

        self.z_lens = z_lens
        self.z_source = z_source
        self.surf_star_dat = surf_star_dat
        self.sigma_star_dat_ARC = sigma_star_dat_ARC
        self.sigma_star_dat_PC = sigma_star_dat_PC
        self.qobs_star_dat = qobs_star_dat
        self.Lum_star_dat = Lum_star_dat

        self.sigmaBH_ARC = sigmaBH_ARC
        self.sigmaBH_PC = sigmaBH_PC
        self.qSMBH = qSMBH
        self.Total_sigma_RAD = Total_sigma_RAD

        self.Jampy_model = Jampy_model
        self.masked_imaging = masked_imaging
        self.mass_profile = mass_profile




    def check_Deprojected_axial(self, parsDic):
        inc = np.radians(parsDic['inc'])
        #Stellar
        qintr_star = self.qobs_star_dat**2 - np.cos(inc)**2
        if np.any(qintr_star <= 0):
            return -np.inf
        
        qintr_star = np.sqrt(qintr_star)/np.sin(inc)
        if np.any(qintr_star <= 0.05):
            return -np.inf

        return 0.0

    
    def check_boundary(self, parsDic):
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
        if not np.isfinite(self.check_Deprojected_axial(parsDic)):
            return -np.inf

        for keys in parsDic:
            if boundary[keys][0] < parsDic[keys] < boundary[keys][1]:
                pass
            else:
                return -np.inf
        return 0.0

    def log_prior(self, parsDic):
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



    def Updt_Pyautolens(self,parsDic):
        '''
        Update the Lens mass model
        input
        parsDic: parameter dictionary {'paraName', value}
        '''
        #Inclination
        inc_model = np.deg2rad(parsDic['inc'])                     #Get new inclination in radians
        
        #Stellar parameters
        ml_model = parsDic['ml'] #New Gaussian Mass-to-light ratio [M_sun/L_sun]
        
        Stellar_Mass_model = self.Lum_star_dat*ml_model         #Updt the stellar mass 
            
        #Total mass and new projected axis here we add the new SMBH mass
        Total_Mass_model = np.concatenate((Stellar_Mass_model, 10**parsDic['log_mbh']), axis=None)  #New total mass
        
        Total_q_model = np.concatenate((self.qobs_star_dat, self.qSMBH), axis=None)  #New axial  ratio

        #Model Updt
        self.mass_profile.MGE_Updt_parameters(Total_Mass_model,self.Total_sigma_RAD,
                                                Total_q_model, parsDic['gamma']) #Update the model



    def Updt_JAM(self,parsDic):
        '''
        Update the dynamical mass model
        input
        parsDic: parameter dictionary {'paraName', value}
        '''
        beta_model = np.full_like(self.surf_star_dat, parsDic['beta'])    #anisotropy parameter
        mbh_model = 10**parsDic['log_mbh']                           #BH mass
        
        #mass-to-light update 
        ml_model = parsDic['ml']
        
        #Model Updt
        self.Jampy_model.upt(inc=parsDic['inc'],ml=ml_model, beta=beta_model, mbh=mbh_model)


    def JAM_log_likelihood(self,parsDic):
        """
            Perform JAM modeling and return the chi2
        """
        
        self.Updt_JAM(parsDic)               #Updt values for each iteration
        
        rmsModel, ml, chi2, chi2T = self.Jampy_model.run()
        return -0.5 * chi2T


    def Pyautolens_log_likelihood(self,parsDic):
        """
            Perform Pyautolens modeling and return the chi2
        """
        
        self.Updt_Pyautolens(parsDic)        #Updt values for each iteration
        shear_comp = al.convert.shear_elliptical_comps_from(magnitude=parsDic['mag_shear'], phi=parsDic['phi_shear']
        ) #external shear
        #New lens model
        lens_galaxy = al.Galaxy(                                            
            redshift=self.z_lens,
            mass=self.mass_profile,
            shear=al.mp.ExternalShear(elliptical_comps=shear_comp),
        )
        
        
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=self.z_source)])
        source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=self.masked_imaging.grid)[1]
        
        #Check if the model has converged. If not, return -inf
        try:
            rectangular = al.pix.Rectangular(shape=(40, 40))
            mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)
        
            inversion = al.Inversion(
                masked_dataset=self.masked_imaging,
                mapper=mapper,
                regularization=al.reg.Constant(coefficient=3.5),
        )
            chi2T = inversion.chi_squared_map.sum()
            #print(parsDic)

            #aplt.Inversion.subplot_inversion(inversion, include=aplt.Include(inversion_border=False,
            #                                                         inversion_pixelization_grid=False))
            return -0.5 * chi2T
        except:
            return -np.inf


    def log_probability(self,pars):
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
        if not np.isfinite(self.check_boundary(parsDic)):
            return -np.inf
        #calculating the log_priors
        lp = self.log_prior(parsDic)

        return lp + self.JAM_log_likelihood(parsDic) + self.Pyautolens_log_likelihood(parsDic)



    def __call__(self, pars):
        return self.log_probability(pars)