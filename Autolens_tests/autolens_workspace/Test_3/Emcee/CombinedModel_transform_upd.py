
"""
    Goal: 
        Make Dynamical modeling + Lensing modeling with MGE mass model easier. 
            Here we have defined the prior functions, parameters boundaries, and update for models. The principal ideia is make use this script with emcee.

        This script makes use of Pyautolens (https://pyautolens.readthedocs.io/en/latest/) and Jampy (https://pypi.org/project/jampy/).
        Beside that, My_Jampy.py class, which is a python class  warp for original Jampy, is required. 
"""


import numpy as np
import autolens as al
import autolens.plot as aplt

#boundaries. [lower, upper]
#Used during non-linear fit (Emcee).
boundary = {'inc': [50, 90], 'beta': [-3, 3], 'ml': [0.5, 15],  
                 'ml0': [0.5, 15], 'delta': [0.1, 2], 'lower': [0, 1],
                 'log_rho_s': [6, 12], 'qDM': [0.2, 1], 'log_mbh':[7, 11],
                 'mag_shear': [0, 0.1], 'phi_shear': [0, 179], 'gamma': [0.8, 1.2]}


#Gaussian priors. [mean, sigma]
"""
    For now, only gaussian prior is for gamma, because we have a prior information about it. All other parameters have flat (or non-informative) priors.
"""
prior = {'gamma': [1.0, 0.04] }

#Transformation for proposal values
"""
    With this set of transformation we do two things:
        - First we transform any proposal step, which can falls in the range (-inf, +inf), to a new range (0,1). This could be made by many continuos functions that maps R -> (0,1). We choose the fastest function (1 + x / (1 + abs(x))) * 0.5.

        - Second we want to transform this value in the range (0,1), to a value inside the boundaries of our problem. This is made by a linear transformation, which presers the distance between two positions.   New_value = (((pars[keys] - 0) * NewRange) / OldRange) + boundary[keys][0]

"""

def absolut_transform(x):
    """
        Map R -> (0,1).
        Input:
        ---------------
        x: Array or scalar
            Real number

        Output:
        ---------------
        x_transformed: Array or scalar
            Transformed value between (0,1)
    """
    return (1 + x / (1 + abs(x))) * 0.5
def linear_transform(pars):
    """
        Map (0,1) -> (boundaryMin, boundaryMax), preserving the ratio.
        Input:
        --------------
            pars: list
                List with real number between (0,1). The elements should be elements of _boundary_ function. 
        Output:
        --------------
            pars_transformed: list
                Transformed values that matches boundaries limits.
    """
    for keys in pars:
        OldRange = 1 - 0    #All transformations tested returns values between [0,1]. OlderMax-OlderMin
        
        if OldRange == 0:  #This part are  unnecessary, once our OldRange is always 1.
            pars[keys] = boundary[keys][0]
        else:
            NewRange   = boundary[keys][1] - boundary[keys][0]
            pars[keys] = (((pars[keys] - 0) * NewRange) / OldRange) + boundary[keys][0]
    
    return pars

#Dark matter component
"""
   You must call def has_DM function always. If a is set False, there are no inclusion of DM. If a is set True, you should give the name of table containing the MGE parametrization. This table should follow the following organization:

    Frist column : mass surface density for each gaussian [M_sun/pc^2];
    Second column: sigma of each gaussian                 [arcsec];
    Third column : axial ratio for each gaussian          [ad]

    If you want to include DM component in your non-linear search, the parametrization of dark matter profile should assume a central density (sometimes called called \rho_s) equal to one. 
    Until this time, only  pseudo-NFW  mass density profile  are accepted (eq. S1 of 10.1126/science.aao2469). This profile leeds two free parameters: central density (\rho_s) and the axial ratio (qDM).
"""


#Gaussian ML function
def gaussian_ml(sigma, delta, ml0=1.0, lower=0.4):
    """
    Create a M*L gradient
    Input:
    -----------
        sigma: Gaussian sigma                           [arcsec]
        delta: Gradient value
        ml0: Central stellar mass to light ratio        [M_sun/L_sun]    
        lower: the ratio between the central and the outer most M/L
    Output:
    ----------
        ML: gaussian mass to light ratio. One component per gaussian in surf_lum.
    """

    sigma = np.atleast_1d(sigma)
    sigma = sigma - sigma[0]
    ML = ml0 * (lower + (1-lower)*np.exp(-0.5 * (sigma * delta)**2))
    
    return ML

class Models(object):

    def __init__(self, 
                    Jampy_model, mass_profile, masked_imaging, quiet=True
                    ):
        """
        Input:
        --------------------------
        Jampy_model: My_Jampy object
            Object containing jampy model, builded with My_Jampy.JAM class. Check it for more informations.
        mass_profile: Pyautolens object
            MGE mass profile for Pyautolens modelling. You can find more information in total_mass_profiles.py, the script were we implement MGE mass model.
        masked_imaging: Pyautolens object
            Pyautolens object containing all the information about the image that we want model. You can find more information in Pyautolens documentation, linked in the header of this document.
        quiet: Boolean
            If False, allow the some plots of Jampy and Pyautolens model.
        """

        self.Jampy_model    = Jampy_model
        self.mass_profile   = mass_profile
        self.masked_imaging = masked_imaging
        self.quiet          = quiet



    def check_ML(self, parsDic):
        """
        Check ML.
        If ML is scalar, check only the boundaries.
        If ML is gradient, check boundaries and gradient.
        If ML is gaussian, check boundaries of its parameters.
        """

        if self.ml_kind == 'scalar':
            if boundary['ml'][0] <= parsDic['ml'] <= boundary['ml'][1]:
                pass
            else:
                return -np.inf
        
        elif self.ml_kind == 'gradient':
            #Boundaries
            for i in range(len(parsDic['ml'])):
                if boundary['ml'][0] <= parsDic['ml'][i] <= boundary['ml'][1] :
                    pass
                else:
                    return -np.inf
            #Gradient
            for i in range( len(parsDic['ml']) - 1):
                if parsDic['ml'][i] >= parsDic['ml'][i+1]:
                    pass
                else:
                    return -np.inf

        elif self.ml_kind == 'gaussian':
            if boundary['ml0'][0] <= parsDic['ml0'] <= boundary['ml0'][1]:
                pass
            else:
                return -np.inf
            
            if boundary['delta'][0] <= parsDic['delta'] <= boundary['delta'][1]:
                pass
            else:
                return -np.inf

            if boundary['lower'][0] <= parsDic['lower'] <= boundary['lower'][1]:
                pass
            else:
                return -np.inf           
        
        return 0.0

    def check_beta(self, beta):
        """
        Check if beta is ok.
        Avoid beta = 1, because this could cause problems.
        Finally check if beta is inside boundaries.
        """

        if self.beta_kind == 'scalar':
            if beta == 1:
                return -np.inf
            else:
                pass

            #Now check if there is inside the boundaries
            if boundary['beta'][0] <= beta <= boundary['beta'][1]:
                pass
            else:
                return -np.inf


        elif self.beta_kind == 'vector':
            if any(beta == 1):
                return -np.inf
            else:
                pass

            #Now check if there is inside the boundaries
            for i in range(len(beta)):
                if boundary['beta'][0] <= beta[i] <= boundary['beta'][1] :
                    pass
                else:
                    return -np.inf
        return 0.0

    def check_Deprojected_axial(self, parsDic):
        """
            Check deprojected axial ratio. If the inclination is too low for decomposition, then rejected it.
            Input:
            ----------
                parsDic: parameter dictionary {'paraName', value}
            Output:
            ----------
                -np.inf or 0.0
        """
        inc = np.radians(parsDic['inc'])
        qobs_star_dat = self.mass_profile.qobs_lum
        #Stellar
        qintr_star = qobs_star_dat**2 - np.cos(inc)**2
        if np.any(qintr_star <= 0):
            return -np.inf
        
        qintr_star = np.sqrt(qintr_star)/np.sin(inc)
        if np.any(qintr_star <= 0.05):
            return -np.inf


        #DM
        #If dark matter axial ratio is a parameter, check its deprojection too. Otherwise, ignore it.
        try:
            qintr_DM = parsDic['qDM']**2 - np.cos(inc)**2
            if qintr_DM <= 0:
                return -np.inf
            
            qintr_DM = np.sqrt(qintr_DM)/np.sin(inc)
            if qintr_DM <= 0.05:
                return -np.inf
        except:
            pass


        return 0.0
    
    def check_boundary(self, parsDic):
        """
            Check whether parameters are within the boundary limits
            Input:
            ----------
                parsDic: parameter dictionary {'paraName', value}
            Output:
            ----------
                -np.inf or 0.0
        """   
        keys = set(parsDic.keys())   #All parameters
        
        #Check beta
        if not np.isfinite(self.check_beta(parsDic['beta'])):
            return -np.inf
        #If it is ok, exclude its keyword
        excludes = set(['beta'])  #Exclude beta, because we already verify above

        #Check ml (scalar, gradient, gaussian).
        if not np.isfinite(self.check_ML(parsDic)):
            return -np.inf
        #If it is ok, exclude its keywords
        excludes.add('ml')
        excludes.add('ml0')
        excludes.add('delta')
        excludes.add('lower')
            


        #Check if deprojected axial ratio is ok  (q' <=0 or q' <= 0.05) for the dynamical model.
        if not np.isfinite(self.check_Deprojected_axial(parsDic)):
            return -np.inf

        for keys in keys.difference(excludes):
            if boundary[keys][0] <= parsDic[keys] <= boundary[keys][1]:
                pass
            else:
                return -np.inf
        return 0.0

    def log_prior(self, parsDic):
        '''
        Input:
        ----------
            parsDic: parameter dictionary {'paraName', value}
        Output:
        ----------
            log_prior
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

    def has_DM(self, a=False, filename=None):
        """
        Includes dark matter component in your model or not.
        Input:
        -----------------
        filename: str
            Name of a table with following format.

            Frist column : mass surface density for each gaussian [M_sun/pc^2];
            Second column: sigma of each gaussian                 [arcsec];
            Third column : axial ratio for each gaussian          [ad]
        """
        if a is True:
        	surf_DM_dat, sigma_DM_dat, qobs_DM_dat = np.loadtxt(filename, unpack=True)

        	self.surf_DM_dat  = surf_DM_dat
        	self.sigma_DM_dat = sigma_DM_dat
        	self.qobs_DM_dat  = qobs_DM_dat
        	self.has_dm       = True
        elif a is False:
        	self.has_dm       = False 
    
    def mass_to_light(self, ml_kind='scalar'):
        """
            Sets the kind of mass to light ratio in non-linear search. Default is scalar.
            Input:
            ------------

            kind: Select the type of ML ratio. Three possibilities are available.
                    scalar  : Constante ML. Only one free parameter

                    gradient: One ML per gaussian following a gradient profile, i.e, ML[i+1] >= ML[i], where i=0 represents the most internal gaussian. This implies a number of free parameters equal to number of gaussians in surface luminosity density. Take care with it!

                    gaussian: A gaussian ML. Here there are three free parameters: central ML0, sigma   of the gaussian and a lower value for the mass to light ratio. This allows a ML per MGE component, following a gaussian shape, without increasing (too much) the number of free parameters.
        """
        if ml_kind == 'scalar' or ml_kind == 'gradient' or ml_kind == 'gaussian':
            self.ml_kind = ml_kind
        else:
            print("Invalid ML !!!")
            print("Setting scalar, to avoid future problems.")
            self.ml_kind = 'scalar'
    
    def beta(self, beta_kind='scalar'):
        """
        Sets the kind of anisotropy in non-linear search. Default is scalar.
            Input:
            ------------
            kind: Select the type of beta. Two possibilities are available.
                    scalar  : Constante anisotropy. Only one free parameter

                    vector  : One beta parameter per gaussian luminosity component. Be careful if your decomposition has many gaussians.         
                
        """
        if beta_kind == 'scalar' or beta_kind == 'vector':
            self.beta_kind = beta_kind
        else:
            print("Invalid anisotropy !!!")
            print("Setting scalar, to avoid future problems.")
            self.beta_kind = 'scalar'

    def Updt_ML(self, parsDic):
        """
        Update the mass to light ratio based on the kind choose previously.
        Input:
        ---------
            parsDic: parameter dictionary {'paraName', value}

        Output:
        ---------
            Return a ML.                [M_sun/L_sun]
        """

        if self.ml_kind == 'scalar':
            return parsDic['ml']
        
        elif self.ml_kind == 'gradient':
            return parsDic['ml']
        
        elif self.ml_kind == 'gaussian':
            return gaussian_ml(self.mass_profile.sigma_lum, parsDic['delta'],
                                    parsDic['ml0'], parsDic['lower'])

    def Updt_beta(self, parsDic):
        """
        Update anisotropy based on the kind choose previously.
        Input:
        ---------
            parsDic: parameter dictionary {'paraName', value}

        Output:
        ---------
            Return a ML.                [M_sun/L_sun]
        """

        if self.beta_kind == 'scalar':
            return np.full_like(self.mass_profile.surf_lum, parsDic['beta'])
        elif self.beta_kind == 'vector':
            assert parsDic['beta'].size == self.mass_profile.surf_lum.size, "Number of betas doesn't match number of luminosity gaussians."
            return parsDic['beta']
    
    def Updt_Pyautolens(self,parsDic):
        """
        Update the Lens mass model
        Input:
        ----------
            parsDic: parameter dictionary {'paraName', value}
        """

        ml_model    = self.Updt_ML(parsDic)
        mbh_model   = 10**parsDic['log_mbh']
        gamma_model = parsDic['gamma']

        if self.has_dm is True:
            surf_dm_model = (10**parsDic['log_rho_s']) * self.surf_DM_dat
            qDM_model     = np.ones(self.qobs_DM_dat.shape)*parsDic['qDM']
            
            self.mass_profile.MGE_Updt_parameters(ml=ml_model, mbh=mbh_model, gamma=gamma_model, 
                                                surf_dm=surf_dm_model, qobs_dm=qDM_model)
        else:
            self.mass_profile.MGE_Updt_parameters(ml=ml_model, mbh=mbh_model, gamma=gamma_model)

    def Updt_JAM(self,parsDic):
        """
        Update the dynamical mass model
        Input:
        ----------
            parsDic: parameter dictionary {'paraName', value}
        """

        beta_model  = self.Updt_beta(parsDic)
        ml_model    = self.Updt_ML(parsDic)
        mbh_model   = 10**parsDic['log_mbh']
        gamma_model = parsDic['gamma']
        inc_model   = parsDic['inc']

        if self.has_dm is True:
            surf_dm_model = (10**parsDic['log_rho_s']) * self.surf_DM_dat
            qDM_model     = np.ones(self.qobs_DM_dat.shape)*parsDic['qDM']
            
            self.Jampy_model.upt(surf_dm=surf_dm_model, qobs_dm=qDM_model, inc=inc_model,
                     ml=ml_model, beta=beta_model, mbh=mbh_model)
        else:
            self.Jampy_model.upt(inc=inc_model, ml=ml_model, beta=beta_model, mbh=mbh_model)

    def JAM_log_likelihood(self,parsDic):
        """
        Perform JAM modeling and return the chi2
        """
        
        self.Updt_JAM(parsDic)               #Updt values for each iteration
        if self.quiet is False:
            rmsModel, ml, chi2, chi2T = self.Jampy_model.run(plot=True, quiet=False)
        else:
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
            redshift=self.mass_profile.z_l,
            mass=self.mass_profile,
            shear=al.mp.ExternalShear(elliptical_comps=shear_comp),
        )
        
        
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=self.mass_profile.z_s)])

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
            
            if self.quiet is False:
                aplt.Inversion.subplot_inversion(inversion, 
                                                    include=aplt.Include(inversion_border=False,
                                                    inversion_pixelization_grid=False))
            return -0.5 * chi2T
        except:
            return -np.inf

    def Dic(self, pars):
        """
        Build a parameter dictionary based on what inputs are set.
        Input:
        -----------
            Parameters of emcee.
        Output:
        -----------
            Dictionary with parameters for emcee.
        """
        if self.has_dm is True:

            if self.ml_kind == 'scalar':
                if self.beta_kind == 'scalar':
                    (ml, beta, inc, log_mbh, log_rho_s, qDM, mag_shear, phi_shear, gamma) = pars
                    
                    parsDic = {'ml': ml, 'beta': beta, 'inc': inc, 'log_mbh': log_mbh,
                                'log_rho_s':log_rho_s, 'qDM':qDM, 'mag_shear':mag_shear,
                                'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml   = pars[0]
                    beta = pars[1:size+1]
                    (inc, log_mbh, log_rho_s,
                         qDM, mag_shear, phi_shear, gamma) = pars[beta.size+1:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'inc': inc, 'log_mbh': log_mbh,
                                'log_rho_s':log_rho_s, 'qDM':qDM, 'mag_shear':mag_shear,
                                'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

            elif self.ml_kind == 'gradient':
                if self.beta_kind == 'scalar':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml = pars[0:size]
                    (beta, inc, log_mbh, log_rho_s, qDM,
                         mag_shear, phi_shear, gamma) = pars[ml.size:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'inc': inc, 'log_mbh': log_mbh,
                                'log_rho_s':log_rho_s, 'qDM':qDM, 'mag_shear':mag_shear,
                                'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml   = pars[0:size]
                    beta = pars[ml.size:ml.size + size]
                    (inc, log_mbh, log_rho_s,
                         qDM, mag_shear, phi_shear, gamma) = pars[ml.size + beta.size:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'inc': inc, 'log_mbh': log_mbh,
                                'log_rho_s':log_rho_s, 'qDM':qDM, 'mag_shear':mag_shear,
                                'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

            elif self.ml_kind == 'gaussian':
                if self.beta_kind == 'scalar':
                    (ml0, delta, lower, beta, inc, log_mbh, log_rho_s, qDM,
                         mag_shear, phi_shear, gamma) = pars
                    
                    parsDic = {'ml0': ml0, 'delta':delta, 'lower':lower, 'beta': beta,
                                 'inc': inc, 'log_mbh': log_mbh, 'log_rho_s':log_rho_s, 'qDM':qDM,
                                 'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml0   = pars[0]
                    delta = pars[1]
                    lower = pars[2]
                    beta  = pars[3:size+3]

                    (inc, log_mbh, log_rho_s,
                         qDM, mag_shear, phi_shear, gamma) = pars[3 + beta.size:]
                    
                    parsDic = {'ml0': ml0, 'delta':delta, 'lower':lower, 'beta': beta,
                                 'inc': inc, 'log_mbh': log_mbh, 'log_rho_s':log_rho_s, 'qDM':qDM,
                                 'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

        elif self.has_dm is False:
            if self.ml_kind == 'scalar':
                if self.beta_kind == 'scalar':
                    (ml, beta, inc, log_mbh, mag_shear, phi_shear, gamma) = pars
                    
                    parsDic = {'ml': ml, 'beta': beta, 'inc': inc, 'log_mbh': log_mbh,
                                'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml   = pars[0]
                    beta = pars[1:size+1]
                    (inc, log_mbh, mag_shear, phi_shear, gamma) = pars[beta.size+1:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'inc': inc, 'log_mbh': log_mbh,
                                'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

            elif self.ml_kind == 'gradient':
                if self.beta_kind == 'scalar':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml = pars[0:size]
                    (beta, inc, log_mbh, mag_shear, phi_shear, gamma) = pars[ml.size:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'inc': inc, 'log_mbh': log_mbh,
                                'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml   = pars[0:size]
                    beta = pars[ml.size:ml.size + size]
                    (inc, log_mbh, mag_shear, phi_shear, gamma) = pars[ml.size + beta.size:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'inc': inc, 'log_mbh': log_mbh,
                                'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

            elif self.ml_kind == 'gaussian':
                if self.beta_kind == 'scalar':
                    (ml0, delta, lower, beta, inc, log_mbh,
                         mag_shear, phi_shear, gamma) = pars
                    
                    parsDic = {'ml0': ml0, 'delta':delta, 'lower':lower, 'beta': beta,
                                 'inc': inc, 'log_mbh': log_mbh, 'mag_shear':mag_shear,
                                 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml0   = pars[0]
                    delta = pars[1]
                    lower = pars[2]
                    beta  = pars[3:size+3]

                    (inc, log_mbh, mag_shear, phi_shear, gamma) = pars[3 + beta.size:]
                    
                    parsDic = {'ml0': ml0, 'delta':delta, 'lower':lower, 'beta': beta,
                                 'inc': inc, 'log_mbh': log_mbh, 'mag_shear':mag_shear,
                                 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

    def log_probability(self,pars):
        """
        Log-probability function for whole model WITH dark matter.
        Input:
        ----------
            pars: current values in the Emcee sample.
        Output:
        ---------
            log probability for the combined model.
        """
        pars = absolut_transform(pars)        #Transform to (0,1)
        parsDic = self.Dic(pars)              #Create a dic with values
        parsDic = linear_transform(parsDic)   #Transform (0,1) to matches boundaries

        if self.quiet is False:
            print("ParsDic", parsDic)
        #Checking boundaries
        if not np.isfinite(self.check_boundary(parsDic)):
            return -np.inf
        #calculating the log_priors
        lp = self.log_prior(parsDic)

        return lp + self.JAM_log_likelihood(parsDic) + self.Pyautolens_log_likelihood(parsDic)


    def __call__(self, pars):
        return self.log_probability(pars)
