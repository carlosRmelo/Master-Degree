
"""
    Goal: 
        Make Dynamical modeling + Lensing modeling with MGE mass model easier. 
            Here we have defined the parameters boundaries, and update for models. The principal ideia is make use this script with dynesty (https://dynesty.readthedocs.io/en/latest/).

        This script makes use of Pyautolens (https://pyautolens.readthedocs.io/en/latest/) and Jampy (https://pypi.org/project/jampy/).
        Beside that, My_Jampy.py class, which is a python class  warp for original Jampy, is required. 
"""


import numpy as np
import autolens as al
import autolens.plot as aplt
from autoarray.fit import fit
import copy



#Dark matter component
"""
  
   There are two ways to set a DM component.
    
    One of them is using a MGE decomposition of a DM mass profile, that could be add in lens model only, dynamical model only or both at same time.

    And the other is add a analytical (it means non-MGE model) DM profile for the lens deflection angle. In this case, the DM profile is add only in the lens model. You should take a *lot* of care using a DM compoent only in the lens model, because this probably makes your complet model non self consistent. This option is recommended when your DM profile has a MGE decomposition (that will be add in the dynamical model) AND has a analitcal (or a faster calculation) deflection angle.
    
    Until now, only Pyautolens  deflection angles could be add in this way. And more, only the pseudo analytical NFW is implemented correctly.


    In any case above, You must call def has_MGE_DM function. Check it for more details. 
    If you want to include an analytical DM model, you should call: def include_DM_analytical function. Also check it for mode details.

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

        self.boundaries()

        #Defining new boundaries for inclination, to avoid too low inclination angles.
        self.set_boundary("qinc", [0.0501, np.min(self.mass_profile.qobs_lum)])

    def boundaries(self):
        """
            Define the boundaries for parameters.
            boundary_name: [lower, upper]
        """

        self.boundary = {'qinc': [0.0, 1], 'beta': [-3, 3], 'ml': [0.5, 15],  
                 'ml0': [0.5, 15], 'delta': [0.1, 2], 'lower': [0, 1],
                 'kappa_s': [0, 1], 'qDM': [0.1, 1], 'log_mbh':[7, 11],
                 'mag_shear': [0, 0.1], 'phi_shear': [0, 179], 'gamma': [0.80, 1.20]}


    def set_boundary(self, key, value):
        """
        Reset the parameter boundary value
        Input:
        ----------------------------------------------------
        
        key: String
            Parameter name.
        value: Length two list
            Boundary values.
        """
        if key not in self.boundary.keys():
            raise ValueError('parameter name \'{}\' not correct'.format(key))
        if len(value) != 2:
            raise ValueError('Boundary limits must be a length 2 list')
        print('Change {} limits to [{}, {}], defaults are [{}, {}]'
              .format(key, value[0], value[1], self.boundary[key][0],
                      self.boundary[key][1]))
        self.boundary[key] = value

    def print_boundary(self):
        """
        Print current boundary parameters.
        """
        print("Your currently boundaries are:")
        for key in self.boundary.keys():
            print(key, self.boundary[key])

    

    def has_MGE_DM(self, a=False, filename=None, include_MGE_DM='Both'):
        """
        Includes a MGE dark matter component in your model or not.
        This means that you had to parameterize your DM profile previously with MGE method. In this case, your DM model has two free parameters: 
            1. The intensity (kappa_s)
            2. The projected axial ratio qDM (assuming the semi-major axis along the x-axis).
        
        The MGE dark matter component could be add in the dynamical model only, in the lens model only, or either. For a more robust analysis, we recommend a self consist model, using the DM MGE component in both model at same time. But, unfortunally, this choice is a little bit slower due the form of the integral in the lens deflection calculation.
        Input:
        -----------------
        a: Bool
            If False, doesn't includes MGE DM component.
            If True, includes a DM MGE component. 

        filename: str
            Name of a table with following format.

            Frist column : mass surface density for each gaussian           [M_sun/pc^2];
            Second column: sigma of each gaussian                           [arcsec];
            Third column : projected axial ratio for each gaussian          [ad]
        
        include_MGE_DM: str
            Three option where the MGE DM model have to be included. Default: Both
            Both:      Lens and Dynamical model
            Dynamical: Only in the Dynamical model
            Lens:      Only in the Lens model
        """
        if a is True:
            surf_DM_dat, sigma_DM_dat, qobs_DM_dat = np.loadtxt(filename, unpack=True)

            self.surf_DM_dat  = surf_DM_dat
            self.sigma_DM_dat = sigma_DM_dat
            self.qobs_DM_dat  = qobs_DM_dat
            self.has_dm       = True
            
            if include_MGE_DM == "Both":
                self.include_MGE_DM = "Both"

            elif include_MGE_DM == "Dynamical":
                self.include_MGE_DM = "Dynamical"

            elif include_MGE_DM == "Lens":
                self.include_MGE_DM = "Lens"
            
            else:
                raise ValueError("You are trying to set MGE DM component in a non existing model, take care.")


        elif a is False:
        	self.has_dm       = False 

    def include_DM_analytical(self, analytical_DM):
        """
        Includes an analytical DM profile only in the lens model. This word *analytical* means that the DM profile was not parametrized by MGE, not that the deflection angle are analytical.
        Only Pyautolens models are allowed.
        Beside that, only Elliptical NFW profile, with phi=0.0, are implemented.

        Input:
        ------------------------------------------------------
        analytical_DM: Pyautolens class
            Lens model of Pyautolens.
        """
        self.analytical_DM = analytical_DM
        self.r_s = self.analytical_DM.scale_radius    #Get scale radius
    
    def mass_to_light(self, ml_kind='scalar'):
        """
            Sets the kind of mass to light ratio in non-linear search. Default is scalar.
            Input:
            ------------

            kind: Select the type of ML ratio. Three possibilities are available.
                    scalar  : Constante ML. Only one free parameter

                    gradient: One ML per gaussian following a gradient profile, i.e, ML[i] >= ML[i+1], where i=0 represents the most internal gaussian. This implies a number of free parameters equal to number of gaussians in surface luminosity density. Take care with it!

                    gaussian: A gaussian ML. Here there are three free parameters: central ML0, sigma of the gaussian and a lower value for the mass to light ratio. This allows a ML per MGE component, following a gaussian shape, without increasing (too much) the number of free parameters.
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
            if self.boundary['beta'][0] <= beta <= self.boundary['beta'][1]:
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
                if self.boundary['beta'][0] <= beta[i] <= self.boundary['beta'][1] :
                    pass
                else:
                    return -np.inf
        return 0.0

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

        try:
            """
                We're sampling the de-projected axial ratio of DM, due  physical reassons.
                But, lens model and Jampy should receive the projected axial ratio. So, we need to compute the inclination, and then project the axial ratio before passing it to the models. This is made bellow.
            """
                # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
            qmin = np.min(self.mass_profile.qobs_lum)
            if qmin**2  == parsDic['qinc']**2:
                """The Galaxy is Edge On"""
                inc_model = 90.0
            else:
                inc_model = np.degrees(np.arctan(np.sqrt((1 - qmin**2)/(qmin**2 - parsDic['qinc']**2))))
            
            qdm_proj  = np.sqrt( (np.sin(np.radians(inc_model)) * parsDic['qDM'] )**2  + np.cos( np.radians(inc_model))**2     )     #Projected DM axial ratio
        except:
            pass

            
        try:
            """
                If there an analytical profile, it will be updated in this code block.
            """
            r_s = self.r_s
            ell_comps = al.convert.elliptical_comps_from(axis_ratio=qdm_proj, phi=0.0) #Elliptical components in Pyautolens units
            eNFW = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=parsDic['kappa_s'],elliptical_comps=ell_comps, scale_radius=r_s) #Set the analytical model

            self.mass_profile.Analytic_Model(eNFW) # Include analytical model
            if self.quiet is False:
                print("Including the following Analytical DM profile:")
                print("#------------------------------------#")
                print(eNFW)
                print("\n")
        except:
            pass
            

        if self.has_dm is True:
            """
            If there an MGE DM  profile, it will be updated in this code block. But only if you have included it in your lens model.
            """
            if self.include_MGE_DM == "Lens" or self.include_MGE_DM == "Both":
                surf_dm_model = parsDic['kappa_s'] * self.surf_DM_dat
                qDM_model     = np.ones(self.qobs_DM_dat.shape) * qdm_proj
                
                self.mass_profile.MGE_Updt_parameters(ml=ml_model, mbh=mbh_model, gamma=gamma_model, 
                                                    surf_dm=surf_dm_model, qobs_dm=qDM_model)
            else:
                self.mass_profile.MGE_Updt_parameters(ml=ml_model, mbh=mbh_model, gamma=gamma_model)
        
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
        

        # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
        qmin = np.min(self.mass_profile.qobs_lum)
        if qmin**2  == parsDic['qinc']**2:
            """The Galaxy is Edge On"""
            inc_model = 90.0
        else:
            inc_model = np.degrees(np.arctan(np.sqrt((1 - qmin**2)/(qmin**2 - parsDic['qinc']**2))))
        
        

        if self.has_dm is True:
            """
                If there an MGE DM  profile, it will be updated in this code block. But only if you have included it in your dynamical model.
            """
            if self.include_MGE_DM == "Dynamical" or self.include_MGE_DM == "Both":
                surf_dm_model = (parsDic['kappa_s']) * self.surf_DM_dat
                qdm_proj  = np.sqrt( (np.sin(np.radians(inc_model)) * parsDic['qDM'] )**2  + np.cos( np.radians(inc_model))**2     )     #Projected DM axial ratio

                qDM_model     = np.ones(self.qobs_DM_dat.shape)*qdm_proj
                
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

        source_galaxy = al.Galaxy(
            redshift=self.mass_profile.z_s,
            pixelization=al.pix.Rectangular(shape=(40, 40)),
            regularization=al.reg.Constant(coefficient=1.0),
)
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
        
        #Check if the model has converged. If not, raise an error and return -inf
        try:
            fit = al.FitImaging(masked_imaging=self.masked_imaging, tracer=tracer)

            log_evidence = fit.log_evidence

            
            if self.quiet is False:
                print("Lens Galaxy Configuration:")
                print("Log Likelihood_with_regularization:", fit.log_likelihood_with_regularization)
                print("Log Likelihood:", fit.log_likelihood)
                print("Log Normalization", fit.noise_normalization)
                print("Log Evidence:", fit.log_evidence)
                print("#------------------------------------#")
                print(lens_galaxy)
                print("\n")
                

                aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))
                aplt.Inversion.reconstruction(fit.inversion)              
                

            return log_evidence
        except:
            return print("An exception ocurres in Pyautolens_log_likelihood().")
             

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
                    (ml, beta, qinc, log_mbh, kappa_s, qDM, mag_shear, phi_shear, gamma) = pars
                    
                    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                                'kappa_s':kappa_s, 'qDM':qDM, 'mag_shear':mag_shear,
                                'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml   = pars[0]
                    beta = pars[1:size+1]
                    (qinc, log_mbh, kappa_s,
                         qDM, mag_shear, phi_shear, gamma) = pars[beta.size+1:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                                'kappa_s':kappa_s, 'qDM':qDM, 'mag_shear':mag_shear,
                                'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

            elif self.ml_kind == 'gradient':
                if self.beta_kind == 'scalar':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml = pars[0:size]
                    (beta, qinc, log_mbh, kappa_s, qDM,
                         mag_shear, phi_shear, gamma) = pars[ml.size:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                                'kappa_s':kappa_s, 'qDM':qDM, 'mag_shear':mag_shear,
                                'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml   = pars[0:size]
                    beta = pars[ml.size:ml.size + size]
                    (qinc, log_mbh, kappa_s,
                         qDM, mag_shear, phi_shear, gamma) = pars[ml.size + beta.size:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                                'kappa_s':kappa_s, 'qDM':qDM, 'mag_shear':mag_shear,
                                'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

            elif self.ml_kind == 'gaussian':
                if self.beta_kind == 'scalar':
                    (ml0, delta, lower, beta, qinc, log_mbh, kappa_s, qDM,
                         mag_shear, phi_shear, gamma) = pars
                    
                    parsDic = {'ml0': ml0, 'delta':delta, 'lower':lower, 'beta': beta,
                                 'qinc': qinc, 'log_mbh': log_mbh, 'kappa_s':kappa_s, 'qDM':qDM,
                                 'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml0   = pars[0]
                    delta = pars[1]
                    lower = pars[2]
                    beta  = pars[3:size+3]

                    (qinc, log_mbh, kappa_s,
                         qDM, mag_shear, phi_shear, gamma) = pars[3 + beta.size:]
                    
                    parsDic = {'ml0': ml0, 'delta':delta, 'lower':lower, 'beta': beta,
                                 'qinc': qinc, 'log_mbh': log_mbh, 'kappa_s':kappa_s, 'qDM':qDM,
                                 'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

        elif self.has_dm is False:
            if self.ml_kind == 'scalar':
                if self.beta_kind == 'scalar':
                    (ml, beta, qinc, log_mbh, mag_shear, phi_shear, gamma) = pars
                    
                    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                                'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml   = pars[0]
                    beta = pars[1:size+1]
                    (qinc, log_mbh, mag_shear, phi_shear, gamma) = pars[beta.size+1:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                                'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

            elif self.ml_kind == 'gradient':
                if self.beta_kind == 'scalar':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml = pars[0:size]
                    (beta, qinc, log_mbh, mag_shear, phi_shear, gamma) = pars[ml.size:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                                'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml   = pars[0:size]
                    beta = pars[ml.size:ml.size + size]
                    (qinc, log_mbh, mag_shear, phi_shear, gamma) = pars[ml.size + beta.size:]
                    
                    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                                'mag_shear':mag_shear, 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

            elif self.ml_kind == 'gaussian':
                if self.beta_kind == 'scalar':
                    (ml0, delta, lower, beta, qinc, log_mbh,
                         mag_shear, phi_shear, gamma) = pars
                    
                    parsDic = {'ml0': ml0, 'delta':delta, 'lower':lower, 'beta': beta,
                                 'qinc': qinc, 'log_mbh': log_mbh, 'mag_shear':mag_shear,
                                 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic
                elif self.beta_kind == 'vector':
                    size = self.mass_profile.surf_lum.size   #Size of surf_lum

                    ml0   = pars[0]
                    delta = pars[1]
                    lower = pars[2]
                    beta  = pars[3:size+3]

                    (qinc, log_mbh, mag_shear, phi_shear, gamma) = pars[3 + beta.size:]
                    
                    parsDic = {'ml0': ml0, 'delta':delta, 'lower':lower, 'beta': beta,
                                 'qinc': qinc, 'log_mbh': log_mbh, 'mag_shear':mag_shear,
                                 'phi_shear': phi_shear, 'gamma': gamma}
                    return parsDic

    def log_probability(self,pars):
        """
        Log-probability function for whole model.
        Input:
        ----------
            pars: current values in the Emcee sample.
        Output:
        ---------
            log probability for the combined model.
        """
        
        parsDic = self.Dic(pars)
        if not np.isfinite( self.check_beta(parsDic["beta"]) ):
            #We need to check if beta is equal to 1.0, because this could cause inf's in Jampy.
            return -1e300
        if self.quiet is False:
            print("ParsDic", parsDic)
        

        return self.JAM_log_likelihood(parsDic) + self.Pyautolens_log_likelihood(parsDic)


    def __call__(self, pars):
        return self.log_probability(pars)

