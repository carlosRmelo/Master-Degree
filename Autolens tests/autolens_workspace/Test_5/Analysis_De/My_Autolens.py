
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
from autoarray.fit import fit



#Dark matter component
"""
  
   There are two ways to set a DM component.
    
    One of them is using a MGE decomposition of a DM mass profile, that could be add in lens model only, dynamical model only or both at same time.

    And the other is add a analytical (it means non-MGE model) DM profile for the lens deflection angle. In this case, the DM profile is add only in the lens model. You should take a *lot* of care using a DM compoent only in the lens model, because this probably makes your complet model non self consistent. This option is recommended when your DM profile has a MGE decomposition (that will be add in the dynamical model) AND has a analitcal (or a faster calculation) deflection angle.
    
    Until now, only Pyautolens  deflection angles could be add in this way. And more, only the pseudo analytical NFW is implemented correctly.


    In any case above, You must call def has_MGE_DM function. Check it for more details. 
    If you want to include an analytical DM model, you should call: def include_DM_analytical function. Also check it for mode details.

"""




class Models(object):

    def __init__(self, 
                    mass_profile, masked_imaging, quiet=True
                    ):
        """
        Input:
        --------------------------
        mass_profile: Pyautolens object
            MGE mass profile for Pyautolens modelling. You can find more information in total_mass_profiles.py, the script were we implement MGE mass model.
        masked_imaging: Pyautolens object
            Pyautolens object containing all the information about the image that we want model. You can find more information in Pyautolens documentation, linked in the header of this document.
        quiet: Boolean
            If False, allow the some plots of Jampy and Pyautolens model.
        """

        self.mass_profile   = mass_profile
        self.masked_imaging = masked_imaging
        self.quiet          = quiet

        self.boundaries()
        self.priors()


    def boundaries(self):
        """
            Define the boundaries for parameters.
            boundary_name: [lower, upper]
        """

        self.boundary = {'ml': [0.5, 15], 'kappa_s': [0, 2], 'qDM': [0.1, 1], 'log_mbh':[7, 11],
                 'mag_shear': [0, 0.1], 'phi_shear': [0, 179], 'gamma': [0.80, 1.20]}

    def priors(self): 
        """
            For now, only gaussian prior is for gamma, because we have a prior information about it. All other parameters have flat (or non-informative) priors.
            #Gaussian priors. [mean, sigma]
        """
        self.prior = {'gamma': [1.0, 0.05] }



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
        
        
        for keys in parsDic:
            if self.boundary[keys][0] <= parsDic[keys] <= self.boundary[keys][1]:
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
        rst += 0.0     #log_mbh
        rst += 0.0     #mag_shear
        rst += 0.0     #phi_shear

        
        #Finaly gaussian prior for gamma
        rst += -0.5 * (parsDic['gamma'] - self.prior['gamma'][0])**2/self.prior['gamma'][1]**2     #gamma
        
        return rst


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
    

    
    def Updt_Pyautolens(self,parsDic):
        """
        Update the Lens mass model
        Input:
        ----------
            parsDic: parameter dictionary {'paraName', value}
        """

        ml_model    = parsDic['ml']
        mbh_model   = 10**parsDic['log_mbh']
        gamma_model = parsDic['gamma']

        

    
        try:
            """
                If there an analytical profile, it will be updated in this code block.
            """

            r_s = self.r_s
            ell_comps = al.convert.elliptical_comps_from(axis_ratio=parsDic['qDM'], phi=0.0) #Elliptical components in Pyautolens units
            eNFW = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=parsDic['kappa_s'],elliptical_comps=ell_comps, scale_radius=r_s) #Set the analytical model

            self.mass_profile.Analytic_Model(eNFW)        #Include analytical model
            if self.quiet is False:
                print("Including the following Analytical DM profile:")
                print("#------------------------------------#")
                print(eNFW)
                print("\n")
        except:
            pass
            

        self.mass_profile.MGE_Updt_parameters(ml=ml_model, mbh=mbh_model, gamma=gamma_model)


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
        
        #Check if the model has converged. If not, return -inf
        try:
            fit = al.FitImaging(masked_imaging=self.masked_imaging, tracer=tracer)

            log_likelihood = fit.log_likelihood_with_regularization

            
            if self.quiet is False:
                print("Lens Galaxy Configuration:")
                print("Log Likelihood_with_regularization:", log_likelihood)
                print("Log Normalization", fit.noise_normalization)
                print("Log Evidence:", fit.log_evidence)
                print("#------------------------------------#")
                print(lens_galaxy)
                print("\n")
                

                aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))
                aplt.Inversion.reconstruction(fit.inversion)              
                

            return log_likelihood
        except:
            print("An exception ocurres in Pyautolens_log_likelihood().")
            return -np.inf

    

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
        ml, kappa_s, qDM, log_mbh, mag_shear, phi_shear, gamma = pars
        parsDic = {"ml": ml, "kappa_s": kappa_s, "qDM": qDM,
                    "log_mbh":log_mbh, "mag_shear": mag_shear, "phi_shear": phi_shear, 
                    "gamma": gamma}
        if self.quiet is False:
            print("ParsDic:")
            print("#------------------------------------#")
            print(parsDic)
            print("\n")

        #Checking boundaries
        if not np.isfinite(self.check_boundary(parsDic)):
            return -np.inf
        #calculating the log_priors
        lp = self.log_prior(parsDic)

        return lp + self.Pyautolens_log_likelihood(parsDic)


    def __call__(self, pars):
        return self.log_probability(pars)

