import autofit as af
import numpy as np
from astropy import cosmology as cosmo
from autoarray.structures import arrays
from autoarray.structures import grids
from autogalaxy import dimensions as dim
from autogalaxy.profiles import geometry_profiles
from autogalaxy.profiles import mass_profiles as mp
from pyquad import quad_grid
from scipy import special
import typing
from time import perf_counter as clock
import time


#------------------------------This entire block defines some functions and quantities for MGE model.-------------------------------------------------------#

#Some packages#

from astropy.cosmology import Planck15
from astropy import constants
import astropy.units as u
import math
import scipy.integrate as integrate
from autogalaxy.util import cosmology_util

from multiprocessing import Pool
import multiprocessing

from jampy.quadva import quadva
from numba import njit, double, jit

#Some useful constants#

G_Mpc = constants.G.to("Mpc3 / (solMass * s2)")     #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = constants.c.to("Mpc / s")                   #Speed of light in Mpc/s
                      

#Deflection angle of MGE model.#
"""
Paper original: Barnabè et al.(2012) 2012 MNRAS.423.1073B

(https://ui.adsabs.harvard.edu/abs/2012MNRAS.423.1073B/abstract)

Disclaimer: Pay too much attention for the  correct axial ratio in deflection angle.  In this paper, we write the deflection angle in terms os de-projected axial ratio (wich for me no make sense). Also, I rederived this expresion and find the same result, but in terms o projected axial ratio.

Another reference that agree with my results are: Glenn van de Ven et al. (2010) ApJ 719 1481
(https://iopscience.iop.org/article/10.1088/0004-637X/719/2/1481)

For this reason, here we advocate the second definition.
"""	

def alpha_x(tau, parameters):
    y, x, M, sigma, q = parameters
    
    x_til = x/sigma
    y_til = y/sigma
    eta = np.sqrt(1.0 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(x_til/(np.sqrt(1 - eta_sq*(tau)**2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    
    sum = np.sum(aux*exp)
    
    function = tau*sum
    
    return function



def alpha_y(tau, parameters):
    y, x, M, sigma,q = parameters
    
    x_til = x/sigma
    y_til = y/sigma
    eta = np.sqrt(1.0 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(y_til/(np.power(1 - eta_sq*(tau)**2, 3/2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    
    sum = np.sum(aux*exp)
    function = tau*sum
    
    return function




def integral_x(parameters):
    return integrate.quad(alpha_x, 0, 1, args=(parameters))

def integral_y(parameters):
    return integrate.quad(alpha_y, 0, 1, args=(parameters))


#We have two function because it is possible to use different integration methods
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def alphax(tau1, 
            y, x, M, sigma, q):
    
    
    tau = np.reshape(tau1.copy(), (tau1.size,1))
   
    x_til = x/sigma
    y_til = y/sigma

   
    
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(x_til/(np.sqrt(1 - eta_sq*(tau)**2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    

    
    arr = aux*exp    
    
    return tau1*np.sum(arr, 1)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def alphay(tau1,
            y, x, M, sigma, q):
    

    tau = np.reshape(tau1.copy(), (tau1.size,1))
    x_til = x/sigma
    y_til = y/sigma
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(y_til/(np.power(1 - eta_sq*(tau)**2, 3/2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    
    
    arr = aux*exp
    
    return tau1*np.sum(arr, 1)

#Here we define alpha_x and alpha_y as numba functions
#alphax_nb = njit(double[:](double[:],double,double,double[:],double[:],double[:]))(alphax)
#alphay_nb = njit(double[:](double[:],double,double,double[:],double[:],double[:]))(alphay)
    

#MGE Class#
class MGE(geometry_profiles.SphericalProfile, mp.MassProfile):
    @af.map_types
    def __init__(
        self, centre: dim.Position = (0.0, 0.0), processes: int = 1, method: str = "quadva", gamma: float = 1.0, epsabs: float = 1e-10, epsrel: float = 1e-5,
        ):
        """
        Represents a MGE.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        processes: int
            Number of processes you want to do at same time during the deflection angle calculation. Default is 1, then in this case the integration will be done in serial.
        method: str
            Method we will use to compute the deflection angle.
            If quadva: uses quadva method by Cappellari. (Not allowed to use parallel loop yet.)
            If sciquad: uses scipy.quad integration. In this case, if processes is set equal to one, serial integration is used.
                        Otherwise, deflection angle is calculated in parallel.
        gamma: float
            Is the Pos-Newtonian Parameter. If gamma is equal to one we recover General Relativity. Otherwise, we may have discovered something new.
        epsabs and epsrel: float, float
            Pure absolute error for quadva. Default: 1e-10
            Pure relative error for quadva. Default: 1e-5
            Generally the error test is a mixed one, but pure absolute error and pure relative error are allowed.  If a pure relative error test is specified, the tolerance must be at least 100*EPS.  (Jampy documentation for more detailed description.)
        """


        super(MGE, self).__init__(
            centre=centre,
        )
        self.epsabs    = epsabs
        self.epsrel    = epsrel
        self.processes = processes
        self.method    = method
        self.gamma     = gamma
        self.analytic_profile = None
    

    def Analytic_Model(self,
                            analytic_profile = None):
        """
        Analytical Mass profile
        ------------------------
        Includes an analytical mass profile in the lens mass model. Until now, only analytical models already implemented in Pyautolens can be used. Besides that, only the deflection angle can be computed.

        Input:
        analytic_profile: Pyautolens mass profile
            Pyautolens mass profile describing the mass for lensing computation.
        """

        self.analytic_profile = analytic_profile


    def MGE_comps(self, z_l, z_s,
                    surf_lum, sigma_lum, qobs_lum, ml=None,
                    surf_dm=None, sigma_dm=None, qobs_dm=None,
                    mbh=None, sigma_mbh=None, qobs_mbh=None):
        """
        MGE mass model parameters
        --------------------
       SURF_LUM: array
            vector of length ``N`` containing the peak surface brightness of the
            MGE Gaussians describing the galaxy surface brightness in units of
            ``Lsun/pc**2`` (solar luminosities per parsec**2).
        SIGMA_LUM:
            vector of length ``N`` containing the dispersion in arcseconds of
            the MGE Gaussians describing the galaxy surface brightness.
        QOBS_LUM:
            vector of length ``N`` containing the observed axial ratio of the MGE
            Gaussians describing the galaxy surface brightness.
        z_l: float
            lens redshift
        z_s: float
            source redshift
        ML:
            Mass-to-light ratio to multiply the values given by SURF_LUM, to convert surface luminosity density to surface mass dendity.

            If this keyword is not set, we assume a mass-to-light ratio iqual to 1.

            Also, you can give a ML vector of same length of surface luminosity, with each component give a correspond mass-to-light ratio. This should be used if you want a model with a variant ML.
                
        Optional Keywords
        -----------------
        SURF_DM:
            vector of length ``S`` containing the peak surface mass density of the
            MGE Gaussians describing the dark matter surface mass in units of
            ``Msun/pc**2`` (solar masses per parsec**2).
        SIGMA_DM:
            vector of length ``S`` containing the dispersion in arcseconds of
            the MGE Gaussians describing the dark matter surface mass.
        QOBS_DM:
            vector of length ``S`` containing the observed axial ratio of the MGE
            Gaussians describing the dark matter surface mass.       
        MBH:
            Mass of a nuclear supermassive black hole in solar masses.
         sigma_mbh:
            This scalar gives the sigma in arcsec of the Gaussian representing the
            central black hole of mass MBH (See Section 3.1.2 of `Cappellari 2008.
            <http://adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_)
            The gravitational potential is indistinguishable from a point source
            for ``radii > 2*RBH``, so the default ``RBH=0.01`` arcsec is appropriate
            in most current situations.
         qobs_mbh:
            Observational axial ratio for black hole. Default considers spherical symmetry, i.e, qobs_mbh=1. But, due the current telescopes resolution, any other values are not recommended.
        """
        
        self.z_l = z_l
        self.z_s = z_s
        critical_density_kpc = cosmology_util.critical_surface_density_between_redshifts_from(
                                                        redshift_0=z_l,   
                                                        redshift_1=z_s,
                                                        cosmology=cosmo.Planck15, 
                                                        unit_mass="solMass", 
                                                        unit_length="kpc") #Critical Surface Density in [Msun/kpc2]
        
        self.critical_density = critical_density_kpc * (1e3)**2  #Critical Surface Density in [Msun/Mpc2]


        assert surf_lum.size == sigma_lum.size == qobs_lum.size, "The luminous MGE components do not match"
        
        self.surf_lum  = surf_lum
        self.sigma_lum = sigma_lum
        self.qobs_lum  = qobs_lum

        if ml is None:
            self.ml = 1.0
        elif np.isscalar(ml):
            self.ml = ml
        else:
            assert ml.size == surf_lum.size, "The ML components do not match"
            self.ml = ml


        if surf_dm is not None:
            assert surf_dm.size == sigma_dm.size == qobs_dm.size, "The DM MGE components do not match"
            
            self.surf_dm  = surf_dm
            self.sigma_dm = sigma_dm
            self.qobs_dm  = qobs_dm
        else:
            self.surf_dm  = None
            self.sigma_dm = None
            self.qobs_dm  = None

        if mbh is not None:
            self.mbh = mbh
            if sigma_mbh is None:
                self.sigma_mbh = 0.01
            else:
                self.sigma_mbh = sigma_mbh
            
            if qobs_mbh is None:
                self.qobs_mbh = 1.0
            else:
                self.qobs_mbh = qobs_mbh
        else:
            self.mbh = None
            self.qobs_mbh = None



        #Angular diametre distances
        self.D_l  = Planck15.angular_diameter_distance(self.z_l)
        self.D_s  = Planck15.angular_diameter_distance(self.z_s)
        self.D_ls = Planck15.angular_diameter_distance_z1z2(self.z_l, self.z_s)

            #Compute total mass of model; And define others quantities. 
            # It's part of initialization.
        self.MGE_total_components()       





    def MGE_total_components(self):
        """
            Compute total components for the model, i.e, Total mass, Totoal sigma, Total qobs and so on.
        """
        #Converting some qauntities to arcsec to parsec, or arcsec to radian

        #Luminous component
        sigma_lum_ARC = self.sigma_lum * u.arcsec           #sigma in arcsec
        sigma_lum_PC  = (sigma_lum_ARC * self.D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc

        self.sigma_lum_ARC = sigma_lum_ARC.value
        self.sigma_lum_PC  = sigma_lum_PC.value

        #Convert surf_lum to total Luminosity per Guassian and then to total mass per gaussian
            #Total luminosity per gaussian component in L_sun

        self.Lum_star = 2 * np.pi * self.surf_lum * (self.sigma_lum_PC**2) * self.qobs_lum
        
        #Update the luminosity mass based on M/L.
            #Total luminosity mass per gaussian in M_sun
        self.Mass_star = self.Lum_star * self.ml                            

        """
        Here we took the opportunity to define all total MGE components, it is, Total mass, Total sigma and Total observational axial ratio.
        """
        Total_mass      = self.Mass_star
        Total_sigma_ARC = self.sigma_lum_ARC
        Total_qobs      = self.qobs_lum

        #DM component
        if self.surf_dm is not None:
            sigma_dm_ARC = self.sigma_dm * u.arcsec           #sigma in arcsec
            sigma_dm_PC  = (sigma_dm_ARC * self.D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc

            self.sigma_dm_ARC = sigma_dm_ARC.value
            self.sigma_dm_PC  = sigma_dm_PC.value

            #Convert surf_DM mass total mass per Guassian 
                #Total DM mass per gaussian in M_sun
            self.Mass_DM = 2 * np.pi * self.surf_dm * (self.sigma_dm_PC**2) * self.qobs_dm

            #Update Total components to include DM
            Total_mass      = np.append(Total_mass, self.Mass_DM)
            Total_sigma_ARC = np.append(Total_sigma_ARC, self.sigma_dm_ARC)
            Total_qobs      = np.append(Total_qobs, self.qobs_dm)

        #BH component
        if self.mbh is not None:
            sigma_mbh_ARC = self.sigma_mbh * u.arcsec           #sigma in arcsec
            sigma_mbh_PC  = (sigma_mbh_ARC * self.D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc

            self.sigma_mbh_ARC = sigma_mbh_ARC.value
            self.sigma_mbh_PC  = sigma_mbh_PC.value
            
            #Total super massive black hole mass 
            self.Mass_bh = self.mbh

            #Update Total components to include black hole
            Total_mass      = np.append(Total_mass, self.Mass_bh)
            Total_sigma_ARC = np.append(Total_sigma_ARC, self.sigma_mbh_ARC)
            Total_qobs      = np.append(Total_qobs, self.qobs_mbh)


        #Total components of MGE model, per gaussian 
        self.Total_Mass      = Total_mass                                      #[M_sun]
        self.Total_sigma_ARC = Total_sigma_ARC                                 #[arcsec]
        self.Total_sigma_RAD = ((Total_sigma_ARC * u.arcsec).to(u.rad)).value  #[rad]
        self.Total_qobs      = Total_qobs                                      #[ad]

        assert self.Total_Mass.size == self.Total_sigma_RAD.size == self.Total_qobs.size, "Total components do not match"





    def MGE_Updt_parameters(self,
                                surf_lum=None, sigma_lum=None, qobs_lum=None, ml=None,
                                surf_dm=None, sigma_dm=None, qobs_dm=None,
                                mbh=None, sigma_mbh=None, qobs_mbh=None,
                                gamma = None, quiet=True):
        """
            Updates the model's parameter grid. Particularly useful when used in conjunction with Emcee.
            It is assumed that the classes MGE.MGE_comps and MGE.MGE_Grid_parameters have already been properly initialized.
            
            Inputs:
            --------------------------------
                Same as MGE_comps
            gamma: float
                Pos-Newtonian Parameter. If gamma is equal to one we recover General Relativity.
            quiet: Boolean
                If false, accuses that the model was updated correctly.
        """

        pars = {'surf_lum':surf_lum, 'sigma_lum':sigma_lum, 'qobs_lum':qobs_lum, 'ml':ml,
                    'surf_dm':surf_dm, 'sigma_dm':sigma_dm, 'qobs_dm':qobs_dm,
                    'mbh':mbh, 'sigma_mbh':sigma_mbh, 'qobs_mbh':qobs_mbh,
                    'gamma':gamma}

        try:

            for key in pars.keys():
                if pars[key] is not None:
                    self.__dict__[key] = pars[key]

        
            self.MGE_total_components()
        except:
            raise ValueError("Your update values are not valid. Please, consider check your parameters before continue.Probably you set up an analytical dark matter component and a MGE at same time, but not properly.")

        
        if quiet is False:
            return print("Parameters updated successfully!")

    
            

    def convergence_from_grid(self, grid):

        squared_distances = np.square(grid[:, 0] - self.centre[0]) + np.square(
            grid[:, 1] - self.centre[1]
        )
        central_pixel = np.argmin(squared_distances)

        convergence = np.zeros(shape=grid.shape[0])
        #    convergence[central_pixel] = np.pi * self.einstein_radius ** 2.0
        return convergence

    

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid, quiet=True):
        """"
        When we use adaptive grid (Voronoi Adaptive or similar) source plane grid are constantly update. The following lines make this update for MGE model.

        Input:
        --------------------------------
        grid: Ndarray
            Grid where we wanto to compute the model, containing (y,x) position in arcsec.
        quiet: Boolean
            If False, print steps (integrator,  number of cores, ...)
        """

        if self.analytic_profile is not None:
            analytical_deflection = self.analytic_profile.deflections_from_grid(grid)
        
        
        
        #Now, depending of your input in **__init__.method**, a different integrator is used.

        if self.method == "sciquad":
            if quiet is False: print("Integration using scipy.quad")
            
            if self.processes != 1:
                #Integration will be done in parallel using the self.processes cores
                print('Parallel Integration. Number of cores is:', self.processes)
                
                #Deflection in x
                pool = Pool(processes=self.processes)                           #Defining the number of cores to be used
                result_x =  pool.map(integral_x, self.Grid_parameters)          #We pass each of the parameter lines as an argument and call the integral
                pool.close()                                                    #We close the process after it finishes



                #Deflection in y
                pool = Pool(processes=self.processes)                           #Defining the number of cores to be used
                result_y =  pool.map(integral_y, self.Grid_parameters)          ##We pass each of the parameter lines as an argument and call the integral
                pool.close()                                                    #e close the process after it finishes

                result_x = np.array(result_x)
                result_y = np.array(result_y)

            else:
                #The integration will be done in serie.
                if quiet is False: print('Serial integration')

                result_x = np.zeros([len(grid), 2])              #Where the results of the deflection at x are saved
                result_y = np.zeros([len(grid), 2])              #Where the results of the deflection at y are saved

                for i in range(len(grid)):                                  
                    result_x[i] = integral_x(self.Grid_parameters[i])       #Deflection in x
                    result_y[i] = integral_y(self.Grid_parameters[i])       #Deflection in y
                    
        elif self.method == "quadva":
            if quiet is False: print("Quadva integration")


            grid = (grid*u.arcsec).to(u.rad).value           #Get grid in rad

            grid_result = np.zeros_like(grid)                #Set grid where results are saved
            start = time.time() #Time controle, if you want to test performance
            #alpha_y integrals
            for i in range(len(grid)):
                if grid_result[i, 0] == 0:
                    result_y = quadva(alphay, [0., 1.], args=(grid[i][0], grid[i][1], self.Total_Mass, self.Total_sigma_RAD, self.Total_qobs),epsrel=self.epsrel, epsabs=self.epsabs)  #Integral in y
                    grid_result[i, 0] = result_y[0]
                    
                    index = np.where( (grid[:,0] == grid[i][0]) & (grid[:,1] == -grid[i][1])  ) #Fix y and change x
                    grid_result[index, 0] = result_y[0]
                    
                    index = np.where( (grid[:,0] == -grid[i][0]) & (grid[:,1] == grid[i][1])  ) #Fix x and change y
                    grid_result[index, 0] = -result_y[0]
                    
                    index = np.where( (grid[:,0] == -grid[i][0]) & (grid[:,1] == -grid[i][1])  ) #Change both
                    grid_result[index, 0] = -result_y[0]
    
    
            #alpha_x integrals
            for i in range(len(grid)):
                if grid_result[i, 1] == 0:
                    result_x = quadva(alphax, [0., 1.], args=(grid[i][0], grid[i][1], self.Total_Mass, self.Total_sigma_RAD, self.Total_qobs),epsrel=self.epsrel, epsabs=self.epsabs)  #Integral in x
                    grid_result[i, 1] = result_x[0]
                    
                    index = np.where( (grid[:,0] == -grid[i][0]) & (grid[:,1] == grid[i][1])  ) #Fix x and change y
                    grid_result[index, 1] = result_x[0]
                    
                    index = np.where( (grid[:,0] == grid[i][0]) & (grid[:,1] == -grid[i][1])  ) #Fix y and change x
                    grid_result[index, 1] = -result_x[0]
                    
                    index = np.where( (grid[:,0] == -grid[i][0]) & (grid[:,1] == -grid[i][1]) ) #Change both 


            #Print core name, and time computation of deflection angle
            if quiet is False: print(multiprocessing.current_process().name, time.time()-start)
        else:
            return print("Invalid integration method")

        #Constant Factor from integral in deflection angles
        const_factor = 1/(np.pi * self.critical_density * self.D_l**2)

        const_factor = const_factor.value

        #Updating the grid with the angle value after deflection
        grid[:, 0] = ((const_factor*grid_result[:, 0])*u.rad).to(u.arcsec).value
        grid[:, 1] = ((const_factor*grid_result[:, 1])*u.rad).to(u.arcsec).value

        if self.analytic_profile is not None:
            return (0.5 * (1.0 + self.gamma))*(grid + analytical_deflection)
        else:
            return (0.5 * (1.0 + self.gamma))*grid
    @property
    def is_MGE(self):
        return True








#-------------------------------------------------FIM--------------------------------------------------------------------------------#


class PointMass(geometry_profiles.SphericalProfile, mp.MassProfile):
    @af.map_types
    def __init__(
        self, centre: dim.Position = (0.0, 0.0), einstein_radius: dim.Length = 1.0
    ):
        """
        Represents a point-mass.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius of the point-mass.
        """
        super(PointMass, self).__init__(centre=centre)
        self.einstein_radius = einstein_radius

    def convergence_from_grid(self, grid):

        squared_distances = np.square(grid[:, 0] - self.centre[0]) + np.square(
            grid[:, 1] - self.centre[1]
        )
        central_pixel = np.argmin(squared_distances)

        convergence = np.zeros(shape=grid.shape[0])
        #    convergence[central_pixel] = np.pi * self.einstein_radius ** 2.0
        return convergence

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(
            grid=grid, radius=self.einstein_radius ** 2 / grid_radii
        )

    @property
    def is_point_mass(self):
        return True


class EllipticalBrokenPowerLaw(mp.EllipticalMassProfile, mp.MassProfile):
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        inner_slope: float = 1.5,
        outer_slope: float = 2.5,
        break_radius: dim.Length = 0.01,
    ):
        """
        Elliptical, homoeoidal mass model with an inner_slope
        and outer_slope, continuous in density across break_radius.
        Position angle is defined to be zero on x-axis and
        +ve angle rotates the lens anticlockwise

        The grid variable is a tuple of (theta_1, theta_2), where
        each theta_1, theta_2 is itself a 2D array of the x and y
        coordinates respectively.~
        """

        super(EllipticalBrokenPowerLaw, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )

        self.einstein_radius = np.sqrt(self.axis_ratio) * einstein_radius
        self.break_radius = break_radius
        self.inner_slope = inner_slope
        self.outer_slope = outer_slope

        # Parameters defined in the notes
        self.nu = break_radius / self.einstein_radius
        self.dt = (2 - self.inner_slope) / (2 - self.outer_slope)

        # Normalisation (eq. 5)
        if self.nu < 1:
            self.kB = (2 - self.inner_slope) / (
                (2 * self.nu ** 2)
                * (1 + self.dt * (self.nu ** (self.outer_slope - 2) - 1))
            )
        else:
            self.kB = (2 - self.inner_slope) / (2 * self.nu ** 2)

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def convergence_from_grid(self, grid):
        """
        Returns the dimensionless density kappa=Sigma/Sigma_c (eq. 1)
        """

        # Elliptical radius
        radius = np.hypot(grid[:, 1] * self.axis_ratio, grid[:, 0])

        # Inside break radius
        kappa_inner = self.kB * (self.break_radius / radius) ** self.inner_slope

        # Outside break radius
        kappa_outer = self.kB * (self.break_radius / radius) ** self.outer_slope

        return kappa_inner * (radius <= self.break_radius) + kappa_outer * (
            radius > self.break_radius
        )

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        return arrays.Array.manual_1d(
            array=np.zeros(shape=grid.shape[0]), shape_2d=grid.sub_shape_2d
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid, max_terms=20):
        """
        Returns the complex deflection angle from eq. 18 and 19
        """
        # Rotate coordinates
        z = grid[:, 1] + 1j * grid[:, 0]

        # Elliptical radius
        R = np.hypot(z.real * self.axis_ratio, z.imag)

        # Factors common to eq. 18 and 19
        factors = (
            2
            * self.kB
            * (self.break_radius ** 2)
            / (self.axis_ratio * z * (2 - self.inner_slope))
        )

        # Hypergeometric functions
        # (in order of appearance in eq. 18 and 19)
        # These can also be computed with scipy.special.hyp2f1(), it's
        # much slower can be a useful test
        F1 = self.hyp2f1_series(
            self.inner_slope, self.axis_ratio, R, z, max_terms=max_terms
        )
        F2 = self.hyp2f1_series(
            self.inner_slope, self.axis_ratio, self.break_radius, z, max_terms=max_terms
        )
        F3 = self.hyp2f1_series(
            self.outer_slope, self.axis_ratio, R, z, max_terms=max_terms
        )
        F4 = self.hyp2f1_series(
            self.outer_slope, self.axis_ratio, self.break_radius, z, max_terms=max_terms
        )

        # theta < break radius (eq. 18)
        inner_part = factors * F1 * (self.break_radius / R) ** (self.inner_slope - 2)

        # theta > break radius (eq. 19)
        outer_part = factors * (
            F2
            + self.dt * (((self.break_radius / R) ** (self.outer_slope - 2)) * F3 - F4)
        )

        # Combine and take the conjugate
        deflections = (
            inner_part * (R <= self.break_radius) + outer_part * (R > self.break_radius)
        ).conjugate()

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((np.imag(deflections), np.real(deflections))).T)
        )

    @staticmethod
    def hyp2f1_series(t, q, r, z, max_terms=20):
        """
        Computes eq. 26 for a radius r, slope t,
        axis ratio q, and coordinates z.
        """

        # u from eq. 25
        q_ = (1 - q ** 2) / (q ** 2)
        u = 0.5 * (1 - np.sqrt(1 - q_ * (r / z) ** 2))

        # First coefficient
        a_n = 1.0

        # Storage for sum
        F = np.zeros_like(z, dtype="complex64")

        for n in range(max_terms):
            F += a_n * (u ** n)
            a_n *= ((2 * n) + 4 - (2 * t)) / ((2 * n) + 4 - t)

        return F


class SphericalBrokenPowerLaw(EllipticalBrokenPowerLaw):
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        inner_slope: float = 1.5,
        outer_slope: float = 2.5,
        break_radius: dim.Length = 0.01,
    ):
        """
        Elliptical, homoeoidal mass model with an inner_slope
        and outer_slope, continuous in density across break_radius.
        Position angle is defined to be zero on x-axis and
        +ve angle rotates the lens anticlockwise

        The grid variable is a tuple of (theta_1, theta_2), where
        each theta_1, theta_2 is itself a 2D array of the x and y
        coordinates respectively.~
        """

        super(SphericalBrokenPowerLaw, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            inner_slope=inner_slope,
            outer_slope=outer_slope,
            break_radius=break_radius,
        )


class EllipticalCoredPowerLaw(mp.EllipticalMassProfile, mp.MassProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        slope: float = 2.0,
        core_radius: dim.Length = 0.01,
    ):
        """
        Represents a cored elliptical power-law density distribution

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(EllipticalCoredPowerLaw, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        self.einstein_radius = einstein_radius
        self.slope = slope
        self.core_radius = core_radius

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters"""
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (
            self.slope - 1
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def convergence_from_grid(self, grid):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        The *grid_like_to_structure* decorator reshapes the NumPy arrays the convergence is outputted on. See \
        *aa.grid_like_to_structure* for a description of the output.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """

        covnergence_grid = np.zeros(grid.shape[0])

        grid_eta = self.grid_to_elliptical_radii(grid)

        for i in range(grid.shape[0]):
            covnergence_grid[i] = self.convergence_func(grid_eta[i])

        return covnergence_grid

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        potential_grid = quad_grid(
            self.potential_func,
            0.0,
            1.0,
            grid,
            args=(self.axis_ratio, self.slope, self.core_radius),
        )[0]

        return self.einstein_radius_rescaled * self.axis_ratio * potential_grid

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        def calculate_deflection_component(npow, index):
            einstein_radius_rescaled = self.einstein_radius_rescaled

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= (
                einstein_radius_rescaled
                * quad_grid(
                    self.deflection_func,
                    0.0,
                    1.0,
                    grid,
                    args=(npow, self.axis_ratio, self.slope, self.core_radius),
                )[0]
            )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, grid_radius):
        return self.einstein_radius_rescaled * (
            self.core_radius ** 2 + grid_radius ** 2
        ) ** (-(self.slope - 1) / 2.0)

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (
            (eta / u)
            * ((3.0 - slope) * eta) ** -1.0
            * (
                (core_radius ** 2.0 + eta ** 2.0) ** ((3.0 - slope) / 2.0)
                - core_radius ** (3 - slope)
            )
            / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)
        )

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (core_radius ** 2 + eta_u ** 2) ** (-(slope - 1) / 2.0) / (
            (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5)
        )

    @property
    def ellipticity_rescale(self):
        return (1.0 + self.axis_ratio) / 2.0

    def summarize_in_units(
        self,
        radii,
        prefix="",
        whitespace=80,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        summary = super().summarize_in_units(
            radii=radii,
            prefix=prefix,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            whitespace=whitespace,
        )

        return summary

    @property
    def unit_mass(self):
        return "angular"


class SphericalCoredPowerLaw(EllipticalCoredPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        slope: float = 2.0,
        core_radius: dim.Length = 0.01,
    ):
        """
        Represents a cored spherical power-law density distribution

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(SphericalCoredPowerLaw, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=core_radius,
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        eta = self.grid_to_grid_radii(grid=grid)
        deflection = np.multiply(
            2.0 * self.einstein_radius_rescaled,
            np.divide(
                np.add(
                    np.power(
                        np.add(self.core_radius ** 2, np.square(eta)),
                        (3.0 - self.slope) / 2.0,
                    ),
                    -self.core_radius ** (3 - self.slope),
                ),
                np.multiply((3.0 - self.slope), eta),
            ),
        )
        return self.grid_to_grid_cartesian(grid=grid, radius=deflection)


class EllipticalPowerLaw(EllipticalCoredPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super(EllipticalPowerLaw, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=dim.Length(0.0),
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.
        ​
        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore, \
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        This code is an adaption of Tessore & Metcalf 2015:
        https://arxiv.org/abs/1507.01819
        ​
        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        slope = self.slope - 1.0
        einstein_radius = (
            2.0 / (self.axis_ratio ** -0.5 + self.axis_ratio ** 0.5)
        ) * self.einstein_radius

        factor = np.divide(1.0 - self.axis_ratio, 1.0 + self.axis_ratio)
        b = np.multiply(einstein_radius, np.sqrt(self.axis_ratio))
        phi = np.arctan2(
            grid[:, 0], np.multiply(self.axis_ratio, grid[:, 1])
        )  # Note, this phi is not the position angle
        R = np.sqrt(
            np.add(np.multiply(self.axis_ratio ** 2, grid[:, 1] ** 2), grid[:, 0] ** 2)
        )
        z = np.add(np.multiply(np.cos(phi), 1 + 0j), np.multiply(np.sin(phi), 0 + 1j))

        complex_angle = (
            2.0
            * b
            / (1.0 + self.axis_ratio)
            * (b / R) ** (slope - 1.0)
            * z
            * special.hyp2f1(1.0, 0.5 * slope, 2.0 - 0.5 * slope, -factor * z ** 2)
        )

        deflection_y = complex_angle.imag
        deflection_x = complex_angle.real

        rescale_factor = (self.ellipticity_rescale) ** (slope - 1)

        deflection_y *= rescale_factor
        deflection_x *= rescale_factor

        return self.rotate_grid_from_profile(np.vstack((deflection_y, deflection_x)).T)

    def convergence_func(self, grid_radius):
        if grid_radius > 0.0:
            return self.einstein_radius_rescaled * grid_radius ** (-(self.slope - 1))
        else:
            return np.inf

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (
            (eta_u / u)
            * ((3.0 - slope) * eta_u) ** -1.0
            * eta_u ** (3.0 - slope)
            / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)
        )


class SphericalPowerLaw(EllipticalPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents a spherical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super(SphericalPowerLaw, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            slope=slope,
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):

        eta = self.grid_to_grid_radii(grid)
        deflection_r = (
            2.0
            * self.einstein_radius_rescaled
            * np.divide(
                np.power(eta, (3.0 - self.slope)), np.multiply((3.0 - self.slope), eta)
            )
        )

        return self.grid_to_grid_cartesian(grid, deflection_r)


class EllipticalCoredIsothermal(EllipticalCoredPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        core_radius: dim.Length = 0.01,
    ):
        """
        Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        einstein_radius : float
            The arc-second Einstein radius.
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(EllipticalCoredIsothermal, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )


class SphericalCoredIsothermal(SphericalCoredPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        core_radius: dim.Length = 0.01,
    ):
        """
        Represents a cored spherical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(SphericalCoredIsothermal, self).__init__(
            centre=centre,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )


class EllipticalIsothermal(EllipticalPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
    ):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        einstein_radius : float
            The arc-second Einstein radius.
        """

        super(EllipticalIsothermal, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
        )

        if not isinstance(self, SphericalIsothermal) and self.axis_ratio > 0.99999:
            self.axis_ratio = 0.99999

    # @classmethod
    # def from_mass_in_solar_masses(cls, redshift_lens=0.5, redshift_source=1.0, centre: unit_label.Position = (0.0, 0.0), axis_ratio_=0.9,
    #                               phi: float = 0.0, mass=10e10):
    #
    #     return self.instance_kpc * self.angular_diameter_distance_of_plane_to_earth(j) / \
    #            (self.angular_diameter_distance_between_planes(i, j) *
    #             self.angular_diameter_distance_of_plane_to_earth(i))

    # critical_covnergence =

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore, \
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        factor = (
            2.0
            * self.einstein_radius_rescaled
            * self.axis_ratio
            / np.sqrt(1 - self.axis_ratio ** 2)
        )

        psi = np.sqrt(
            np.add(
                np.multiply(self.axis_ratio ** 2, np.square(grid[:, 1])),
                np.square(grid[:, 0]),
            )
        )

        deflection_y = np.arctanh(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 0]), psi)
        )
        deflection_x = np.arctan(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 1]), psi)
        )
        return self.rotate_grid_from_profile(
            np.multiply(factor, np.vstack((deflection_y, deflection_x)).T)
        )


class SphericalIsothermal(EllipticalIsothermal):
    @af.map_types
    def __init__(
        self, centre: dim.Position = (0.0, 0.0), einstein_radius: dim.Length = 1.0
    ):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        """
        super(SphericalIsothermal, self).__init__(
            centre=centre, elliptical_comps=(0.0, 0.0), einstein_radius=einstein_radius
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        eta = self.grid_to_elliptical_radii(grid)
        return 2.0 * self.einstein_radius_rescaled * eta

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        return self.grid_to_grid_cartesian(
            grid=grid,
            radius=np.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled),
        )
