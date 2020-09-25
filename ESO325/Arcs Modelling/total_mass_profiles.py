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

#------------------------------Bloco destinado a implementação do método MGE-------------------------------------------------------#

#Importando pacotes necessários#

from astropy.cosmology import Planck15
from astropy.constants import G, M_sun, c
import astropy.units as u
import scipy.integrate as integrate



from multiprocessing import Pool

from jampy.quadva import quadva

#Definindo algumas constantes úteis#

metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s

#Defindo os ângulos de deflexão do método MGE#
"""
Paper original: https://ui.adsabs.harvard.edu/abs/2012MNRAS.423.1073B/abstract
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



def alphax(tau1, 
            y, x, M, sigma, q):
    
    
    tau = tau1[:, None]
   
    x_til = x/sigma
    y_til = y/sigma

   
    
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(x_til/(np.sqrt(1 - eta_sq*(tau)**2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    

    
    arr = aux*exp    
    
    return tau1*arr.sum(1)



def alphay(tau1,
            y, x, M, sigma, q):
    

    tau = tau1[:, None]    
    x_til = x/sigma
    y_til = y/sigma
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(y_til/(np.power(1 - eta_sq*(tau)**2, 3/2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    
    
    arr = aux*exp
    
    return tau1*arr.sum(1)


    

#Defindo a Classe MGE#
class MGE(geometry_profiles.SphericalProfile, mp.MassProfile):
    @af.map_types
    def __init__(
        self, centre: dim.Position = (0.0, 0.0), processes: int = 1, method: str = "quadva", gamma: float = 1.0
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
            If quadva: uses quadva method by Cappellari.
            If sciquad: uses scipy.quad integration. In this case, if processes is set equal to one, serial integration is used.
                        Otherwise, deflection angle is calculated in parallel.
        gamma: float
            Is the Pos-Newtonian Parameter. If gamma is equal to one we recover General Relativity. Otherwise, we may have discovered something new.
        """


        super(MGE, self).__init__(
            centre=centre,
   
        )
        self.processes = processes
        self.method = method
        self.gamma = gamma


    def MGE_comps(self, M, sigma, q, z_l, z_s):
        """
        Parâmetros do modelo
        --------------------
        M: Ndim Array
            Array Ndim contendo a massa de cada gaussiana em M_sun.
        sigma: Ndim Array
            Array Ndim contendo a dispersão de cada gausiana em rad.
        q: Ndim Array
            Array Ndim contendo o axial ratio deprojetado de cada gaussiana. Adimensional
        z_l: float
            Redshift da lente
        z_s: float
            Redshift da fonte
        """

        self.M = M
        self.sigma = sigma
        self.q = q
        self.z_l = z_l
        self.z_s = z_s


        #Angular diametre distance
        D_l = Planck15.angular_diameter_distance(self.z_l)
        D_s = Planck15.angular_diameter_distance(self.z_s)
        D_ls = Planck15.angular_diameter_distance_z1z2(self.z_l, self.z_s)

        self.D_l = D_l
        self.D_s = D_s
        self.D_ls = D_ls

        #Constant Factor from integral in deflection angles
        critical_density = (c_Mpc**2/(4*np.pi*G_Mpc))*(D_s/(D_ls*D_l))
        const_factor = 1/(np.pi*critical_density*D_l**2)

        self.const_factor = const_factor.value



    def MGE_Grid_parameters(self, grid):
        """
            Cria um array com as seguintes propriedades:
            Cada linha do array terá 5 vetores representando os parâmetros necessários para o cálculo do ângulo de deflexão.
            O total de linhas é o total de pares de pontos (pxs) onde vamos calcular o modelo.
            As entradas são na ordem: y, x, M, sigma, q, onde y,x são escalares e M, sigma, q arrays de mesma dimensão.
            Supõe-se que a classe MGE.MGE_comps já foi iniciada.
            _____________________________________

            y: Array escalar
                Posição y do grid
            x: Array escalar
                Posição x do grid
            M: Ndim Array
                Array Ndim contendo a massa de cada gaussiana em M_sun.
            sigma: Ndim Array
                Array Ndim contendo a dispersão de cada gausiana em rad.
            q: Ndim Array
                Array Ndim contendo o axial ratio deprojetado de cada gaussiana. Adimensional
            grid: Ndim Array
                Array com dimensão (N,2) contendo os pontos (y,x) onde queremos calcular os ângulos de deflexão. Posições (y,x) em arcsec
        """

        #Primeiro é criado um grid genérico com as dimensões necessárias
        y0 = np.array([0])
        x0 = np.array([0])
        initial = np.array([y0, x0, self.M, self.sigma, self.q])
        Grid_parameters = np.array([y0, x0, self.M, self.sigma, self.q])

        #Agora realizamos um loop para criar todas as posições:
        for i in range(len(grid)-1): #-1 pois já começamos com uma posição (initial)
            Grid_parameters = np.vstack([Grid_parameters, initial])

        #Agora atualizamos as posições (y,x) e convertemos suas unidades para rad
        Grid_parameters[:, 0] = (grid[:, 0]*u.arcsec).to(u.rad).value
        Grid_parameters[:, 1] = (grid[:, 1]*u.arcsec).to(u.rad).value


        #Class parameter
        self.Grid_parameters = Grid_parameters

        return print("Pyautolens MGE Class successfully initialized!!")



    def MGE_Updt_parameters(self, M, sigma, q, gamma = 1.0):
        """
            Atualiza a grid de pâmetros do modelo. Particularmente útil quando utilizada em conjunto com Emcee.
            Assume-se que as classes MGE.MGE_comps e MGE.MGE_Grid_parameters já foram devidamente inicializadas.
            --------------------------------
            
            M: Ndim Array
                Array Ndim contendo a massa de cada gaussiana em M_sun.
            sigma: Ndim Array
                Array Ndim contendo a dispersão de cada gausiana em rad.
            q: Ndim Array
                Array Ndim contendo o axial ratio deprojetado de cada gaussiana. Adimensional
            gamma: float
                Is the Pos-Newtonian Parameter. If gamma is equal to one we recover General Relativity.
        """
        self.gamma = gamma
        for i in range(len(self.Grid_parameters)):
            self.Grid_parameters[i][2] = M
            self.Grid_parameters[i][3] = sigma
            self.Grid_parameters[i][4] = q
            self.M = M
            self.sigma = sigma
            self.q = q

        #return print("Parâmetros atualizados com sucesso!")

    
            

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
        #print('entrou')
        #print("Tamanho do grid", len(grid))
        if len(grid) == len(self.Grid_parameters):
            #print("Os grids são Iguais!")
            pass
        else:
            #print("Os grids de parâmetros e fonte são diferentes. Atualizando...")
            self.MGE_Grid_parameters(grid)
            #print("Novo tamanho da grid de parâmetros:", len(self.Grid_parameters))

            #Verify the integration method

        if self.method == "sciquad":
            print("Integration using scipy.quad")
            

                
            if self.processes != 1:
                #A integração será feita em paralelo usando o self.processes núcleos
                print('Integração em Paralelo. Número de núcleos é:', self.processes)
                #Deflexão em x
                pool = Pool(processes=self.processes)                           #Definindo o número de núcleos a serem usados
                result_x =  pool.map(integral_x, self.Grid_parameters)          #Passamos cada uma das linhas de parâmetros como argumento e chamamos a integral
                pool.close()                                                    #Fechamos o processo após ele terminar

                #Deflexão em y
                pool = Pool(processes=self.processes)                           #Definindo o números de núcleos a serem usados
                result_y =  pool.map(integral_y, self.Grid_parameters)          #Passamos cada uma das linhas de parâmetros como argumento e chamamos a integral
                pool.close()                                                    #Fechamos o processo após ele terminar

                result_x = np.array(result_x)
                result_y = np.array(result_y)

            else:
                #A integração será feita em série.
                print('Integração em série')
                result_x = np.zeros([len(grid), 2])              #Onde ficarão salvos os resultados da deflexão em x
                result_y = np.zeros([len(grid), 2])              #Onde ficarão salvos os resultados da deflexão em y

                for i in range(len(grid)):                                  #Começo do loop
                    result_x[i] = integral_x(self.Grid_parameters[i])       #Cálculo da integral em x
                    result_y[i] = integral_y(self.Grid_parameters[i])       #Cálculo da integral em y
                    
        elif self.method == "quadva":
            #print("Integração usando quadva")

            result_x = np.zeros([len(grid), 3])              #Onde ficarão salvos os resultados da deflexão em x
            result_y = np.zeros([len(grid), 3])              #Onde ficarão salvos os resultados da deflexão em y

         

            for i in range(len(grid)):                      #Começo do loop
                result_x[i] = quadva(alphax, [0., 1.], args=(self.Grid_parameters[i]),epsrel=1e-10)   #Integral em x
                result_y[i] = quadva(alphay, [0., 1.], args=(self.Grid_parameters[i]),epsrel=1e-10)   #Integral em y

        else:
            return print("Invalid integration method")


        #Atualizando a grid com o valor dos ângulos após deflexão
        grid[:, 0] = ((self.const_factor*result_y[:,0])*u.rad).to(u.arcsec).value
        grid[:, 1] = ((self.const_factor*result_x[:,0])*u.rad).to(u.arcsec).value
        
        
        
        return (0.5*(1.0 + self.gamma))*grid
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
