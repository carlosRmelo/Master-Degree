# Dark Matter MGE decomposion
"""
    Perfom NFW dark matter decomposion using MGE formalism.
    Specially designed for gravitational lensing.
    TODO: Need to include references
"""

# Packges needed
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, double, jit                                 #Improve velocity (maybe)

from mgefit import mge_fit_1d                                       #MGE 1-d parametrization

from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from astropy import constants


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def NFW_profile(kappa_s, r_s, critical_density, r):
        """
        Function to sample the NFW profile. This evaluation is passed to the routine mge_fit_1d.
        Profile:
            (kappa_s * den_crit) / (r * (1 + r/r_s)**2 )
        """
        aux1 = np.multiply(kappa_s, critical_density)
        aux2 = np.multiply(r, np.power( np.add(1,  np.divide(r, r_s) ), 2) )


        return np.divide( aux1, aux2 )


class NFW(object):
    def __init__(self, z_l, z_s,
                        r_min, r_max, nsample, cosmology=None, quiet=True):
        """
        Classical NFW profile, usually used in gravitational lensing.
        For the MGE parametrization, it is only necessary the spherical model, because
        the ellipticity is introduced before the parametrization. 
        In the context of gravitational lensing, this profile is described by:
            \rho(r) = ( \kappa_s * \Sigma_crit )/ ( r * ( 1 + r/r_s)**2 )
        where:
            - \kappa_s [Ad]: Lens scale factor
                \kappa_s = ( \rho_s * r_s) / \Sigma_crit
            
            - \Sigma_crit [solMass/pc2]: (usual) Lens critial density

            - r_s [pc]: Scale radius

            - \rho_s [solMass/pc3]: Scale density
        """
        """
        Input:
        -------------------------
            z_l: int
                Lens Redshift
            z_s: int
                Source Redshift
            r_min: int [pc]
                Minimum radius where we want to sample the profile, in LOG SCALE. Zero should be avoid, because the
                parametrization will logsampling this radius.
            r_max: int [pc]
                Maximum radius where we want to sample the profile, in LOG SCALE. You probably will want to parameterize the profile at large scales, to capture all the features. This is **really** recommended.
            nsample: int
                Number of points where we want to sample your profile. This points will be in log scale.
                Check mge_fit_1d documentation for more information.
            cosmology: astropy cosmology
                Set cosmology used. Default is Planck15.
            quiet: bool
                Print some outputs during the initialization. Useful for check parameters.
        """
        
        if cosmology is not None:
            self.cosmology = cosmology
        else:
            self.cosmology = cosmo
        
        self.z_l = z_l
        self.z_s = z_s
        self.r   = np.logspace(r_min, r_max, nsample) #Radii [pc]


        # Angular diameter distances
        D_l  = self.cosmology.angular_diameter_distance(self.z_l) #Distance to lens
        D_s  = self.cosmology.angular_diameter_distance(self.z_s) #Distance to source
        D_ls = self.cosmology.angular_diameter_distance_z1z2(self.z_l, self.z_s) #Distance between them

        # Constante factor in lens critical density
        const = constants.c.to("Mpc / s") ** 2.0 / (
                    4 * np.pi * constants.G.to("Mpc3 / (" + "solMass" + " s2)"))
        # Lens Critical Density
        critical_density = (const * (D_s/(D_ls * D_l))).to(u.solMass/u.pc**2)

        if quiet is False:
            rmin_arc = self.r.min()/(D_l.value*np.pi/0.648)
            rmax_arc = self.r.max()/(D_l.value*np.pi/0.648)
            print(f"Critical Density:%.3e" %critical_density.value, critical_density.unit)
            print(f"At Redshift %.3f, your dark matter halo goes from %.3e arcsec to %.3e arcsec:"
                                %(self.z_l, rmin_arc, rmax_arc))


        self.critical_density = critical_density.value
        self.D_l  = D_l.value
        self.D_s  = D_s.value
        self.D_ls = D_ls.value 

                            

    def fit(self, kappa_s, rs, quiet=True, **kwargs):
        """ 
        MGE Fit 1d routine.
        Input:
        ------------------------------------------
        kappa_s: float [Ad]
            Lens scale factor
        rs: float [arcsec]
            Dark matter scale radius in arcsec
        quiet: bool
            Print deafult outputs of mge_fit_1d.
        **kwargs: dic
            Same parameters of mge_fit_1d. Our Default uses:
            - ngauss = 20
            - outer_slope = 2

            If the absolute error of the parametrization is > 10%, this routine adds 1 more gaussian component and fit the profile again, until the criteria be fulfilled. 
        Output:
        ---------------------------------------------
        surf_density: array [Msun/pc2]
            Peak of surface density of each gaussin.
        sigma: array [arcsec]
            Gaussian dispersion of each gaussian.

        """
        rs_pc = ( (rs*u.arcsec * self.D_l*u.Mpc ).to(u.pc,u.dimensionless_angles()) ).value #Scale radius [pc]


        if "ngauss" in kwargs:
            ngauss = kwargs["ngauss"]
            del kwargs["ngauss"]
        else:
            ngauss = 20

        if "outer_slope" in kwargs:
            outer_slope = kwargs["outer_slope"]
            del kwargs["outer_slope"]
        else:
            outer_slope = 2
        
        rho = NFW_profile(kappa_s, rs_pc, self.critical_density, self.r)


        if quiet is True:
            converged = False
            while converged is False:
                m = mge_fit_1d.mge_fit_1d(self.r, rho, ngauss=ngauss,
                                            outer_slope=outer_slope,
                                            plot=False, quiet=True,
                                            **kwargs)
                if np.any(np.abs(m.err) > 0.1):
                    print("Adding a gaussian and fitting again...")
                    ngauss = ngauss + 1  #Add a gaussian and try again
                else:
                    converged = True

        if quiet is False:
            converged = False
            while converged is False:
                print(f"The scale radius are %.3e pc"%rs_pc)
                m = mge_fit_1d.mge_fit_1d(self.r, rho, ngauss=ngauss,
                                            outer_slope=outer_slope,
                                            plot=True, quiet=False,
                                            **kwargs)
                plt.show()
                plt.close()
                if np.any(np.abs(m.err) > 0.1):
                    print("Adding a gaussian and fitting again...")
                    ngauss = ngauss + 1  #Add a gaussian and try again
                else:
                    converged = True
            
       #Converting quantities
        surf_density  =  m.sol[0]                           # Surface density in Msun/pc**2
        sigma         = m.sol[1]/(self.D_l*np.pi/0.648)     # Gaussian dispersion in arcsec

        return surf_density, sigma


