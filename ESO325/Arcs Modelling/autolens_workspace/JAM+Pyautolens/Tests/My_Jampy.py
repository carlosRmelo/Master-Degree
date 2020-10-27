"""
Author: Carlos Roberto de Melo
Date: 09/16/2020
Goal: Build a python class based on Jampy (by Cappellari) to automate the dynamical modeling.
      In particular, making Lens Modelling + Dynamical Modelling easier.
"""


#Useful and needed packages

import numpy as np
import matplotlib.pyplot as plt
from jampy.legacy.jam_axi_rms import jam_axi_rms
from plotbin.sauron_colormap import register_sauron_colormap
from plotbin.plot_velfield import plot_velfield
from jampy.legacy.jam_axi_rms import jam_axi_rms





class Jam_axi_rms():
    def __init__(self, ybin, xbin, rms, erms, sigmapsf, pixsize, inc,
                     distance, mbh, surf_lum, sigma_lum, qobs_lum,
                     surf_DM=None, sigma_DM=None, qobs_DM=None,
                     beta=None, ml=None, goodBins=None, norm=None, quiet=True):
        """
        Class wrapper from jampy_axis_rms code by Michele Cappellari.
        Check his code for more details, and also the original papers:
            - Cappellari, M. (2020). Efficient solution of the anisotropic spherically aligned axisymmetric
                Jeans equations of stellar hydrodynamics for galactic dynamics. Monthly Notices of the Royal
                Astronomical Society, 494(4), 4819–4837. https://doi.org/10.1093/mnras/staa959
                
            - Cappellari, M. (2008). Measuring the inclination and mass-to-light ratio of axisymmetric galaxies
                via anisotropic Jeans models of stellar kinematics. Monthly Notices of the Royal Astronomical
                Society, 390(1), 71–86. https://doi.org/10.1111/j.1365-2966.2008.13754.x
    

        Parameters
        ----------

        ybin: Ndim array
            Vector of length ``P`` with the Y coordinates in arcseconds of the bins
            (or pixels) at which one wants to compute the model predictions. The
            Y-axis is assumed to coincide with the projected galaxy symmetry axis.

        xbin: Ndim array
             Vector of length ``P`` with the X coordinates in arcseconds of the bins
            (or pixels) at which one wants to compute the model predictions. The
            X-axis is assumed to coincide with the galaxy projected major axis. The
            galaxy center is at (0,0).

            When no PSF/pixel convolution is performed (``SIGMAPSF=0`` or
            ``PIXSIZE=0``) there is a singularity at (0,0) which should be avoided
            by the input coordinates.

        rms: Ndim array
            Vector of length ``P`` with the input observed velocity second moment::

            V_RMS = sqrt(velBin**2 + sigBin**2)

            at the coordinates positions given by the vectors ``XBIN`` and ``YBIN``.

            If ``RMS`` is set and ``ML`` is negative or not set, then the model is
            fitted to the data, otherwise the adopted ML is used and just the
            ``chi**2`` is returned.

        erms: Ndim array
            Vector of length ``P`` with the 1sigma errors associated to the ``RMS``
            measurements. From the error propagation::

            ERMS = sqrt((dVel*velBin)^2 + (dSig*sigBin)^2)/RMS,

            where ``velBin`` and ``sigBin`` are the velocity and dispersion in each
            bin and ``dVel`` and ``dSig`` are the corresponding errors.
            (Default: constant errors ``ERMS=0.05*np.median(RMS)``)

        sigmapsf: Float or Ndim array
            Vector of length ``Q`` with the dispersion in arcseconds of the
            circular Gaussians describing the PSF of the observations.

            If this is not set, or ``SIGMAPSF=0``, then convolution is not performed.

            IMPORTANT: PSF convolution is done by creating a 2D image, with pixels
            size given by ``STEP=MAX(SIGMAPSF, PIXSIZE/2)/4``, and convolving it
            with the PSF + aperture. If the input radii RAD are very large with
            respect to STEP, the 2D image may require a too large amount of memory.
            If this is the case one may compute the model predictions at small radii
            separately from those at large radii, where PSF convolution is not
            needed.

            If is set as a float, generally, is used the FWHM of the PSF model, in arcsec.

        normpsf: Ndim array
        (only needed if sigmapsf is set as an array)
            Vector of length ``Q`` with the fraction of the total PSF flux
            contained in the circular Gaussians describing the PSF of the
            observations. It has to be ``np.sum(NORMPSF) = 1``. The PSF will be used
            for seeing convolution of the model kinematics.

        pixsize: Float
            Size in arcseconds of the (square) spatial elements at which the
            kinematics is obtained. This may correspond to the side of the spaxel
            or lenslets of an integral-field spectrograph. This size is used to
            compute the kernel for the seeing and aperture convolution.

            If this is not set, or ``PIXSIZE=0``, then convolution is not performed.

        inc: Float
             Inclination in degrees (90 being edge-on).

        distance: Float
            Distance of the galaxy in Mpc.

        mbh: Float
            Mass of a nuclear supermassive black hole in solar masses.

            VERY IMPORTANT: The model predictions are computed assuming SURF_POT
            gives the total mass. In the common self-consistent case one has
            ``SURF_POT = SURF_LUM`` and if requested (keyword ML) the program can
            scale the output ``RMSMODEL`` to best fit the data. The scaling is
            equivalent to multiplying *both* SURF_POT and MBH by a factor M/L.
            To avoid mistakes, the actual MBH used by the output model is printed
            on the screen.

        surf_lum: Ndim array
            Vector of length ``N`` containing the peak surface brightness of the
            MGE Gaussians describing the galaxy surface brightness in units of
            ``Lsun/pc**2`` (solar luminosities per parsec**2).
            
        sigma_lum:
            vector of length ``N`` containing the dispersion in arcseconds of
            the MGE Gaussians describing the galaxy surface brightness.
            
        qobs_lum:
            vector of length ``N`` containing the observed axial ratio of the MGE
            Gaussians describing the galaxy surface brightness.


        Optional Keywords:
        ------------------

        surf_DM: Ndim array
            Vector of length ``M`` containing the peak surface mass density of the
            MGE Gaussians describing the dark matter density halo profile in units of
            ``Msun/pc**2`` (solar masses per parsec**2).

            Usually, an arbitrary profile for dark matter is parametrized by the MGE method.
            Then the peak of each gaussian is convert to mass projected density. This dark matter
            mass density profile (and the mbh) will be appended to stellar mass to generate the final
            gravitational potential.

            In a common usage scenario, with a self-consistent model, the DM contribution is neglected,
            and the gravitational potential will be generated only by the mbh and stars.

        sigma_DM: Ndim array
            vector of length ``M`` containing the dispersion in arcseconds of
            the MGE Gaussians describing the dark matter profile.

        qobs_lum: Ndim array
            vector of length ``M`` containing the observed axial ratio of the MGE
            Gaussians describing the dark matter profile.

        beta: Ndim array
            Vector of length ``N`` with the anisotropy
            ``beta_z = 1 - (sigma_z/sigma_R)**2`` of the individual MGE Gaussians.

        ml: Floar or Ndim array
            Mass-to-light ratio to multiply surf_lum to obtain the mass density of stars,
            and then append to total gravitacional potencial.

            If is set as a Float, a constant mass-to-light ratio is assumed for each gaussian
            in surf_lum. The other option is give a vector of  length  "N" where each input
            correspond a M/L for each gaussian in surf_lum. 

            If this keyword is not set, or set to a negative number in input, the M/L
            is fitted from the data and the best-fitting M/L is returned in output.
            The BH mass of the best-fitting model is ``MBH*(M/L)``

        goodBins: Ndim array
            Boolean vector of length ``P`` with values True for the bins which
            have to be included in the fit (if requested) and ``chi**2`` calculation.
            (Default: fit all bins).
    
    """

            #Making all variables global variables

        super(Jam_axi_rms, self).__init__(
   
        )
 
        self.ybin = ybin
        self.xbin = xbin
        self.rms = rms
        self.erms = erms
        self.sigmapsf = sigmapsf
        self.pixsize = pixsize
        self.inc = inc
        self.distance = distance
        self.mbh = mbh
        self.surf_lum = surf_lum
        self.sigma_lum = sigma_lum
        self.qobs_lum = qobs_lum

        self.surf_DM = surf_DM
        self.sigma_DM = sigma_DM
        self.qobs_DM = qobs_DM
        self.beta = beta
        self.ml = ml
        self.goodBins = goodBins
        self.norm = norm

        if quiet is not True:
        	print("Jampy Class successfully initialized!!")

    def Updt_parameters(self, surf_DM, qobs_DM, ml, beta, mbh, inc):

        """
            Update parameters from the Emcee search.
        """

        self.surf_DM = surf_DM
        self.qobs_DM = qobs_DM
        self.ml = ml
        self.beta = beta
        self.mbh = mbh
        self.inc = inc


    def run(self, plot=False, colorbar=True, linescolor='b', nodots=True, quiet=True, tensor='zz'):
        
        
        """
            Here where the fit actually occurs.

            Parameters:
            -----------

            Potential:

                Neste bloco começamos a definir as componentes do potencial gravitacional. Em um cenário auto-consistente,
                surf_pot = surf_lum, sigma_pot = sigma_lum, qobs_pot = qObs_lum. Mas pode-se adicionar uma componente de 
                matéria escura, de modo que as componentes do potencial terão as gaussianas da fotometria + gaussianas do DM.
                Neste segundo caso, é necessário transformar os valores de luminosidade das surf_lum em valores de massa,
                multiplicando cada gaussiana da surf_lum por um valor de ml. Após esse passo, podemos adicionar as componentes
                de DM ao surf_pot. Quando é setado uma ml anterior, é necessário informar a função jam_axi_rms que ml=1. Isso
                é necessário pois, caso não seja dado um ml ao chamar a função, ela irá calcular o melhor ml e escalonar a
                solução da Vrms com base neste valor. Por outro lado, se ao chamar a função setamos ml=1, ela usa esse valor 1
                para escalonar a solução, o que na prática não muda os resultados.
        """

         #Check the M/L

        if self.ml is not None:
            
            
            surf_pot = self.surf_lum*self.ml                        #Converting surface luminosity to mass surface density and append to potential. [M_sun/pc^2]
            sigma_pot = self.sigma_lum               
            qobs_pot = self.qobs_lum

            surf_pot = np.append(surf_pot, self.surf_DM)            #Appending DM contribution to potential.
            sigma_pot = np.append(sigma_pot, self.sigma_DM)
            qobs_pot = np.append(qobs_pot, self.qobs_DM)

            rmsModel, ml, chi2, flux = jam_axi_rms(self.surf_lum, self.sigma_lum, self.qobs_lum, surf_pot, sigma_pot, qobs_pot,
                                                        self.inc, self.mbh, self.distance, self.xbin, self.ybin, plot=plot, rms=self.rms, erms=self.erms,
                                                        sigmapsf=self.sigmapsf, goodbins=self.goodBins, beta=self.beta, pixsize=self.pixsize, tensor=tensor,
                                                        cmap=plt.cm.hot, colorbar=colorbar, linescolor=linescolor, ml=1.0, nodots=nodots, quiet=quiet)


    
            return rmsModel, ml, chi2, chi2*self.goodBins.sum()


        

