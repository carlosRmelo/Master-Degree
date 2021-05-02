""""
Author: Carlos Roberto de Melo
Date: 12/12/2020
Goal: This is a wrapper for JAM code by Cappellari (https://pypi.org/project/jampy/), specifically jam_axi_rms.

With this Class  script it is possible to construit a JAM model with and without dark matter, with constant or variant mass-light (ML) ratio. It is particularly useful if you want to run Jampy models with MCMC (e.g Emcee).
"""

#General packeges
import numpy as np
import matplotlib.pyplot as plt
from jampy.legacy.jam_axi_rms import jam_axi_rms
from plotbin.sauron_colormap import register_sauron_colormap
from plotbin.plot_velfield import plot_velfield
from jampy.legacy.jam_axi_rms import jam_axi_rms

class JAM():
    def __init__(
        self, ybin, xbin, inc, distance, mbh, rbh=0.01, rms=None, erms=None, beta=None,
             goodbins=None, tensor='zz', normpsf=1., sigmapsf=0., pixang=0., pixsize=0.):
            """
                Fundamental inputs for model.

                Parameters
                ----------


                INC:
                    inclination in degrees (90 being edge-on).
                MBH:
                    Mass of a nuclear supermassive black hole in solar masses.

                    VERY IMPORTANT: The model predictions are computed assuming SURF_POT
                    gives the total mass. In the common self-consistent case one has
                    ``SURF_POT = SURF_LUM`` and if requested (keyword ML) the program can
                    scale the output ``RMSMODEL`` to best fit the data. The scaling is
                    equivalent to multiplying *both* SURF_POT and MBH by a factor M/L.
                    To avoid mistakes, the actual MBH used by the output model is printed
                    on the screen.
                DISTANCE:
                    distance of the galaxy in Mpc.
                XBIN:
                    Vector of length ``P`` with the X coordinates in arcseconds of the bins
                    (or pixels) at which one wants to compute the model predictions. The
                    X-axis is assumed to coincide with the galaxy projected major axis. The
                    galaxy center is at (0,0).

                    When no PSF/pixel convolution is performed (``SIGMAPSF=0`` or
                    ``PIXSIZE=0``) there is a singularity at (0,0) which should be avoided
                    by the input coordinates.
                YBIN:
                    Vector of length ``P`` with the Y coordinates in arcseconds of the bins
                    (or pixels) at which one wants to compute the model predictions. The
                    Y-axis is assumed to coincide with the projected galaxy symmetry axis.

                Optional Keywords
                -----------------
                BETA:
                    Vector of length ``N`` with the anisotropy
                    ``beta_z = 1 - (sigma_z/sigma_R)**2`` of the individual MGE Gaussians.

                ERMS:
                    Vector of length ``P`` with the 1sigma errors associated to the ``RMS``
                    measurements. From the error propagation::

                        ERMS = sqrt((dVel*velBin)^2 + (dSig*sigBin)^2)/RMS,

                    where ``velBin`` and ``sigBin`` are the velocity and dispersion in each
                    bin and ``dVel`` and ``dSig`` are the corresponding errors.
                    (Default: constant errors ``ERMS=0.05*np.median(RMS)``)
                GOODBINS:
                    Boolean vector of length ``P`` with values True for the bins which
                    have to be included in the fit (if requested) and ``chi**2`` calculation.
                    (Default: fit all bins).
                NORMPSF:
                    Vector of length ``Q`` with the fraction of the total PSF flux
                    contained in the circular Gaussians describing the PSF of the
                    observations. It has to be ``np.sum(NORMPSF) = 1``. The PSF will be used
                    for seeing convolution of the model kinematics.
                PIXANG:
                    angle between the observed spaxels and the galaxy major axis X.
                PIXSIZE:
                    Size in arcseconds of the (square) spatial elements at which the
                    kinematics is obtained. This may correspond to the side of the spaxel
                    or lenslets of an integral-field spectrograph. This size is used to
                    compute the kernel for the seeing and aperture convolution.

                    If this is not set, or ``PIXSIZE=0``, then convolution is not performed.
                RBH:
                    This scalar gives the sigma in arcsec of the Gaussian representing the
                    central black hole of mass MBH (See Section 3.1.2 of `Cappellari 2008.
                    <http://adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_)
                    The gravitational potential is indistinguishable from a point source
                    for ``radii > 2*RBH``, so the default ``RBH=0.01`` arcsec is appropriate
                    in most current situations.

                    ``RBH`` should not be decreased unless actually needed!
                RMS:
                    Vector of length ``P`` with the input observed velocity second moment::

                        V_RMS = sqrt(velBin**2 + sigBin**2)

                    at the coordinates positions given by the vectors ``XBIN`` and ``YBIN``.

                    If ``RMS`` is set and ``ML`` is negative or not set, then the model is
                    fitted to the data, otherwise the adopted ML is used and just the
                    ``chi**2`` is returned.
                SIGMAPSF:
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
                TENSOR:
                    String specifying the component of the velocity dispersion tensor.

                    ``TENSOR='xx'`` gives sigma_xx=sqrt<V_x'^2> of the component of the
                    proper motion dispersion tensor in the direction parallel to the
                    projected major axis.

                    ``TENSOR='yy'`` gives sigma_yy=sqrt<V_y'^2> of the component of the
                    proper motion dispersion tensor in the direction parallel to the
                    projected symmetry axis.

                    ``TENSOR='zz'`` (default) gives the usual line-of-sight V_rms=sqrt<V_z'^2>.

                    ``TENSOR='xy'`` gives the mixed component <V_x'V_y'> of the proper
                    motion dispersion tensor.

                    ``TENSOR='xz'`` gives the mixed component <V_x'V_z'> of the proper
                    motion dispersion tensor.

                    ``TENSOR='yz'`` gives the mixed component <V_y'V_z'> of the proper
                    motion dispersion tensor.
            """
            

            
            self.ybin = ybin
            self.xbin = xbin
            self.inc = inc
            self.distance = distance
            self.mbh = mbh
            self.rbh = rbh
            self.rms = rms 
            self.erms = erms
            self.beta = beta
            self.goodbins = goodbins
            self.tensor = tensor
            self.normpsf = normpsf
            self.sigmapsf = sigmapsf
            self.pixang = pixang
            self.pixsize = pixsize


    def luminosity_component(self, surf_lum, sigma_lum, qobs_lum, ml=None):
        """
            Luminosity component from MGE parametrization
               Parameters
               -----------

                SURF_LUM:
                    vector of length ``N`` containing the peak surface brightness of the
                    MGE Gaussians describing the galaxy surface brightness in units of
                    ``Lsun/pc**2`` (solar luminosities per parsec**2).
                SIGMA_LUM:
                    vector of length ``N`` containing the dispersion in arcseconds of
                    the MGE Gaussians describing the galaxy surface brightness.
                QOBS_LUM:
                    vector of length ``N`` containing the observed axial ratio of the MGE
                    Gaussians describing the galaxy surface brightness.
                
                Optional Keywords
                -----------------
                ML:
                    Mass-to-light ratio to multiply the values given by SURF_POT.
                    Setting this keyword is completely equivalent to multiplying the
                    output ``RMSMODEL`` by ``np.sqrt(M/L)`` after the fit. This implies that
                    the BH mass becomes ``MBH*(M/L)``.

                    If this keyword is not set, or set to a negative number in input, the M/L
                    is fitted from the data and the best-fitting M/L is returned in output.
                    The BH mass of the best-fitting model is ``MBH*(M/L)``.

                    Also, you can give a ML vector of same length of surface luminosity, with each component give a correspond mass-to-light ratio. This should be used if you want a model with a variant ML.
  

        """
        

        self.surf_lum = surf_lum
        self.sigma_lum = sigma_lum
        self.qobs_lum = qobs_lum
        self.ml = ml

        assert surf_lum.size == sigma_lum.size == qobs_lum.size, "The luminous MGE components do not match"

        assert surf_lum.size == self.beta.size, "The beta components do not match"

    def DM_component(self, surf_dm = None, sigma_dm = None, qobs_dm= None):

        """
            Dark Matter component from MGE parametrization
               Parameters
               -----------

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
        """


        assert surf_dm.size == sigma_dm.size == qobs_dm.size, "The DM MGE components do not match"

        self.surf_dm = surf_dm
        self.sigma_dm = sigma_dm
        self.qobs_dm = qobs_dm


    def upt(self, surf_lum=None, sigma_lum=None, qobs_lum=None, ml=None,
                surf_dm=None, sigma_dm=None, qobs_dm=None,
                inc=None, mbh=None, beta=None):

        """
            Updates model parameters.  
            The input parameters are described in **__init__**, **luminosity_component** and **DM_component**, please check them in case of doubt. 
        """

        if surf_lum is not None:
            self.surf_lum = surf_lum
        if sigma_lum is not None:
            self.sigma_lum = sigma_lum
        if qobs_lum is not None:
            self.qobs_lum = qobs_lum
        if beta is not None:
            self.beta = beta

        assert self.surf_lum.size == self.sigma_lum.size == self.qobs_lum.size, "The luminous MGE components do not match"
        assert self.surf_lum.size == self.beta.size, "The beta components do not match"

        try:
            if surf_dm is not None:
                self.surf_dm = surf_dm
            if sigma_dm is not None:
                self.sigma_dm = sigma_dm
            if qobs_dm is not None:
               self.qobs_dm = qobs_dm

            assert self.surf_dm.size == self.sigma_dm.size == self.qobs_dm.size, "The DM MGE components do not match"
        except:
            if surf_dm is not None:
                raise ValueError("Dark Matter components were not properly initialized")
            else:
                pass
        

        if ml is not None:
            self.ml = ml
        if inc is not None:
            self.inc = inc
        if mbh is not None:
            self.mbh = mbh

    
    def run_simulation(self, flux_obs = None, nang=10, nrad=20, quiet = True, step=0.,
                 plot = False, vmax = None, vmin = None, nodots = True,
                    colorbar = True, linescolor = 'b', cmap=plt.cm.hot, **kwargs):
        """
            Run Jampy simulation.
            Execute the same as run.

            But for a simulation we don't have the rms data, so we are not able to compute chi2, for this reason run_simulation don't try to compute it.

            Parameters
            ----------

                SURF_POT:
                    vector of length ``M`` containing the peak value of the MGE Gaussians
                    describing the galaxy surface density in units of ``Msun/pc**2`` (solar
                    masses per parsec**2). This is the MGE model from which the model
                    potential is computed.

                    In a common usage scenario, with a self-consistent model, one has
                    the same Gaussians for both the surface brightness and the potential.
                    This implies ``SURF_POT = SURF_LUM``, ``SIGMA_POT = SIGMA_LUM`` and
                    ``QOBS_POT = QOBS_LUM``. The global M/L of the model is fitted by the
                    routine when passing the ``RMS`` and ``ERMS`` keywords with the observed
                    kinematics.
                SIGMA_POT:
                    vector of length ``M`` containing the dispersion in arcseconds of
                    the MGE Gaussians describing the galaxy surface density.
                QOBS_POT:
                    vector of length ``M`` containing the observed axial ratio of the MGE
                    Gaussians describing the galaxy surface density.
            
            Optinal Inputs
            --------------
                FLUX_OBS:
                     Optional mean surface brightness of each bin for plotting.  
                NANG:
                    Same as for ``NRAD``, but for the number of angular intervals
                    (default: ``NANG=10``).
                NRAD:
                    Number of logarithmically spaced radial positions for which the
                    models is evaluated before interpolation and PSF convolution. One may
                    want to increase this value if the model has to be evaluated over many
                    orders of magnitutes in radius (default: ``NRAD=20``). The computation
                    time scales as ``NRAD*NANG``.

                PLOT:
                    Set this keyword to produce a plot at the end of the calculation.
                QUIET:
                    Set this keyword to avoid printing values on the screen.

                STEP:
                    Spatial step for the model calculation and PSF convolution in arcsec.
                    This value is automatically computed by default as
                    ``STEP=MAX(SIGMAPSF,PIXSIZE/2)/4``. It is assumed that when ``PIXSIZE``
                    or ``SIGMAPSF`` are big, high resolution calculations are not needed. In
                    some cases however, e.g. to accurately estimate the central Vrms in a
                    very cuspy galaxy inside a large aperture, one may want to override the
                    default value to force smaller spatial pixels using this keyword.

            Output Parameters
            -----------------

                RMSMODEL:
                    Vector of length P with the model predictions for the velocity
                    second moments for each bin::

                        V_RMS = sqrt(vel**2 + sig**2)

                    Any of the six components of the symmetric proper motion dispersion
                    tensor can be provided in output using the ``TENSOR`` keyword.
                ML:
                    Best fitting M/L of the model.
                CHI2:
                    Reduced ``chi**2`` describing the quality of the fit::

                        chi2 = (((rms[goodBins] - rmsModel[goodBins])/erms[goodBins])**2).sum()
                            / goodBins.sum()

                FLUX:
                    Vector of length ``P`` with the PSF-convolved MGE surface brightness of
                    each bin in ``Lsun/pc**2``, used to plot the isophotes on the model results.

                CHI2*goodBins.sum():
                    Total chi2.
        """

        if np.isscalar(self.ml):
            ML = np.ones_like(self.surf_lum)*self.ml
            ml = 1.0
        elif self.ml is None:
            ML = np.ones_like(self.surf_lum)
            ml = None
        else:
            assert self.ml.size == self.surf_lum.size, "The ML components do not match"
            ML = self.ml
            ml = 1.0

        try:
            surf_pot = np.append(self.surf_lum*ML, self.surf_dm)
            sigma_pot = np.append(self.sigma_lum, self.sigma_dm)
            qobs_pot = np.append(self.qobs_lum, self.qobs_dm)
        except:
            surf_pot = self.surf_lum*ML
            sigma_pot = self.sigma_lum
            qobs_pot = self.qobs_lum


        rmsModel, ml_model, chi2, flux  = jam_axi_rms(
                         self.surf_lum, self.sigma_lum, self.qobs_lum,
                         surf_pot, sigma_pot, qobs_pot, self.inc, self.mbh, self.distance,
                         self.xbin, self.ybin, plot=plot, rms=self.rms, erms=self.erms, rbh=self.rbh,
                         sigmapsf=self.sigmapsf, normpsf=self.normpsf, goodbins=self.goodbins,
                         beta=self.beta, pixsize=self.pixsize, tensor=self.tensor, cmap=cmap,
                         colorbar=colorbar, linescolor=linescolor, ml=ml, nodots=nodots, quiet=quiet,
                         nrad=nrad, pixang=self.pixang, step=step, flux_obs=flux_obs, nang=nang,
                         vmax=vmax, vmin=vmin, **kwargs)

        return rmsModel, ml_model

   
    def run(self, flux_obs = None, nang=10, nrad=20, quiet = True, step=0.,
                 plot = False, vmax = None, vmin = None, nodots = True,
                    colorbar = True, linescolor = 'b', cmap=plt.cm.hot, **kwargs):
        """
            Run Jampy Rms Model.

            Parameters
            ----------

                SURF_POT:
                    vector of length ``M`` containing the peak value of the MGE Gaussians
                    describing the galaxy surface density in units of ``Msun/pc**2`` (solar
                    masses per parsec**2). This is the MGE model from which the model
                    potential is computed.

                    In a common usage scenario, with a self-consistent model, one has
                    the same Gaussians for both the surface brightness and the potential.
                    This implies ``SURF_POT = SURF_LUM``, ``SIGMA_POT = SIGMA_LUM`` and
                    ``QOBS_POT = QOBS_LUM``. The global M/L of the model is fitted by the
                    routine when passing the ``RMS`` and ``ERMS`` keywords with the observed
                    kinematics.
                SIGMA_POT:
                    vector of length ``M`` containing the dispersion in arcseconds of
                    the MGE Gaussians describing the galaxy surface density.
                QOBS_POT:
                    vector of length ``M`` containing the observed axial ratio of the MGE
                    Gaussians describing the galaxy surface density.
            
            Optinal Inputs
            --------------
                FLUX_OBS:
                     Optional mean surface brightness of each bin for plotting.  
                NANG:
                    Same as for ``NRAD``, but for the number of angular intervals
                    (default: ``NANG=10``).
                NRAD:
                    Number of logarithmically spaced radial positions for which the
                    models is evaluated before interpolation and PSF convolution. One may
                    want to increase this value if the model has to be evaluated over many
                    orders of magnitutes in radius (default: ``NRAD=20``). The computation
                    time scales as ``NRAD*NANG``.

                PLOT:
                    Set this keyword to produce a plot at the end of the calculation.
                QUIET:
                    Set this keyword to avoid printing values on the screen.

                STEP:
                    Spatial step for the model calculation and PSF convolution in arcsec.
                    This value is automatically computed by default as
                    ``STEP=MAX(SIGMAPSF,PIXSIZE/2)/4``. It is assumed that when ``PIXSIZE``
                    or ``SIGMAPSF`` are big, high resolution calculations are not needed. In
                    some cases however, e.g. to accurately estimate the central Vrms in a
                    very cuspy galaxy inside a large aperture, one may want to override the
                    default value to force smaller spatial pixels using this keyword.

            Output Parameters
            -----------------

                RMSMODEL:
                    Vector of length P with the model predictions for the velocity
                    second moments for each bin::

                        V_RMS = sqrt(vel**2 + sig**2)

                    Any of the six components of the symmetric proper motion dispersion
                    tensor can be provided in output using the ``TENSOR`` keyword.
                ML:
                    Best fitting M/L of the model.
                CHI2:
                    Reduced ``chi**2`` describing the quality of the fit::

                        chi2 = (((rms[goodBins] - rmsModel[goodBins])/erms[goodBins])**2).sum()
                            / goodBins.sum()

                FLUX:
                    Vector of length ``P`` with the PSF-convolved MGE surface brightness of
                    each bin in ``Lsun/pc**2``, used to plot the isophotes on the model results.

                CHI2*goodBins.sum():
                    Total chi2.
        """

        if np.isscalar(self.ml):
            ML = np.ones_like(self.surf_lum)*self.ml
            ml = 1.0
        elif self.ml is None:
            ML = np.ones_like(self.surf_lum)
            ml = None
        else:
            assert self.ml.size == self.surf_lum.size, "The ML components do not match"
            ML = self.ml
            ml = 1.0

        try:
            surf_pot = np.append(self.surf_lum*ML, self.surf_dm)
            sigma_pot = np.append(self.sigma_lum, self.sigma_dm)
            qobs_pot = np.append(self.qobs_lum, self.qobs_dm)
        except:
            surf_pot = self.surf_lum*ML
            sigma_pot = self.sigma_lum
            qobs_pot = self.qobs_lum


        rmsModel, ml_model, chi2, flux  = jam_axi_rms(
                         self.surf_lum, self.sigma_lum, self.qobs_lum,
                         surf_pot, sigma_pot, qobs_pot, self.inc, self.mbh, self.distance,
                         self.xbin, self.ybin, plot=plot, rms=self.rms, erms=self.erms, rbh=self.rbh,
                         sigmapsf=self.sigmapsf, normpsf=self.normpsf, goodbins=self.goodbins,
                         beta=self.beta, pixsize=self.pixsize, tensor=self.tensor, cmap=cmap,
                         colorbar=colorbar, linescolor=linescolor, ml=ml, nodots=nodots, quiet=quiet,
                         nrad=nrad, pixang=self.pixang, step=step, flux_obs=flux_obs, nang=nang,
                         vmax=vmax, vmin=vmin, **kwargs)
        
        if self.goodbins is None:
            goodbins = np.ones_like(self.rms, dtype=bool)
        else:
            goodbins = self.goodbins

        return rmsModel, ml_model, chi2, chi2*goodbins.sum()



