#MGE ESO325
#Obj: Perform the MGE for the galaxy ESO325-G004 in the HST f184W image
#data: 02/04/2020


#Import important packages

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from os import path

import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from mgefit.mge_fit_sectors_twist import mge_fit_sectors_twist
from mgefit.sectors_photometry_twist import sectors_photometry_twist
from mgefit.mge_print_contours_twist import mge_print_contours_twist
from mgefit.mge_fit_sectors_regularized import mge_fit_sectors_regularized as mge_fit_sectors


def dist_circle(xc, yc, s):
    """
    Returns an array in which the value of each element is its distance from
    a specified center. Useful for masking inside a circular aperture.

    The (xc, yc) coordinates are the ones one can read on the figure axes
    e.g. when plotting the result of my find_galaxy() procedure.

    """
    x, y = np.ogrid[:s[0], :s[1]] - np.array([yc, xc])  # note yc before xc
    rad = np.sqrt(x**2 + y**2)

    return rad


#Open the file

file = '/home/carlos/Desktop/ESO325 MGE and JAM/Data/F814w Interpolated.fits' #Caminho da Imagem
#file = '/home/carlos/Desktop/ESO325 HST Data Reduction/Data/F814w Interpolated.fits'
hdu = fits.open(file) #Abrindo imagem
img = hdu[1].data #Pega os dados da img


#Proporties of the image
skylev= 0.016   #counts/px
img -= skylev   #subtract sky
minlevel = 0.115  #counts/px
scale = 0.04    #arcsec/px
ngauss = 7     #number of desire gaussians



r = dist_circle(574, 69, img.shape)
mask = r > 20





#sigmapsf = [0.494, 1.44, 4.71, 13.4]      # In PC1 pixels
#normpsf = [0.294, 0.559, 0.0813, 0.0657]  # total(normpsf)=1
sigmapsf = 1.27
    # Here we use FIND_GALAXY directly inside the procedure. Usually you may want
    # to experiment with different values of the FRACTION keyword, before adopting
    # given values of Eps, Ang, Xc, Yc.
    
plt.clf()
f = find_galaxy(img, fraction=0.05, plot=1)
plt.savefig('/home/carlos/Desktop/ESO325 MGE and JAM/Images/Find My Galaxy.png', fmt='png')
plt.show()  # Allow plot to appear on the screen

    # Perform galaxy photometry
plt.clf()
s = sectors_photometry(img, f.eps, f.theta, f.xpeak, f.ypeak,
                           minlevel=minlevel, plot=1, mask=mask)
plt.savefig('/home/carlos/Desktop/ESO325 MGE and JAM/Images/Sectora Photometry.png', fmt='png')
plt.show()  # Allow plot to appear on the screen

    # Do the actual MGE fit
    # *********************** IMPORTANT ***********************************
    # For the final publication-quality MGE fit one should include the line
    # "from mge_fit_sectors_regularized import mge_fit_sectors_regularized"
    # at the top of this file, rename mge_fit_sectors() into
    # mge_fit_sectors_regularized() and re-run the procedure.
    # See the documentation of mge_fit_sectors_regularized for details.
    # *********************************************************************

plt.clf()
m = mge_fit_sectors(s.radius, s.angle, s.counts, f.eps,
                        ngauss=ngauss, sigmapsf=sigmapsf,
                        scale=scale, plot=1, bulge_disk=0, linear=0)
np.savetxt('MGE Output.txt', np.column_stack([m.sol[0], m.sol[1], m.sol[2]]),fmt=b'\t%10.6f\t %10.6f\t %10.6f\t ', header='\tTotal_Counts\t Sigma_Pixels\t     qObs\t', delimiter='\t')
plt.savefig('/home/carlos/Desktop/ESO325 MGE and JAM/Images/MGE.png', fmt='png')
plt.show()  # Allow plot to appear on the screen


    # Show contour plots of the results
plt.clf()
plt.subplot(121)
mge_print_contours(img.clip(minlevel), f.theta, f.xpeak, f.ypeak, m.sol, scale=scale,
                       binning=7, sigmapsf=sigmapsf, magrange=9, mask=mask)

    # Extract the central part of the image to plot at high resolution.
    # The MGE is centered to fractional pixel accuracy to ease visual comparson.

n = 50
img = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
plt.subplot(122)
mge_print_contours(img, f.theta, xc, yc, m.sol,
                      sigmapsf=sigmapsf, scale=scale)
plt.savefig('/home/carlos/Desktop/ESO325 MGE and JAM/Images/MGE Zoom.png', fmt='png')            
plt.show()  # Allow plot to appear on the screen


