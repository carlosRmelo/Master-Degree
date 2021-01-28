"""
    Run only the Model
"""


#General packages
import numpy as np
from My_Jampy import JAM
import matplotlib.pyplot as plt

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u

#Gaussian ML function
def gaussian_ml(sigma, delta, ml0=1.0, lower=0.0):
    '''
    Create a M*L gradient
    sigma: Gaussian sigma [arcsec]
    delta: Gradient value
    ml0: Central stellar mass to light ratio
    lower: the ratio between the central and the outer most M*/L
    '''

    sigma = np.atleast_1d(sigma)
    sigma = sigma - sigma[0]
    ML = ml0 * (lower + (1-lower)*np.exp(-0.5 * (sigma * delta)**2))
    
    return ML
#------------------------------------------------------------------------------------#

#Reading data
y_px, x_px, vrms, erms = np.loadtxt('pPXF_rot_data.txt', unpack=True)                  #pPXF
surf_star_dat, sigma_star_dat, qstar_dat = np.loadtxt('JAM_Input.txt', unpack=True)    #photometry
surf_DM_dat, sigma_DM_dat, qDM_dat  = np.loadtxt('pseudo-DM Input.txt', unpack=True)   #DM


muse_normpsf, muse_sigmapsf = np.loadtxt("MUSE_Psf_model.txt", unpack=True)             #Muse PSF

### Global Constantes

#readshift

z_galaxy = 0.299                                #galaxy redshifth

#Angular diameter distances
D_l = cosmo.angular_diameter_distance(z_galaxy)                       


#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s


### Global Parameters
"""
    To inicialize the model, we set some random values for the parameters. But it's only necessary for initialize the model. During the non-linear search, this values will be updated constantly until the best fit.
"""   

#Galaxy
distance = D_l    #Angular diameter distance [Mpc]
inc = 99.45                                                   #Inclination [deg]
mbh =  10**(8.51)*u.solMass                                        #Mass of black hole [M_sun]
beta0 = np.array([-0.05, 0.92, 0.96, -0.46, 0.02, 0.12, 0.62, -0.74])                         #Anisotropy parameter, one for each gaussian component
ML0 = gaussian_ml(sigma=sigma_star_dat, delta=1.03,
                     ml0=10.26, lower=0.80)*(u.solMass/u.solLum) #Gaussian Mass-to-light ratio [M_sun/L_sun]



#DM
surf_DM_dat = 10**(9.57)*(surf_DM_dat*(u.solMass/u.pc**2))                          #Surface Density in M_sun/pc²
sigma_DM_dat_ARC = sigma_DM_dat*u.arcsec                               #Sigma in arcsec
sigma_DM_dat_PC = (sigma_DM_dat_ARC*D_l).to(u.pc, u.dimensionless_angles())    #Convert sigma in arcsec to sigma in pc
qDM_dat = np.ones_like(qDM_dat)*0.60                                                             #axial ratio of DM halo


#Stars
surf_star_dat = surf_star_dat*(u.solLum/u.pc**2)               #Surface luminosity Density in L_sun/pc²
sigma_star_dat_ARC = sigma_star_dat*u.arcsec                   #Sigma in arcsec
sigma_star_dat_PC = (sigma_star_dat_ARC*D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc
qstar_dat = qstar_dat                                          #axial ratio of star photometry


#----------------------------------------------------------------------------------------------------#


# JAMPY MODEL

#Defining some instrumental quantities and galaxy characteristics

muse_pixsize=0.2                                            #pixscale of IFU [arcsec/px]
muse_normpsf=muse_normpsf                                   #normalized intensity of IFU PSF
muse_sigmapsf=muse_sigmapsf                                 #sigma of each gaussian IFU PSF [arcsec]

#Create model
Jampy_model = JAM(ybin=y_px, xbin=x_px,inc=inc, distance=distance.value, mbh=mbh.value,
                  rms=vrms, erms=erms, beta=beta0, normpsf=muse_normpsf, sigmapsf=muse_sigmapsf, pixsize=muse_pixsize)

#Add Luminosity component
Jampy_model.luminosity_component(surf_lum=surf_star_dat.value, sigma_lum=sigma_star_dat_ARC.value,
                                    qobs_lum=qstar_dat, ml=ML0.value)

#Add Dark Matter component
Jampy_model.DM_component(surf_dm=surf_DM_dat.value, sigma_dm=sigma_DM_dat_ARC.value, qobs_dm=qDM_dat)


Jampy_model.run(plot=True, quiet=False)
plt.show()
