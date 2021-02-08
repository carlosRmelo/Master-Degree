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

from os import path
dataset_path = "/home/carlos/Documents/GitHub/Master-Degree/ESO325/Dynamical Modelling/Emcee-JAM/Data"

#Reading data
y_px, x_px, vrms, erms = np.loadtxt('pPXF_rot_data.txt', unpack=True)                  #pPXF
surf_star_dat, sigma_star_dat, qstar_dat = np.loadtxt('JAM_Input.txt', unpack=True)    #photometry
surf_DM_dat, sigma_DM_dat, qDM_dat  = np.loadtxt('pseudo-DM Input.txt', unpack=True)   #DM

### Global Constantes

#Redshifth

z_galaxy = 0.035                                 #galaxy redshifth

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
distance = D_l                                              #Angular diameter distance [Mpc]
inc = 93.23                                                   #Inclination [deg]
mbh =  10**(7.85)*u.solMass                                        #Mass of black hole [M_sun]
beta0 = np.array([3.31, 1.53, 0.73, -3.83, 0.37, 0.10, 0.05])                #Anisotropy parameter, one for each gaussian component

ML0 = np.array([5.66      , 5.65483046, 5.63114305, 5.57277872, 5.50195229,
       5.49020144, 5.4902])*u.solMass/u.solLum       #Mass-to-light ratio per gaussian [M_sun/L_sun]


#DM
surf_DM_dat = surf_DM_dat*(u.solMass/u.pc**2)                          #Surface Density in M_sun/pc²
sigma_DM_dat_ARC = sigma_DM_dat*u.arcsec                               #Sigma in arcsec
sigma_DM_dat_PC = (sigma_DM_dat_ARC*D_l).to(u.pc, u.dimensionless_angles())    #Convert sigma in arcsec to sigma in pc
qDM_dat = np.ones_like(qDM_dat)*0.68                                                              #axial ratio of DM halo


#Stars
surf_star_dat = surf_star_dat*(u.solLum/u.pc**2)               #Surface luminosity Density in L_sun/pc²
sigma_star_dat_ARC = sigma_star_dat*u.arcsec                   #Sigma in arcsec
sigma_star_dat_PC = (sigma_star_dat_ARC*D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc
qstar_dat = qstar_dat                                          #axial ratio of star photometry



#----------------------------------------------------------------------------------------------------#


# JAMPY MODEL

#Defining some instrumental quantities and galaxy characteristics

muse_pixsize=0.6                                            #pixscale of IFU [arcsec/px]
muse_sigmapsf= 0.2420                                       ##Sigma of psf from MUSE [arcsec]

#Create model
Jampy_model = JAM(ybin=y_px, xbin=x_px,inc=inc, distance=distance.value, mbh=mbh.value,
                  rms=vrms, erms=erms, beta=beta0, sigmapsf=muse_sigmapsf, pixsize=muse_pixsize)

#Add Luminosity component
Jampy_model.luminosity_component(surf_lum=surf_star_dat.value, sigma_lum=sigma_star_dat_ARC.value,
                                    qobs_lum=qstar_dat, ml=ML0.value)

#Add Dark Matter component
Jampy_model.DM_component(surf_dm=10**(8.01)*surf_DM_dat.value, sigma_dm=sigma_DM_dat_ARC.value, qobs_dm=qDM_dat)

Jampy_model.run(plot=True, quiet=False, vmax=375, vmin=300)

plt.show()
