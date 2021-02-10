"""
Attention!
This code runs in MPI mode.
"""


#Control time packages
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"


#General packages
import numpy as np
from My_Jampy import JAM
import emcee
import matplotlib.pyplot as plt

#MPI
from schwimmbad import MPIPool

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.constants import G, M_sun, c
import astropy.units as u

#Autolens Model packages
import autolens as al
import autolens.plot as aplt

#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s



#Dataset path
dataset_path = "/home/carlos/Documents/GitHub/Master-Degree/Autolens_tests/autolens_workspace/Test_2/Data"

#Reading data of MGE and velocity dispersion maps
surf_lum, sigma_lum, qobs_lum = np.loadtxt("mge.txt", unpack=True)   #MGE data
xbin, ybin, goodbins, vrms  = np.loadtxt("vrms.txt", unpack=True)       #velocity dispersion map data

## Global informations and parameters
distance = 16.5 * u.Mpc                         #Lens galaxy distance [Mpc]

z = z_at_value(cosmo.angular_diameter_distance, distance, zmax=1.0) #Convert distance to redshifth 
z_lens = z                                    #Lens redshifth
z_source = 2.1                                #Source redshift

#Angular diameter distances
D_l = cosmo.angular_diameter_distance(z_lens)                   #Lens              
D_s = cosmo.angular_diameter_distance(z_source)                 #Source
D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)   #Lens to source

## Models inicialization

"""
    To inicialize the model, we set some random values for the parameters. But it's only necessary for initialize the model. During the non-linear search, this values will be updated constantly until the best fit.
"""  
#This quantities are our unknown parameters
inc = np.random.random()                    #Inclination [deg]
mbh = np.random.random()                    #Mass of black hole [M_sun]
beta = np.full_like(surf_lum, np.random.random()) #Anisotropy
ml = np.random.random()                     #Mass to light ratio [M_sun/L_sun]
mag_shear = np.random.random()              #Shear magnitude
phi_shear = np.random.random()              #Shear angle
gamma = np.random.random()                  #Gamma


#Now we define some quantities base on theses parameters

#Stars component for Jampy and Autolens
surf_star_dat = surf_lum                            #Surface luminosity Density in L_sun/pc²
sigma_star_dat_ARC = sigma_lum * u.arcsec           #Sigma in arcsec
sigma_star_dat_PC = (sigma_star_dat_ARC*D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc

     #After convertion, get only the values.   
sigma_star_dat_PC = sigma_star_dat_PC.value                 #Sigma of each gaussian [arcsec]
sigma_star_dat_ARC = sigma_star_dat_ARC.value               #Sigma of each gaussian [pc]
qobs_star_dat = qobs_lum                                    #Axial ratio of star photometry 

#Convert  surf_lum_sim to total mass per Guassian
Lum_star_dat = 2*np.pi*surf_star_dat*(sigma_star_dat_PC**2)*qobs_star_dat    #Total luminosity per gaussian component in L_sun

#Update the stellar mass based on M/L.
Mass_star = Lum_star_dat * ml                                  #Total star mass per gaussian in M_sun

#Inserting a Gaussian to represent SMBH at the center of the galaxy
sigmaBH_ARC = 0.01*u.arcsec
"""
        This scalar gives the sigma in arcsec of the Gaussian representing the
        central black hole of mass MBH (See Section 3.1.2 of `Cappellari 2008.
        <http://adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_)
        The gravitational potential is indistinguishable from a point source
        for ``radii > 2*RBH``, so the default ``RBH=0.01`` arcsec is appropriate
        in most current situations.

        ``RBH`` should not be decreased unless actually needed!
"""


sigmaBH_PC = (sigmaBH_ARC*D_l).to(u.pc, u.dimensionless_angles())        #Sigma of the SMBH in pc

    #After convertion, get only the values
sigmaBH_ARC = sigmaBH_ARC.value         #Sigma of gaussian BH [arcsec] 
sigmaBH_PC = sigmaBH_PC.value           #Sigma of gaussian BH [pc] 


surfBH_PC = mbh/(2*np.pi*sigmaBH_PC**2)                       #Mass surface density of SMBH [M_sun]
qSMBH = 1.                                                    #Assuming a circular gaussian
Mass_mbh = 2*np.pi*surfBH_PC*(sigmaBH_PC**2)*qSMBH            #SMBH Total mass 


Total_Mass = np.concatenate((Mass_star, Mass_mbh), axis=None)    #Mass per gaussian component in M_sun
Total_q = np.concatenate((qobs_star_dat, qSMBH), axis=None)      #Total axial ratio per gaussian


Total_sigma_ARC = np.concatenate((sigma_star_dat_ARC, sigmaBH_ARC), axis=None)  #Total sigma per gaussian in arcsec
Total_sigma_RAD = (Total_sigma_ARC * u.arcsec).to(u.rad)    #Total sigma per gaussian in radians
Total_sigma_RAD = Total_sigma_RAD.value                     #Only the value


#----------------------------------------------------------------------------------------------------#
# JAMPY MODEL

#Defining some instrumental quantities and galaxy characteristics

pixsize=0.8                                            #pixscale of IFU [arcsec/px]
normpsf=np.array([0.7, 0.3])                           #normalized intensity of IFU PSF
sigmapsf=np.array([0.6, 1.2])                          #sigma of each gaussian IFU PSF [arcsec]

#Create model
Jampy_model = JAM(ybin=ybin, xbin=xbin, inc=inc, distance=D_l.value, mbh=Mass_mbh,
                  rms=vrms, beta=beta, normpsf=normpsf, sigmapsf=sigmapsf, pixsize=pixsize)

#Add Luminosity component
Jampy_model.luminosity_component(surf_lum=surf_star_dat, sigma_lum=sigma_star_dat_ARC,
                                    qobs_lum=qobs_star_dat, ml=ml)

#----------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------#
# Pyautolens Model
#Reading fits file with the arcs data

#Imaging
pixel_scales = 0.1
imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        noise_map_path=f"{dataset_path}/noise.fits",
        psf_path=f"{dataset_path}/psf.fits",
        pixel_scales=pixel_scales,
    )

#Load mask
mask_custom = al.Mask.from_fits(
    file_path=f"{dataset_path}/mask.fits", hdu=0, pixel_scales=pixel_scales)
masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask_custom, inversion_uses_border=True)

#Plot
#aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask_custom, include=aplt.Include(border=False))

#Initializing the MGE model for the lens

mass_profile = al.mp.MGE(centre=(0.0, 0.0))                                #Mass model
mass_profile.MGE_comps(M=Total_Mass, sigma=Total_sigma_RAD,
                       q=Total_q, z_l=z_lens, z_s=z_source)               #Input parameters

mass_profile.MGE_Grid_parameters(masked_imaging.grid)
shear_comp = al.convert.shear_elliptical_comps_from(magnitude=mag_shear, phi=phi_shear) #external shear

lens_galaxy = al.Galaxy(
    redshift=z_lens,
    mass=mass_profile,
    shear=al.mp.ExternalShear(elliptical_comps=shear_comp)
)
#----------------------------------------------------------------------------------------------------#

#---------------------------------------- EMCEE -----------------------------------------------------#

