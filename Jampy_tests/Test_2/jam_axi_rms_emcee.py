"""
Attention!
This code makes use of MPI
"""
""""
The goal is reproduce Cappellari's jam_axi_rms_example.py making use of Emcee.
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
from jam_axi_rms_example import jam_axi_rms_example             #Cappellari's example

#MPI
from schwimmbad import MPIPool

#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u


#Generating the input parameters from Cappellari's original example.

(xbin, ybin, inc, rms, surf_lum, sigma_lum, qobs_lum, 
        distance, mbh, beta, sigmapsf, normpsf, pixsize, goodbins) = jam_axi_rms_example()


#Saving the input

np.savetxt("vrm.txt", np.column_stack([xbin, ybin, goodbins, rms]),
                      header="xbin \t\t ybin \t\t goodbins \t\t\t\t rms",
                      fmt=b"%e \t\t %e \t\t %e \t\t %e" )

np.savetxt("mge.txt", np.column_stack([surf_lum, sigma_lum, qobs_lum]),
                      header="surf_lum \t\t sigma_lum \t\t qobs_lum",
                      fmt=b"%e \t\t %e \t\t %e" )


parameters =[inc, distance, mbh, beta,sigmapsf, normpsf, pixsize]

import sys
with open("others_parameters.txt", 'w') as sys.stdout:
    print("Inclination [deg]:", inc)
    print("Distance [Mpc]:", distance)
    print("MBH [solar mass]:%e" %mbh)
    print("Anisotropy:", beta)
    print("Sigma PSF [arcsec]:", sigmapsf)
    print("Normpsf:", normpsf)
    print("Pixel size [arcsec]:", pixsize)


