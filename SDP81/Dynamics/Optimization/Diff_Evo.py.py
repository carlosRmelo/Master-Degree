#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Control time packages
import time
import os


#General packages
import numpy as np
from My_Jampy import JAM
import emcee
import matplotlib.pyplot as plt


#Constants and usefull packages
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u

#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s


#Dataset path
data_folder = "/home/carlos/Documents/GitHub/Master-Degree/SDP81/Dynamics/Emcee/Data/"


# In[2]:


#Reading MGE inputs
    #attention to units
surf_lum, sigma_lum, qobs_lum = np.loadtxt(data_folder+"JAM_Input.txt", unpack=True)          #MGE decomposition
surf_dm, sigma_dm , qobs_dm   = np.loadtxt(data_folder+"SDP81_pseudo-DM_halo.txt", unpack=True)    #DM component
norm_psf, sigma_psf           = np.loadtxt(data_folder+"MUSE_Psf_model.txt", unpack=True)     #PSF
ybin, xbin, vrms, erms        = np.loadtxt(data_folder+"pPXF_rot_data.txt", unpack=True)          #Vrms data

muse_pixsize = 0.2                            #Muse pixel size [arcsec/px]

z_lens   = 0.299                                    #Lens redshifth
z_source = 3.100                                    #Source redshift

#Angular diameter distances
D_l = cosmo.angular_diameter_distance(z_lens)                   #Lens              
D_s = cosmo.angular_diameter_distance(z_source)                 #Source
D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)   #Lens to source


#This quantities are our unknown parameters
inc       = 75                              #Inclination [deg]
mbh       = 1e10                            #Mass of black hole [M_sun]
beta      = np.full_like(surf_lum, 0.3)     #Anisotropy
ml        = 10                              #Mass to light ratio [M_sun/L_sun]
rho_s     = 1e10                            #dark matter intensity
qdm       = np.full_like(qobs_dm, 0.5)      #dark matter axial ratio
    

#--------------------------------------------------------------------------------------------------#
# JAMPY MODEL

Jam_model = JAM(ybin=ybin, xbin=xbin, inc=inc, distance=D_l.value, mbh=mbh, beta=beta, rms=vrms,
                erms=erms,normpsf=norm_psf, sigmapsf=sigma_psf*muse_pixsize, pixsize=muse_pixsize)

    #Add Luminosity component
Jam_model.luminosity_component(surf_lum=surf_lum, sigma_lum=sigma_lum,
                                qobs_lum=qobs_lum, ml=ml)
    #Add DM component
Jam_model.DM_component(surf_dm=rho_s * surf_dm, sigma_dm=sigma_dm, qobs_dm=qdm)


# In[14]:


#--------------------------------------- EMCEE ------------------------------------------------------#

### Priors

###boundaries. [lower, upper]
boundary = {'qinc': [0.0501, np.min(qobs_lum)], 'beta': [-5, 5], 'ml': [0.5, 15], 'log_mbh':[5, 11],
                'log_rho': [3, 11], 'qDM': [0.4, 1]}





def check_boundary(parsDic):
    """
        Check whether parameters are within the boundary limits
        input
            parsDic: parameter dictionary {'paraName', value}
        output
            -np.inf or 0.0
    """   

    #Check if beta is ok

    #Avoid beta[i] == 1, because this could cause problems
    for i in range(len(parsDic['beta'])):
        if parsDic['beta'][i] != 1:
            pass
        else:
            return -np.inf

    #Check beta boundary
    for i in range(len(parsDic['beta'])):
        if boundary['beta'][0] < parsDic['beta'][i] < boundary['beta'][1] :
            pass
        else:
            return -np.inf




    #Check if the others parameters are within the boundary limits
    keys = set(parsDic.keys())
    excludes = set(['beta'])  #Exclude beta and ml, because we already verify above

    
    for keys in keys.difference(excludes):
        if boundary[keys][0] <= parsDic[keys] <= boundary[keys][1]:
            pass
        else:
            return -np.inf
    return 0.0

def log_prior(parsDic):
    '''
    Calculate the prior lnprob
    input
      parsDic: parameter dictionary {'paraName', value}
    output
      lnprob
    '''
    
    rst = 0
    """
    The lines above is only for doble check, because we are assuming flat prior for all parameters, Once we got here, all values ​​have already been accepted, so just return 0.0 for one of them
    """

    rst += 0.0     #mass-to-light
    rst += 0.0     #beta
    rst += 0.0     #qinc
    rst += 0.0     #log_mbh
    rst += 0.0     #log_rho
    rst += 0.0     #qDM

    
    return rst


def Updt_JAM(parsDic):
    '''
       Update the dynamical mass model
    input
      parsDic: parameter dictionary {'paraName', value}
    '''

        # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc_model = np.degrees(np.arctan(np.sqrt((1 - qmin**2)/(qmin**2 - parsDic['qinc']**2))))
   

    ml_model  = parsDic['ml']                        #mass-to-light update 
    mbh_model = 10**parsDic['log_mbh']               #BH mass
    rho_model = 10**parsDic['log_rho']               #rho DM
    qdm_proj  = np.sqrt( (np.sin(np.radians(inc_model)) * parsDic['qDM'] )**2  + np.cos( np.radians(inc_model))**2     )                            #Projected DM axial ratio
    qdm_model = np.full_like(surf_dm, qdm_proj)      #qDM

    #We project the DM axial ratio because during the JAM fit it will be deprojected.

    #Model Updt
    Jam_model.upt(inc=inc_model, ml=ml_model, beta=parsDic['beta'], mbh=mbh_model,
                        surf_dm= rho_model * surf_dm, qobs_dm=qdm_model)


def JAM_log_likelihood(parsDic):
    """
        Perform JAM modeling and return the chi2
    """
    Updt_JAM(parsDic)               #Updt values for each iteration
    
    rmsModel, ml, chi2, chi2T = Jam_model.run()
    plt.show()
    return -0.5 * chi2T


def log_probability(pars):
    """
        Log-probability function for whole model.
        input:
            pars: current values in the Emcee sample.
        output:
            log probability for the combined model.
    """

    (ml, b1, b2, b3, b4, b5, b6, b7, b8, qinc, log_mbh, log_rho, qdm) = pars
    
    beta =  np.array([b1, b2, b3, b4, b5, b6, b7, b8])


    parsDic = {'ml': ml, 'beta': beta, 'qinc': qinc, 'log_mbh': log_mbh,
                        'log_rho': log_rho, 'qDM': qdm}

    
    #Checking boundaries
    if not np.isfinite(check_boundary(parsDic)):
        return -np.inf
    #calculating the log_priors
    lp = log_prior(parsDic)

    return lp + JAM_log_likelihood(parsDic)


# In[18]:


def call_model(x0):
    #print(x0)
    value =  log_probability(x0)
    #print(value)
    if np.isfinite(value):
        return -value
    else:
        return -1e200


# In[19]:


from scipy.optimize import differential_evolution
#In order: ML, b1, b2, b3, b4, b5, b6, b7, qinc, log_mbh, log_rho_s, qDM
bounds = [(0.5, 15), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), 
              (0.05, np.min(qobs_lum)), (6, 12), (6, 12), (0.4, 1)]


# In[20]:


result = differential_evolution(call_model,maxiter=5000, bounds=bounds, workers=20, updating='deferred', disp=True)


# In[ ]:




