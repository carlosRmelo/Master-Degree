####### THIS IS NOT WORKING YET  #######

#Author: Carlo Roberto de Melo
#Date: 10/05/2020
#Obj:
'''
    Here we read all data necessary for both models (Jampy and Auto Lens). In summary, the data are:
        - kinematic map from Ppxf;
        - stars luminosity from MGE decomposition;
        - dark matter mass from 1d MGE decomposition.

    and all erros associted with this measurementes.
'''
####### THIS IS NOT WORKING YET  #######


#Packages to convert units and generate input arrays.

import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, M_sun, c
import astropy.units as u



#Data from photometry, DM halo and kinematic.

surf_star_dat, sigma_star_dat, qstar_dat = np.loadtxt('JAM Input.txt', unpack=True) #Star
surf_DM_dat, sigma_DM_dat, qDM_dat = np.loadtxt('pseudo-DM Input.txt', unpack=True) #DM
y_px, x_px, vel,  disp, chi, dV, dsigma = np.loadtxt('pPXF DATA.txt', unpack=True)  #pPXF

### Global Constantes

#Lens parameters

z_lens = 0.035
z_source = 2.1

D_l = cosmo.angular_diameter_distance(z_lens)
D_s = cosmo.angular_diameter_distance(z_source)
D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

#Useful constants
metre2Mpc = (1*u.m).to(u.Mpc)/u.m           #Constant factor to convert metre to Mpc.
kg2Msun = (1*u.kg/M_sun)*u.solMass/u.kg     #Constant factor to convert kg to Msun

G_Mpc = G*(metre2Mpc)**3/kg2Msun            #Gravitational constant in Mpc³/(Msun s²)
c_Mpc = c*metre2Mpc                         #Speed of light in Mpc/s


class Global_Parameters():
        """
            In this class we define all the global parameters  necessary to combined model. This class allow us get the data no matter where we are.
        """


    ### Global Parameters

        #To inicialize the model, we set an ML igual to 1 for every component in Star MGE.
            #But it's only necessary for initialize the model. 
            #During the non-linear search, this ML will be updated constantly until the best fit.
            #Same as above for the Anisotropy parameter in Jampy data.

        z_lens = 0.035                                               #lens redshift
        z_source = 2.1                                               #source redshift
            
        inc = 120.                                                    #Assumed galaxy inclination in deg                 
        distance = D_l                                                #Distance in Mpc
        mbh =  1e8*u.solMass                                          #Mass of SMBH in solar masses
        ML = np.ones(surf_star_dat.shape)*u.solMass/u.solLum          #Mass to light ratio per gaussian in M_sun/L_sun

        #DM
        surf_DM_dat = surf_DM_dat*(u.solMass/u.pc**2)                                    #Surface Density in M_sun/pc²
        sigma_DM_dat_ARC = sigma_DM_dat*u.arcsec                                         #Sigma in arcsec
        sigma_DM_dat_PC = (sigma_DM_dat_ARC*D_l).to(u.pc, u.dimensionless_angles())      #Convert sigma in arcsec to sigma in pc
        qDM_dat = qDM_dat                                                                #axial ratio of DM halo

        #Stars
        surf_star_dat = surf_star_dat*(u.solLum/u.pc**2)                                #Surface luminosity Density in L_sun/pc²
        sigma_star_dat_ARC = sigma_star_dat*u.arcsec                                    #Sigma in arcsec
        sigma_star_dat_PC = (sigma_star_dat_ARC*D_l).to(u.pc, u.dimensionless_angles()) #Convert sigma in arcsec to sigma in pc
        qstar_dat = qstar_dat                                                           #axial ratio of star photometry


class Jampy_data():
    """
    Here we define the Jampy data. As above, this allow us get the kinematic data no matter where we are.
    """
    #Defining some quantities of the instruments (IFU) and features of galaxy

    sigmapsf = 0.2420                                   #FWHM of PSF from IFU
    pixsize = 0.6                                       #pixel scale, in px/arcsec from kinematic (IFU)
    e = 0.24                                            #galaxy ellipticity. Value from find_my_galaxy


    #Selecting px where we want to compute the model

    x_good = []
    y_good = []
    disp_good = []
    vel_good = []
    dV_good = []
    dsigma_good = []

    for i in range(len(disp)):
        r = np.sqrt((x_px[i]*pixsize)**2 + ((y_px[i])*pixsize/(1-e))**2)
        if r < 5:
            x_good.append(x_px[i])
            y_good.append(y_px[i])
            disp_good.append(disp[i])
            vel_good.append(vel[i])
            dV_good.append(dV[i])
            dsigma_good.append(dsigma[i])

    #Calculating Vrms Speed
        #Note that we first identify the px with the highest velocity dispersion, in order to identify the center of the
        #galaxy. After that, we calculate the rotation speed with respect to that center. Only then can we
        #calculate the Vrms speed and the associated propagated erms error
    idx_max = np.where(np.array(disp_good) == max(disp_good))

    vel_good = vel_good - vel_good[idx_max[0][0]]
    vrms = np.sqrt(np.array(vel_good)**2 + np.array(disp_good)**2) #Vrms velocity
    erms = np.sqrt((np.array(dV_good)*np.array(vel_good))**2 + (np.array(dsigma_good)*np.array(disp_good))**2)/vrms #error in vrms

    #Defining the input data of the dynamic model

        #Position, in arcsec, where we will calculate the model
    xbin = np.array(x_good)*pixsize
    ybin = np.array(y_good)*pixsize

    r = np.sqrt(xbin**2 + (ybin/(1-e))**2)                  #Radius in the plane of the disk
    rms = vrms                                              #Vrms field in km/s
    erms = erms                                             #1-sigma erro na dispersão
    goodBins =    (r > 0)                                   #tells you which r values ​​are good for generating the model.
    beta = np.zeros(Global_Parameters.surf_star_dat.shape)  #Anisotropy parameter. One for each gaussian component. See Global_Parameters class for more details 


class Autolens_data():
    """
    Here we define the Autolens data. As above, this allow us get the lens data no matter where we are.
    """
    
    #Convert  surf_DM_dat to total mass per Guassian

    Mass_DM_dat = 2*np.pi*Global_Parameters.surf_DM_dat*(Global_Parameters.sigma_DM_dat_PC**2)*Global_Parameters.qDM_dat      #Total mass per gaussian component in M_sun

    #print("Total Mass per Gaussian component in DM profile:")
    #print(Mass_DM_dat)

    #Convert surf_star_dat to total Luminosity per Guassian and then to total mass per gaussian

    Lum_star_dat = 2*np.pi*Global_Parameters.surf_star_dat*(Global_Parameters.sigma_star_dat_PC**2)*Global_Parameters.qstar_dat    #Total luminosity per gaussian component in L_sun

    #print("Total Luminosity per Gaussian component of Stars:")
    #print(Lum_star_dat)

    #Update the stellar mass based on M/L.

    Mass_star_dat = Lum_star_dat*Global_Parameters.ML                          #Total star mass per gaussian in M_sun

    #print("Total Mass per Gaussian component of Star:")
    #print(Mass_star_dat)

    #Inserting a Gaussian to represent SMBH at the center of the galaxy

    sigmaBH_ARC = 0.01*u.arcsec
    '''
            This scalar gives the sigma in arcsec of the Gaussian representing the
            central black hole of mass MBH (See Section 3.1.2 of `Cappellari 2008.
            <http://adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_)
            The gravitational potential is indistinguishable from a point source
            for ``radii > 2*RBH``, so the default ``RBH=0.01`` arcsec is appropriate
            in most current situations.

            ``RBH`` should not be decreased unless actually needed!
        '''

    sigmaBH_PC = (sigmaBH_ARC*D_l).to(u.pc, u.dimensionless_angles())        #Sigma of the SMBH in pc
    surfBH_PC = Global_Parameters.mbh/(2*np.pi*sigmaBH_PC**2)                                  #Mass surface density of SMBH
    qSMBH = 1.                                                               #Assuming a circular gaussian
    Mass_SMBH_dat = 2*np.pi*surfBH_PC*(sigmaBH_PC**2)*qSMBH                  #SMBH Total mass 

    #print("Total Mass of SMBH")
    #print(Mass_SMBH_dat)

    #Defining the general inputs for the model
    i = np.deg2rad(Global_Parameters.inc)*u.rad                                                             #Inclination angle in rad
    Total_Mass = np.concatenate((Mass_star_dat, Mass_DM_dat, Mass_SMBH_dat), axis=None)   #Mass per gaussian component in M_sun
    Total_q = np.concatenate((Global_Parameters.qstar_dat, Global_Parameters.qDM_dat, qSMBH), axis=None)                      #Total axial ratio per gaussian

    Total_q_proj = np.sqrt(Total_q**2 - np.cos(i)**2)/np.sin(i)                                       #Total projected axial ratio per gaussian
    Total_sigma_ARC = np.concatenate((Global_Parameters.sigma_star_dat_ARC, Global_Parameters.sigma_DM_dat_ARC, sigmaBH_ARC), axis=None)  #Total sigma per gaussian in arcsec
    Total_sigma_RAD = Total_sigma_ARC.to(u.rad)                                                       #Total sigma per gaussian in radians

    #print("Total Mass per Gaussian of Model:")
    #print(Total_Mass)
    





