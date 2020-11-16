"""
Author: Carlos Roberto de Melo
    
Date: 11/08/2020

Obj: Parametrizar o perfil g-NFW apresentado em Collett et al. 2018. A parametrização é feita com gaussianas, de modo que possam ser adicionadas ao potencial do JAM mais tarde.
"""

from mgefit import mge_fit_1d
import numpy as np
import matplotlib.pyplot as plt

#Definindo algumas funções úteis.
SQRT_TOW_PI = np.sqrt(2*np.pi)


#Perfil gNFW

def gNFW(rho_0, alpha, r_s, r):
    rho = rho_0/((r**alpha)*(r_s**2 + r**2)**((3-alpha)/2))

    return rho 

#Realializa a deprojeção de 2d para 3d das gaussianas. Isso assume uma certa parametrização mge2d, um ângulo
    #de inclinação para deprojeção e um formato (oblate ou prolate)
def deprojection(mge2d, inc, shape):
        '''
        Return the 3D deprojected MGE coefficients
        '''
        mge3d = np.zeros_like(mge2d)
        if shape == 'oblate':
            qintr = mge2d[:, 2]**2 - np.cos(inc)**2
            if np.any(qintr <= 0):
                raise RuntimeError('Inclination too low q < 0')
            qintr = np.sqrt(qintr)/np.sin(inc)
            if np.any(qintr < 0.05):
                raise RuntimeError('q < 0.05 components')
            dens = mge2d[:, 0]*mge2d[:, 2] /\
                (mge2d[:, 1]*qintr*SQRT_TOW_PI)
            mge3d[:, 0] = dens
            mge3d[:, 1] = mge2d[:, 1]
            mge3d[:, 2] = qintr
        elif shape == 'prolate':
            qintr = np.sqrt(1.0/mge2d[:, 2]**2 -
                            np.cos(inc)**2)/np.sin(inc)
            if np.any(qintr > 10):
                raise RuntimeError('q > 10.0 conponents')
            sigmaintr = mge2d[:, 1]*mge2d[:, 2]
            dens = mge2d[:, 0] / (SQRT_TOW_PI*mge2d[:, 1] *
                                       mge2d[:, 2]**2*qintr)
            mge3d[:, 0] = dens
            mge3d[:, 1] = sigmaintr
            mge3d[:, 2] = qintr
        return mge3d


#Calcula a densidade luminosa de gaussianadas 3d para algum raio R
def luminosityDensity(mge3d, R, z):
        '''
        Return the luminosity density at coordinate R, z (in L_solar/pc^3)
        '''
        rst = 0.0
        ngauss = mge3d.shape[0]
        for i in range(ngauss):
            rst += mge3d[i, 0] * np.exp(-0.5/mge3d[i, 1]**2 *
                                        (R**2 + (z/mge3d[i, 2])**2))
        return rst


#Main code

r = np.logspace(np.log10(0.1), np.log10(10e3), 1500) #Cria valores espaçados em de um delta-log (requerido para o mge1d_fit). Unidade de pc
logr = np.log10(r)#Unidade de pc

#Definindo os parâmetros do gNFW

rho_0 = 1.0     #[M_sun]
alpha = 2.61    #[adm]
r_s = 1*10e3    #[pc]


#Calculando o perfil gNFW analítico e tomando o log para posterior plot
profile_DM = np.log10(gNFW(rho_0=rho_0, r_s=r_s, r=r, alpha=alpha))

#Agora vamos realizar a parametrização em MGE 1d
r_mge2d = np.logspace(np.log10(0.1), np.log10(10e3), 1500) #Cria valores espaçados em de um delta-log (requerido para o mge1d_fit)
rho_mge2d = gNFW(rho_0=rho_0, r_s=r_s, r=r_mge2d, alpha=alpha) #Perfil gNFW

#-------------Aqui é onde realmente começa a parametrização MGE--------------------------------

mge = mge_fit_1d.mge_fit_1d(r_mge2d, rho_mge2d, quiet=False, ngauss=100, plot=True)
mge_peak = mge.sol[0]         #Pico de cada gaussiana
mge_sigma = mge.sol[1]        #Sigma de cada gaussiana
#O resultado acima já está em M_sun/pc²

mge2d = np.zeros((len(mge_peak), 3)) #Agora iremos criar um array com o número de linhas igual ao número de
                                        #gaussianas e três colunas para armazenar os dados
mge2d[:, 0] = mge_peak     #Pico de cada gaussiana
mge2d[:, 1] = mge_sigma    #Sigma de cada gausiana
mge2d[:, 2] = 0.32         #qObs de cada gaussiana. 


plt.savefig('/home/carlos/autolens_workspace/JAM+Pyautolens/Images/gNFW_profile_Collett.png', fmt='png')


from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

z_lens = 0.035                                             #Lens redshift
D_l = cosmo.angular_diameter_distance(z_lens)              #Distance to lens in Mpc
D_l

np.savetxt('gProfile-DM Input.txt', np.column_stack([mge2d[:, 0], mge2d[:,1]/(D_l.value*np.pi/0.648),mge2d[:,2]]),
                            fmt=b'%e\t\t\t %e\t\t %10.6f', 
                            header='Surface Potential(M_sun/pc²) Sigma Potential(arcsec)    qObs')

#Aqui, quando salvamos o sigma de cada gaussiana dividimos pelo valor (D_l*np.pi/0.648) pois o JAM requer 
    #que o sigma seja dado em arcsec. Nesta expressão D_l é a distância da galáxia em Mpc
