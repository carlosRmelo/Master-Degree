#!/usr/bin/env python
# coding: utf-8

# In[1]:


# gaussian_beam_funcs.py
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift


# In[2]:


#Performs convolution using fft
def fft_conv(x,y,pix):
    return pix*pix*np.real(fftshift(ifft2((fft2(x))*(fft2(y)))))


# In[4]:


#Convert Gaussian beam paramters from d1, d2, t (semi-major, semi-minor, theta)
    #↪→to a, b, g (alpha, beta, gamma)
    
def convert_to_abg(parms_ddt):
    d1 = parms_ddt[0]
    d2 = parms_ddt[1]
    t = parms_ddt[2]
    m = (4*np.log(2))/(d1**2)
    n = (4*np.log(2))/(d2**2)
    a = m*(np.cos(t)**2) + n*(np.sin(t)**2)
    b = 2*(m-n)*np.sin(t)*np.cos(t)
    g = m*(np.sin(t)**2) + n*(np.cos(t)**2)
    return np.array([a, b, g])


# In[5]:


#Convert Gaussian beam paramters from a, b, g (alpha, beta, gamma) to d1, d2, t 
    #(↪→semi-major, semi-minor, theta)
def convert_to_ddt(parms_abg):
    a = parms_abg[0]
    b = parms_abg[1]
    g = parms_abg[2]
    d1 = np.sqrt((8*np.log(2))/(a + g - np.sqrt(a**2 - 2*a*g + g**2 + b**2)))
    d2 = np.sqrt((8*np.log(2))/(a + g + np.sqrt(a**2 - 2*a*g + g**2 + b**2)))
    t = 0.5*np.arctan2(-b,g-a)
    return np.array([d1, d2, t])


# In[6]:


#Calculates the abg parameters for a Fourier Transformed Gaussian
def fft_parms_abg(parms_abg):
    a = parms_abg[0]
    b = parms_abg[1]
    g = parms_abg[2]
    a_ft = (4*a*g*np.pi**2)/(4*(a**2)*g-a*(b**2))
    print(a_ft)
    b_ft = (-4*a*b*np.pi**2)/(4*(a**2)*g-a*(b**2))
    g_ft = (4*(a**2)*np.pi**2)/(4*(a**2)*g-a*(b**2))
    return np.array([a_ft, b_ft, g_ft])


# In[7]:


#Find the amplitude and ddt parameters for the correcting beam
def gaussian_deconvolve(new_beam_parms_ddt,old_beam_parms_ddt,A_new_beam,A_old_beam):
    nb_abg = convert_to_abg(new_beam_parms_ddt)
    ob_abg = convert_to_abg(old_beam_parms_ddt)
    ft_nb_abg = fft_parms_abg(nb_abg)
    ft_ob_abg = fft_parms_abg(ob_abg)
    corrected_parms_abg = fft_parms_abg(ft_nb_abg-ft_ob_abg)
    corrected_parms_ddt = convert_to_ddt(corrected_parms_abg)
    
    A_nb_ft = ft_amp(new_beam_parms_ddt, A_new_beam)
    A_ob_ft = ft_amp(old_beam_parms_ddt, A_old_beam)
    A_cor = ift_amp(corrected_parms_ddt, A_nb_ft/A_ob_ft)
    
    return A_cor, corrected_parms_ddt


# In[8]:


#Fourier transformed amplitude of beam
def ft_amp(beam_parms_ddt, A):
    return A*(np.pi*beam_parms_ddt[0]*beam_parms_ddt[1])/np.log(16)


# In[9]:


#Inverse Fourier transformed amplitude of beam
def ift_amp(beam_parms_ddt, A):
    return A*np.log(16)/(np.pi*beam_parms_ddt[0]*beam_parms_ddt[1])


# In[10]:


#Generating beams from parameters 
def gauss_beam_ddt(amp,parms_ddt,x,y):
### Values should be in rad
    d1 = parms_ddt[0]
    d2 = parms_ddt[1]
    t = parms_ddt[2]
    x_r = np.cos(t)*x + np.sin(t)*y
    y_r = -np.sin(t)*x + np.cos(t)*y
    r = 4*np.log(2)*(x_r**2)/(d1**2) + 4*np.log(2)*(y_r**2)/(d2**2)
    B = amp*np.exp(-r)
    return B


# In[ ]:




