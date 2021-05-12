#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.environ["OMP_NUM_THREADS"] = "1"


import autolens as al
import autolens.plot as aplt

import numpy as np

from time import perf_counter as clock

from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

import emcee
from multiprocessing import Pool
from multiprocessing import Process

import time

data_folder = "/home/carlos/Documents/GitHub/Master-Degree/Autolens tests/autolens_workspace/Test_6/Simulation_Data/"


# In[2]:



#Only for lensing modelling 
z_l    = 0.299                                                         #Lens Redshift
z_s    = 4.100                                                         #Source Redshift 
kappa_ = 1.1                                                         #kappa_s of DM profile
r_s    = 2.0                                                         #scale radius
q      = 0.85                                                          #Axis Ratio
shear_comp = al.convert.shear_elliptical_comps_from(magnitude=0.02, phi=88) #external shear


# In[3]:


imaging = al.Imaging.from_fits(
		image_path=f"{data_folder}/arcs_simulation.fits",
		noise_map_path=f"{data_folder}/noise_simulation.fits",
		psf_path=f"{data_folder}/psf_simulation.fits",
		pixel_scales=0.1,
	)

mask        = al.Mask.from_fits( file_path=f"{data_folder}/new_mask.fits", hdu=1, 
									pixel_scales=imaging.pixel_scales)#You should check Mask_Maker folder before continue

masked_image = al.MaskedImaging(imaging=imaging, mask=mask, inversion_uses_border=True)     #Masked image


#aplt.Imaging.subplot_imaging(
#	imaging=imaging, mask=mask, include=aplt.Include(border=True),                         #Plot
#)


# In[4]:



#Initializing
ell_comps = al.convert.elliptical_comps_from(axis_ratio=q, phi=0.0) #Elliptical components in Pyautolens units
mass_profile = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=kappa_, elliptical_comps=ell_comps, scale_radius=r_s)

#Lens galaxy
lens_galaxy = al.Galaxy(
	redshift=z_l,
	mass=mass_profile,
	shear=al.mp.ExternalShear(elliptical_comps=shear_comp)
)


# In[5]:


source_galaxy = al.Galaxy(
	redshift=z_s,
	pixelization=al.pix.Rectangular(shape=(40, 40)),
	regularization=al.reg.Constant(coefficient=2.0),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(masked_imaging=masked_image, tracer=tracer)

#aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True,critical_curves=False,caustics=False))
print("Log Likelihood with Regularization:", fit.log_likelihood_with_regularization)


# In[6]:


boundary = {"kappa_s": [0, 3], "r_s": [0, 10], "q": [0.2, 1], "mag_shear": [0, 0.1], "phi_shear": [0, 179]}
def check_boundaries(parsDics):
	for key in parsDics:
		if boundary[key][0] <= parsDics[key] <= boundary[key][1]:
			pass
		else:
			return - np.inf
	return 0.0

def fit_model(parsDics):
	ell_model  = al.convert.elliptical_comps_from(axis_ratio=parsDics["q"],
												phi=0.0) #Elliptical components in Pyautolens units
	mass_model = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=parsDics["kappa_s"], 
														elliptical_comps=ell_model, 
														scale_radius=parsDics["r_s"])#Mass model
	shear_model = al.convert.shear_elliptical_comps_from(magnitude=parsDics["mag_shear"],
														phi=parsDics["phi_shear"]) #external shear

	#Lens galaxy
	lens_model = al.Galaxy(
		redshift=z_l,
		mass=mass_model,
		shear=al.mp.ExternalShear(elliptical_comps=shear_model)
	)
	tracer_model = al.Tracer.from_galaxies(galaxies=[lens_model, source_galaxy])

	fit_model    = al.FitImaging(masked_imaging=masked_image, tracer=tracer_model)
	
	return fit_model.log_likelihood_with_regularization

def log_likelihood(p0):
	kappa_s, r_s, q, mag_shear, phi_shear = p0
	
	parsDics = {"kappa_s": kappa_s, "r_s": r_s, "q": q,
					"mag_shear": mag_shear, "phi_shear": phi_shear}
	#print(parsDics)
	if np.isinf(check_boundaries(parsDics)):
		return - np.inf
	else:
		return float(fit_model(parsDics))


# In[7]:


pos = np.array([kappa_, r_s, q, 0.02, 88])
log_likelihood(pos)


# In[8]:


nwalkers = 15                                                   #Number of walkers
initial = np.random.uniform(low=[0., 0.1, 0.5, 0, 0], high=[2, 5, 1.0, 0.1, 179], size=[nwalkers, 5])
nwalkers, ndim = initial.shape                                      #Number of walkers/dimensions


# In[9]:


#Backup
filename = "Autolens_eNFW.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.20)]


# In[10]:


np.savetxt('Output_LogFile.txt', np.column_stack([0, 0, 0, 0]),
							fmt=b" %i \t %e \t %e \t %e", 
							header="Iteration \t Mean acceptance fraction \t Processing Time \t  Last 100 Mean Accp.")


# In[11]:

def run_emcee():
	with Pool(processes=5) as pool:

		#Print the number os cores/workers
		print("Workers nesse job:", pool._processes)
		

		#Initialize the new sampler
		sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, pool=pool,
										 backend=backend,)
		
		nsteps = 50000

		# We'll track how the average autocorrelation time estimate changes
		index = 0
		autocorr = np.empty(nsteps)
		# This will be useful to testing convergence
		old_tau = np.inf
		# This saves how many walkers have been accepted in the last 100 steps
		old_accp = np.zeros(nwalkers,)

		# Now we'll sample for up to max_n steps
		start = time.time()
		global_time = time.time()
		print("Start")
		for sample in sampler.sample(initial, iterations=nsteps, progress=True):
			# Only check convergence every 100 steps
			if sampler.iteration % 100:
				continue
			print("\n")
			print("##########################")

			#Compute how many walkes have been accepted during the last 100 steps

			new_accp = sampler.backend.accepted             #Total number of accepted
			old_accp = new_accp - old_accp                  #Number of accepted in the last 100 steps
			mean_accp_100 = np.mean(old_accp/float(100))    #Mean accp fraction of last 100 steps

			#Update a table output with acceptance
			table = np.loadtxt("Output_LogFile.txt")

			iteration = sampler.iteration
			accept = np.mean(sampler.acceptance_fraction)
			total_time = time.time() - global_time
			upt = np.column_stack([iteration, accept, total_time, mean_accp_100])

			np.savetxt('Output_LogFile.txt', np.vstack([table, upt]),
									fmt=b'	%i	 %e			 %e             %e', 
								header="Iteration	 Mean acceptance fraction	 Processing Time    Last 100 Mean Accp. Fraction")

			# Compute the autocorrelation time so far
			# Using tol=0 means that we'll always get an estimate even
			# if it isn't trustworthy
			tau = sampler.get_autocorr_time(tol=0)
			autocorr[index] = np.mean(tau)
			index += 1

			# Check convergence
			converged = np.all(tau * 100 < sampler.iteration)
			converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
			if converged:
				if 0.2 < accept < 0.35:
					break
			old_tau = tau



		end = time.time()
		print('\n')
		print("Final")
		multi_time = end - start
	return print("Multiprocessing took {0:.1f} seconds".format(multi_time))

if __name__ == '__main__':

    run_emcee()


# In[ ]:



