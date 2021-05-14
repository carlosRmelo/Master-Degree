#Control time packages
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"

import autolens as al
import autolens.plot as aplt
import numpy as np

from time import perf_counter as clock

from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

from multiprocessing import Pool

data_folder = "/home/carlos/Documents/GitHub/Master-Degree/Autolens tests/autolens_workspace/Test_5/Simulation_Data/"

#Reading MGE inputs
surf_lum, sigma_lum, qobs_lum = np.loadtxt("Input/JAM_Input.txt", unpack=True)        #MGE decomposition
surf_dm, sigma_dm , qobs_dm   = np.loadtxt("Input/eNFW.txt", unpack=True)             #DM component

#Only for lensing modelling 
z_l    = 0.299                                                         #Lens Redshift
z_s    = 4.100                                                         #Source Redshift 
D_l    = cosmo.angular_diameter_distance(z_l).value                    #Distance to lens [Mpc] 
mbh    = 1e9                                                           #mass of black hole [log10(M_sun)]
kappa_ = 0.075                                                         #kappa_s of DM profile
ml     = 7.00                                                          #mass to light ratio
r_s    = 11.5                                                          #scale radius [arcsec]
shear_comp = al.convert.shear_elliptical_comps_from(magnitude=0.02, phi=88) #external shear

#Autolens Data
imaging = al.Imaging.from_fits(
        image_path=f"{data_folder}/arcs_simulation.fits",
        noise_map_path=f"{data_folder}/noise_simulation.fits",
        psf_path=f"{data_folder}/psf_simulation.fits",
        pixel_scales=0.1,
    )

mask        = al.Mask.from_fits( file_path=f"{data_folder}/new_mask.fits", hdu=1, 
                                pixel_scales=imaging.pixel_scales)

masked_image = al.MaskedImaging(imaging=imaging, mask=mask, inversion_uses_border=True)   #Masked image
#aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

#--------------------------------------------------------------------------------------------------#
# PYAUTOLENS MODEL
#MGE mass profile
mass_profile = al.mp.MGE()    #Mass class

ell_comps    = al.convert.elliptical_comps_from(axis_ratio=qobs_dm[0], phi=0.0) #Elliptical components in Pyautolens units
eNFW         = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=kappa_, elliptical_comps=ell_comps ,scale_radius=r_s) #Analytical eNFW profile

#Components
#Do not include MGE DM component here
mass_profile.MGE_comps(z_l=z_l, z_s=z_s, 
                       surf_lum=surf_lum, sigma_lum=sigma_lum, qobs_lum=qobs_lum, ml=ml, mbh=mbh) 
mass_profile.Analytic_Model(eNFW)  #Include Analytical NFW

lens_galaxy = al.Galaxy(
    redshift=z_l,
    mass=mass_profile,
    shear=al.mp.ExternalShear(elliptical_comps=shear_comp)
)

source_galaxy = al.Galaxy(
    redshift=z_s,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.0),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(masked_imaging=masked_image, tracer=tracer)

#aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))
print("Log Likelihood with Regularization:", fit.log_likelihood_with_regularization)
print("Log Evidence:", fit.log_evidence)

boundary = {'ml': [0.5, 15], 'kappa_s': [0, 2], 'qDM': [0.1, 1], 'log_mbh':[7, 11],
                 'mag_shear': [0, 0.1], 'phi_shear': [0, 179], 'gamma': [0.80, 1.20]}


def prior_transform(theta):
    ml, kappa_s, qDM, log_mbh, mag_shear, phi_shear, gamma = theta
    parsDic = {"ml": ml, "kappa_s": kappa_s, "qDM": qDM,
                    "log_mbh":log_mbh, "mag_shear": mag_shear, "phi_shear": phi_shear, 
                    "gamma": gamma}
    for key in parsDic:
        parsDic[key] = boundary[key][0] + parsDic[key]*(boundary[key][1] - boundary[key][0])
        
    return np.array(list(parsDic.values()))

def log_likelihood(pars):
    quiet=True
    ml_model, kappa_s_model, qDM_model, log_mbh_model, mag_shear_model, phi_shear_model, gamma_model = pars
    
    ell_comps = al.convert.elliptical_comps_from(axis_ratio=qDM_model, phi=0.0) #Elliptical components in Pyautolens units
    eNFW = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=kappa_s_model,elliptical_comps=ell_comps, scale_radius=r_s) #Set the analytical model
    mass_profile.Analytic_Model(eNFW)        #Include analytical model
    mass_profile.MGE_Updt_parameters(ml=ml_model, mbh=10**log_mbh_model, gamma=gamma_model)
    
    #New lens model
    lens_galaxy = al.Galaxy(                                            
            redshift=mass_profile.z_l,
            mass=mass_profile,
            shear=al.mp.ExternalShear(elliptical_comps=shear_comp),
        )

    source_galaxy = al.Galaxy(
            redshift=mass_profile.z_s,
            pixelization=al.pix.Rectangular(shape=(40, 40)),
            regularization=al.reg.Constant(coefficient=1.0),
        )
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
    
    try:
        fit = al.FitImaging(masked_imaging=masked_image, tracer=tracer)

        log_likelihood = fit.log_likelihood_with_regularization

            
        if quiet is False:
            print("Lens Galaxy Configuration:")
            print("Log Likelihood_with_regularization:", log_likelihood)
            print("Log Normalization", fit.noise_normalization)
            print("Log Evidence:", fit.log_evidence)
            print("#------------------------------------#")
            print(lens_galaxy)
            print("\n")


            aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))
            aplt.Inversion.reconstruction(fit.inversion)              


        return log_likelihood
    except:
        print("An exception ocurres in Pyautolens_log_likelihood().")
        return -np.inf


# ml, kappa_s, qDM, log_mbh, mag_shear, phi_shear, gamma = pars
p0 = np.array([ml, kappa_, qobs_dm[0], np.log10(mbh), 0.02, 88., 1.0])
print("Best Input Log Likelihood", log_likelihood(p0))


from dynesty import NestedSampler

nlive = 100              # number of (initial) live points
ndim  = p0.size          # number of dimensions


# Now run with the static sampler
sampler = NestedSampler(log_likelihood, prior_transform, ndim, pool=Pool(),queue_size=6,
                        nlive=nlive)
                        
import pickle 
import shutil

from dynesty import plotting as dyplot
from dynesty import utils as dyfunc


labels = ["ml", "kappa_s", "qDM",
                    "log_mbh", "mag_shear", "phi_shear", 
                    "gamma"]

original = r"dynesty_lens.pickle"
beckup   = r"beckup/dynesty_lens_beckup.pickle"

def run_dynesty():
    start = time.time()
    for it, res in enumerate(sampler.sample()):
        if (it+1) % 50:
            continue
        
        print("Saving...")
        with open(f"dynesty_lens.pickle", "wb") as f:
                pickle.dump(sampler, f)
        print("File Save!")
        print(sampler.results.summary())
        print("Cumulative Time [s]:", (time.time() - start))
        print("#############################")
        print("\n")
        original = r"dynesty_lens.pickle"
        beckup   = r"beckup/dynesty_lens_beckup.pickle"
        
        beckup = shutil.copyfile(original, beckup)   
    return print("Final")  


if __name__ == '__main__':

    run_dynesty()        
