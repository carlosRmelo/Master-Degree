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

from multiprocessing import Pool, cpu_count

data_folder = "/home/carlos/Documents/GitHub/Master-Degree/SDP81/Autolens/ALMA/Data"#Reading MGE inputs
surf_lum, sigma_lum, qobs_lum = np.loadtxt("Input/JAM_Input.txt", unpack=True)        #MGE decomposition

#Only for lensing modelling 
z_l    = 0.299                                                         #Lens Redshift
z_s    = 3.042                                                         #Source Redshift 
D_l    = cosmo.angular_diameter_distance(z_l).value                    #Distance to lens [Mpc] 
mbh    = 1e9                                                           #mass of black hole [log10(M_sun)]
kappa_ = 0.075                                                         #kappa_s of DM profile
r_s    = 11.5
ml     = 7.00                                                          #mass to light ratio
phi_shear = 88                                                         #Inclination of external shear [deg]
mag_shear = 0.02                                                       #magnitude of shear
shear_comp = al.convert.shear_elliptical_comps_from(magnitude=mag_shear, phi=phi_shear) #external shear

#Autolens Data
imaging = al.Imaging.from_fits(
        image_path=f"{data_folder}/Alma_with_lens_center.fits",
        noise_map_path=f"{data_folder}/rms_noise_map.fits",
        psf_path=f"{data_folder}/Alma_psf_rot.fits",
        pixel_scales=0.01,
        image_hdu=1, noise_map_hdu=1, psf_hdu=1,
    )

mask        = al.Mask.from_fits( file_path=f"{data_folder}/mask2.fits", 
                                pixel_scales=imaging.pixel_scales)

masked_image = al.MaskedImaging(imaging=imaging, mask=mask, inversion_uses_border=True)   #Masked image
#aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
# PYAUTOLENS MODEL
#MGE mass profile

#Initializing
mass_profile = al.mp.MGE()
ell_comps = al.convert.elliptical_comps_from(axis_ratio=0.85, phi=0.0) #Elliptical components in Pyautolens units
eNFW      = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=kappa_, elliptical_comps=ell_comps, scale_radius=r_s) #pseudo elliptical NFW


#Components

mass_profile.MGE_comps(z_l=z_l, z_s=z_s, 
                       surf_lum=surf_lum, sigma_lum=sigma_lum, qobs_lum=qobs_lum, ml=ml,
                       mbh=mbh)
mass_profile.Analytic_Model(analytic_profile=eNFW)



#Lens galaxy
lens_galaxy = al.Galaxy(
    redshift=z_l,
    mass=mass_profile,
    shear=al.mp.ExternalShear(elliptical_comps=shear_comp)
)

source_galaxy = al.Galaxy(
    redshift=z_s,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.50),
)

print("Starting functions...")
start = time.time()
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fit = al.FitImaging(masked_imaging=masked_image, tracer=tracer)

#aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True,critical_curves=False,caustics=False))
print("Log Likelihood with Regularization:", fit.log_likelihood_with_regularization)
print("Log Evidence:", fit.log_evidence)
print("Log Likelihood :", fit.log_likelihood)
print("Elapsed Time [s]:", (time.time() - start))


boundary = {'ml': [0.5, 15], 'kappa_s': [0, 2], 'r_s': [5, 30], 'qDM': [0.1, 1], 'log_mbh':[7, 11],
                 'mag_shear': [0, 0.1], 'phi_shear': [0, 179], 'gamma': [0.95, 1.05]}


def prior_transform(theta):
    ml, kappa_s, qDM, r_s, log_mbh, mag_shear, phi_shear, gamma = theta
    parsDic = {"ml": ml, "kappa_s": kappa_s, "r_s": r_s, "qDM": qDM,
                    "log_mbh":log_mbh, "mag_shear": mag_shear, "phi_shear": phi_shear, 
                    "gamma": gamma}
    for key in parsDic:
        parsDic[key] = boundary[key][0] + parsDic[key]*(boundary[key][1] - boundary[key][0])
        
    return np.array(list(parsDic.values()))

def log_likelihood(pars):
    quiet=True
    ml_model, kappa_s_model, r_s_model, qDM_model, log_mbh_model, mag_shear_model, phi_shear_model, gamma_model = pars
    
    ell_comps = al.convert.elliptical_comps_from(axis_ratio=qDM_model, phi=0.0) #Elliptical components in Pyautolens units
    eNFW = al.mp.dark_mass_profiles.EllipticalNFW(kappa_s=kappa_s_model,elliptical_comps=ell_comps, scale_radius=r_s_model) #Set the analytical model
    mass_profile.Analytic_Model(eNFW)        #Include analytical model
    mass_profile.MGE_Updt_parameters(ml=ml_model, mbh=10**log_mbh_model, gamma=gamma_model)
    shear_comp_model = al.convert.shear_elliptical_comps_from(magnitude=mag_shear_model, phi=phi_shear_model)
    #New lens model
    lens_galaxy = al.Galaxy(                                            
            redshift=mass_profile.z_l,
            mass=mass_profile,
            shear=al.mp.ExternalShear(elliptical_comps=shear_comp_model),
        )

    source_galaxy = al.Galaxy(
            redshift=mass_profile.z_s,
            pixelization=al.pix.Rectangular(shape=(40, 40)),
            regularization=al.reg.Constant(coefficient=1.5),
        )
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
    
    try:
        fit = al.FitImaging(masked_imaging=masked_image, tracer=tracer)

        log_evidence = fit.log_evidence

            
        if quiet is False:
            print("Lens Galaxy Configuration:")
            print("Log Likelihood_with_regularization:", fit.log_likelihood)
            print("Log Normalization", fit.log_likelihood)
            print("Log Evidence:", fit.log_evidence)
            print("#------------------------------------#")
            print(lens_galaxy)
            print("\n")


            aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))
            aplt.Inversion.reconstruction(fit.inversion)              


        return log_evidence
    except:
        print("An exception ocurres in Pyautolens_log_likelihood().")
        return -np.inf



# ml, kappa_s, r_s qDM, log_mbh, mag_shear, phi_shear, gamma = pars
print("Testing Likelihood call...")
start = time.time()
p0 = np.array([ml, kappa_, r_s, 0.85, np.log10(mbh), 0.02, 88., 1.0])
value = log_likelihood(p0)
print("Likelihood Call Valeu:", value)
print("Likelihood Call Time:", (time.time() - start))

from dynesty import NestedSampler
import _pickle  as pickle
import shutil




labels = ["ml", "kappa_s", "r_s", "qDM",
                    "log_mbh", "mag_shear", "phi_shear", 
                    "gamma"]

original = r"dynesty_lens.pickle"
beckup   = r"beckup/dynesty_lens_beckup.pickle"
def run_dynesty():
    niter=1
    ncall=0
    with Pool(cpu_count()-1) as executor:
        print("Number of CPUS:", cpu_count())
        nlive = 40             # number of (initial) live points
        ndim  = p0.size         # number of dimensions


        # Now run with the static sampler
        sampler = NestedSampler(log_likelihood, prior_transform, ndim, 
                                    pool=executor, queue_size=cpu_count(),
                                    nlive=nlive, sample="rwalk",
                                )
        print_func      = None
        print_progress  = True
        pbar,print_func = sampler._get_print_func(print_func,print_progress)                        
        #fit_fun         = sampler.loglikelihood
        
        start = time.time()
        print("\n")
        print("Job started!")
        for it, res in enumerate(sampler.sample(dlogz=0.01)):
            
            (worst, ustar, vstar, loglstar, logvol,
                    logwt, logz, logzvar, h, nc, worst_it,
                    boundidx, bounditer, eff, delta_logz) = res

            niter+=1
            ncall+=nc

            print_func(res,niter,ncall,nbatch=0,dlogz=0.01,logl_max=np.inf)

            
            if it%50:
                continue
            print("\n")
            print("Saving...")
            with open(f"dynesty_lens.pickle", "wb") as f:
                pickle.dump(sampler, f, -1)
                f.close()
            print("File Save!\n")
            print("Cumulative Time [s]:", (time.time() - start))
            print("\n ########################################## \n")

            original = r"dynesty_lens.pickle"
            beckup   = r"beckup/dynesty_lens_beckup.pickle"
            
            beckup = shutil.copyfile(original, beckup)  
        
    print("Elapse Time:", (time.time() - start))  
    print("Saving Final Sample:")
    # Adding the final set of live points.
    for it_final, res in enumerate(sampler.add_live_points()):
        pass

    with open(f"final_dynesty_lens.pickle", "wb") as f:
        pickle.dump(sampler, f, -1) 
    print("Saved!!")
    return print("Final")  


if __name__ == '__main__':

    run_dynesty()        
