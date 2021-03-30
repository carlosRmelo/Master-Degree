import emcee
import numpy as np

boundary = {'inc': [50, 90], 'beta': [-5, 5], 'ml': [0.5, 15],  
                 'ml0': [0.5, 15], 'delta': [0.1, 2], 'lower': [0, 1],
                 'log_rho_s': [6, 12], 'qDM': [0.2, 1], 'log_mbh':[7, 11],
                 'mag_shear': [0, 0.1], 'phi_shear': [0, 179], 'gamma': [0.8, 1.2]}

def linear_tranform(pars):
    for keys in pars:
        OldRange = 1 - 0    #All transformations tested returns values between [0,1]
        
        if OldRange == 0:
            pars[keys] = boundary[keys][0]
        else:
            NewRange   = boundary[keys][1] - boundary[keys][0]
            pars[keys] = (((pars[keys] - 0) * NewRange) / OldRange) + boundary[keys][0]
    
    return pars

def absolut(x):
    return (1 + x / (1 + abs(x))) * 0.5


read = emcee.backends.HDFBackend("simulation.h5")
labels = ["ML", "beta", "inclination", "log_mbh", "log_rho_s", "qDM", "mag_shear", "phi_shear", "gamma"]


tau = read.get_autocorr_time(tol=0)
log = read.get_log_prob()

print("\n")
print("\n")
print("Autocorr:")
print(tau)
print("Last Log_prob. Position", log.shape[0])
print(log[ log.shape[0]-1 ])
print("Higher Log_prob:", np.max(log))
print("\n")
print("\n")

flat_samples = read.get_chain(flat=True)
chains, ndim = flat_samples.shape



mcmc = np.zeros(ndim)
for i in range(ndim):
    mcmc[i] = np.percentile(flat_samples[:, i], [16, 50, 84])[1]

result = absolut(mcmc)

parsDic = {'ml': result[0], 'beta': result[1], 'inc': result[2],
             'log_mbh': result[3], 'log_rho_s':result[4], 'qDM':result[5], 'mag_shear':result[6], 'phi_shear': result[7], 'gamma': result[8]}

parsDic = linear_tranform(parsDic)
print(parsDic)

print("\n")

tau = read.get_autocorr_time()
