import emcee
import numpy as np

read = emcee.backends.HDFBackend("Simulation4.h5")
pars_input = {'ml': 7.0, 'beta': -0.15, 'qinc': 0.38837304829549607, 'log_mbh': 9.0, 
          'kappa_s': 0.075, 'qDM': 0.813731905546916, 'mag_shear': 0.02, 'phi_shear': 88.0, 'gamma': 1.0}
labels = list(pars_input.keys())


tau = read.get_autocorr_time(tol=0)
log = read.get_log_prob()
print("Total walkers above 0:", (log[-1] > 0).sum() )

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



for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(labels[i]," {0:.2f}".format(mcmc[1]))

print("\n")

tau = read.get_autocorr_time(tol=0)
print(tau, read.iteration/50)
print("accepted", read.accepted)
print("Mean accepted", np.mean(read.accepted/read.iteration))
