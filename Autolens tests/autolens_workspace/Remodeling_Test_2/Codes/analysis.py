import emcee
import numpy as np

read = emcee.backends.HDFBackend("simulation.h5")
labels = ["ML", "beta", "inclination", "log_mbh", "mag_shear", "phi_shear", "gamma"]

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



for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(labels[i]," {0:.2f}".format(mcmc[1]))

print("\n")

tau = read.get_autocorr_time()
