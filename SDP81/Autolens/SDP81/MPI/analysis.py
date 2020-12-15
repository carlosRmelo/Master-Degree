import emcee
import numpy as np

read = emcee.backends.HDFBackend("SDP81_Combined.h5")
labels = ["Ml0", "Delta", "Lower","b1", "b2", "b3", "b4","b5","b6","b7",
		 "b8","Inc", "Log mbh", "qDM", "Logrho_s", "Mag_shear", "Phi_Shear", "Gamma"]


tau = read.get_autocorr_time(tol=0)
log = read.get_log_prob()

print("\n")
print("\n")
print("Autocorr:")
print(tau)
print("Last Log_prob:")
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
