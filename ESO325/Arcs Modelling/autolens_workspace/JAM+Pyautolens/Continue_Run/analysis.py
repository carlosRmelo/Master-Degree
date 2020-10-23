import emcee
import numpy as np

read = emcee.backends.HDFBackend("save.h5")
labels = ["ml1-2", "ml3", "ml4","ml5","ml6","ml7","b1", "b2", "b3", "b4","b5","b6","b7",
              "Inc", "qDM", "Log rho_s", "Log mbh", "Mag Shear", "Phi Shear", "gamma"]


tau = read.get_autocorr_time(tol=0)
log = read.get_log_prob()

print("\n")
print("\n")
print("Autocorr:")
print(tau)
print("All Log_prob:")
print(log)
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
last = read.get_last_sample()
