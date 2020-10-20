import emcee
import numpy as np

read = emcee.backends.HDFBackend("save.h5")
labels = ["ml1-2", "ml3", "ml4","ml5","ml6","ml7","b1", "b2", "b3", "b4","b5","b6","b7",
              "Inc", "qDM", "Log rho_s", "Log mbh", "Mag Shear", "Phi Shear", "gamma"]


print("Parameters are:", labels)

tau = read.get_autocorr_time(tol=0)
print("Autocorr:", tau)

flat_samples = read.get_chain(flat=True)
chains, ndim = flat_samples.shape

from IPython.display import display, Math

values = []
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    values.append(mcmc[1])
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
read.get_autocorr_time()