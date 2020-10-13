import emcee
import numpy as np
import time
from schwimmbad import MPIPool

np.random.seed(42)

# The definition of the log probability function
# We'll also use the "blobs" feature to track the "log prior" for each step
def log_prob(theta):
    log_prior = -0.5 * np.sum((theta - 1.0) ** 2 / 100.0)
    log_prob = -0.5 * np.sum(theta ** 2) + log_prior
    return log_prob, log_prior
print("Entrou")

# Run
with MPIPool() as pool:
    
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    # Initialize the walkers
    coords = np.random.randn(32, 5)
    nwalkers, ndim = coords.shape

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "tutorial.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
    max_n = 5000
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    start = time.time()
    print("Início")
    print("Workers nesse job:", pool.workers)
    sampler.run_mcmc(coords, max_n, progress=True)
    end = time.time()
    print('\n')
    print("Final")
    multi_time = end - start
    print("MPI took {0:.1f} seconds".format(multi_time))