import sys
import time
import emcee
import numpy as np
from schwimmbad import MPIPool

def log_prob(theta):
    t = time.time() + np.random.uniform(0.005, 0.008)
    while True:
        if time.time() >= t:
            break
    return -0.5*np.sum(theta**2)
print("Entrou")



with MPIPool() as pool:
    
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    print("Início")
    print("Workers nesse job:", pool.workers)
    np.random.seed(42)
    initial = np.random.randn(32, 5)
    nwalkers, ndim = initial.shape
    nsteps = 1000

    filename = "mpi.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    start = time.time()
    sampler.run_mcmc(initial, nsteps,progress=True)
    end = time.time()
    print('\n')
    print("Final")
    multi_time = end - start
    print("MPI took {0:.1f} seconds".format(multi_time))