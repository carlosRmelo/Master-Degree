
import sys
import time
import emcee
import numpy as np
from schwimmbad import MPIPool
import matplotlib.pyplot as plt
from time import perf_counter as clock

np.random.seed(42)

# The definition of the log probability function
# We'll also use the "blobs" feature to track the "log prior" for each step
def log_prob(theta):
    log_prior = -0.5 * np.sum((theta - 1.0) ** 2 / 100.0)
    log_prob = -0.5 * np.sum(theta ** 2) + log_prior
    return log_prob, log_prior


np.savetxt('Output_LogFile.txt', np.column_stack([0, 0, 0]),
                            fmt=b'	%i	 %e			 %e	 ', 
                            header="Iteration	 Mean acceptance fraction	 Processing Time")

np.savetxt("LogFile_LastFit.txt", np.column_stack([0, 0, 0, 0, 0, 0]),
                            fmt=b'%e	 %e	 %e	 %e	 %e	 %e	', 
                            header="Iteration	 theta1	 theta2	 theta3	 theta4	 theta5	")


with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Initialize the walkers
    coords = np.random.randn(32, 5)
    nwalkers, ndim = coords.shape

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "MPI_SAVE_and_OUTPUT.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend)
    
    max_n = 100000
    
    global_time = clock()
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(coords, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1



        #Update a table output with acceptance
        table = np.loadtxt("Output_LogFile.txt")


        iteration = sampler.iteration
        accept = np.mean(sampler.acceptance_fraction)
        total_time = clock() - global_time
        upt = np.column_stack([iteration, accept, total_time])

        np.savetxt('Output_LogFile.txt', np.vstack([table, upt]),
                                fmt=b'	%i	 %e			 %e	 ', 
                                header="Iteration	 Mean acceptance fraction	 Processing Time")

        #Update table output with last best fit
        last_fit_table = np.loadtxt("LogFile_LastFit.txt")
        flat_samples = sampler.get_chain()
        values = []
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            values.append(mcmc[1])

        values = np.array(values)
        upt = np.append(iteration, values)

        np.savetxt("LogFile_LastFit.txt", np.vstack([last_fit_table, upt]),
                                fmt=b'%e	 %e	 %e	 %e	 %e	 %e	', 
                                header="Iteration	 theta1	 theta2	 theta3	 theta4	 theta5	") 

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
