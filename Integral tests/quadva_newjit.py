import ctypes
import timeit

from tqdm import tqdm
import numpy as np
from scipy import integrate, LowLevelCallable
import numba as nb
from numba import types

import astropy.units as u
import time
from jampy.quadva import quadva


def alpha_x(tau1, y, x, array1, array2, array3):
    
    tau = np.reshape(tau1.copy(), (tau1.size,1))
    x_til   = x/array2
    y_til   = y/array2
    eta     = np.sqrt(1.0 - array3**2)
    eta_sq  = 1 - array3**2
    
    aux     = (array1/array2) * (x_til/(np.sqrt(1 - eta_sq*(tau)**2)))
    exp_arg = (tau**2/2) * (x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp     = np.exp(-exp_arg)
    
    arr = aux*exp    
    
    return tau1 * np.sum(arr, 1)

def alpha_y(tau1,
            y, x, array1, array2, array3):
    

    tau = np.reshape(tau1.copy(), (tau1.size,1))
    x_til = x/array2
    y_til = y/array2
    eta = np.sqrt(1 - array3**2)
    eta_sq = 1 - array3**2
    
    aux = (array1/array2)*(y_til/(np.power(1 - eta_sq*(tau)**2, 3/2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    
    arr = aux*exp    
    
    return tau1 * np.sum(arr, 1)


##################################################
# JIT INTEGRAND
alpha_x_nb = nb.njit(alpha_x)
alpha_y_nb = nb.njit(alpha_y)




def do_integrate_alpha_x(ybin, xbin, array1, array2, array3, lolim=0, hilim=1):
    return quadva(alpha_x_nb, [lolim, hilim], args=(ybin, xbin, array1, array2, array3))

def do_integrate_alpha_y(ybin, xbin, array1, array2, array3, lolim=0, hilim=1):
    return quadva(alpha_y_nb, [lolim, hilim], args=(ybin, xbin, array1, array2, array3))



M, sigma, q = np.loadtxt("Input.txt", unpack=True, dtype=np.float64)
grid = np.loadtxt("grid.txt")
grid = (grid*u.arcsec).to(u.rad).value



start = time.time()
do_integrate_alpha_x(grid[0][0], grid[0][1], M, sigma, q)
print("alpha_x elapsed time:", time.time() - start)

start = time.time()
do_integrate_alpha_y(grid[0][0], grid[0][1], M, sigma, q)
print("alpha_y elapsed time:", time.time() - start)


result_x = np.zeros([len(grid), 3])              #Onde ficarão salvos os resultados da deflexão em x
result_y = np.zeros([len(grid), 3])              #Onde ficarão salvos os resultados da deflexão em y



start = time.time()
for i in range(len(grid)):                      #Começo do loop
    result_x[i] = do_integrate_alpha_x(grid[i][0], grid[i][1], M, sigma, q)    #Integral em x
    result_y[i] = do_integrate_alpha_y(grid[i][0], grid[i][1], M, sigma, q)    #Integral em y
print("All grid elapsed time:", time.time() - start)