import ctypes
import timeit

from tqdm import tqdm
import numpy as np
from scipy import integrate, LowLevelCallable
import numba as nb
from numba import types

import astropy.units as u
import time

def alpha_x(tau,
             y, x, array1, array2, array3):
    x_til   = x/array2
    y_til   = y/array2
    eta     = np.sqrt(1.0 - array3**2)
    eta_sq  = 1 - array3**2
    
    aux     = (array1/array2) * (x_til/(np.sqrt(1 - eta_sq*(tau)**2)))
    exp_arg = (tau**2/2) * (x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp     = np.exp(-exp_arg)
    
    return tau * np.sum(aux*exp)

def alpha_y(tau,
            y, x, array1, array2, array3):
    

   
    x_til = x/array2
    y_til = y/array2
    eta = np.sqrt(1 - array3**2)
    eta_sq = 1 - array3**2
    
    aux = (array1/array2)*(y_til/(np.power(1 - eta_sq*(tau)**2, 3/2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    
    
    return tau * np.sum(aux*exp)




##################################################
# LOWLEV CALLABLE

def create_jit_integrand_function_quad(integrand_function, args_dtype):
    jitted_function = nb.njit(integrand_function)

    @nb.cfunc(types.float64(types.float64,types.CPointer(args_dtype)))
    def wrapped(tau,user_data_p):
        #Array of structs
        user_data = nb.carray(user_data_p, 1)

        #Extract the data
        y=user_data[0].ybin
        x=user_data[0].xbin
        array1=user_data[0].M
        array2=user_data[0].sigma
        array3=user_data[0].q

        return jitted_function(tau, y, x, array1, array2, array3)
    return wrapped

def do_integrate_w_arrays_lowlev_quad(func,args,lolim=0, hilim=1):
    integrand_func=LowLevelCallable(func.ctypes,user_data=args.ctypes.data_as(ctypes.c_void_p))
    return integrate.quad(integrand_func, lolim, hilim)


def process_lowlev_callable_quad(lowcall_fun, ybin, xbin):
    return do_integrate_w_arrays_lowlev_quad(lowcall_fun, np.array((ybin, xbin, M, sigma, q), dtype=args_dtype), lolim=0, hilim=1)



M, sigma, q = np.loadtxt("Input.txt", unpack=True, dtype=np.float64)
grid = np.loadtxt("grid.txt")
grid = (grid*u.arcsec).to(u.rad).value




args_dtype = types.Record.make_c_struct([
            ('ybin', types.float64),
            ('xbin', types.float64),
            ('M', types.NestedArray(dtype=types.float64, shape=M.shape)),
            ('sigma', types.NestedArray(dtype=types.float64, shape=sigma.shape)),
            ('q', types.NestedArray(dtype=types.float64, shape=q.shape)),])




lowcall_alpha_y = create_jit_integrand_function_quad(alpha_y, args_dtype)
lowcall_alpha_x = create_jit_integrand_function_quad(alpha_x, args_dtype)

start = time.time()
process_lowlev_callable_quad(lowcall_alpha_x, grid[0][0], grid[0][1])
print("alpha_x elapsed time:", time.time() - start)

start = time.time()
process_lowlev_callable_quad(lowcall_alpha_y, grid[0][0], grid[0][1])
print("alpha_y elapsed time:", time.time() - start)



result_x = np.zeros([len(grid), 2])              #Onde ficarão salvos os resultados da deflexão em x
result_y = np.zeros([len(grid), 2])              #Onde ficarão salvos os resultados da deflexão em y



start = time.time()
for i in range(len(grid)):                      #Começo do loop
    result_x[i] = process_lowlev_callable_quad(lowcall_alpha_x, grid[i][0], grid[i][1])#Integral em x
    result_y[i] = process_lowlev_callable_quad(lowcall_alpha_y, grid[i][0], grid[i][1])#Integral em y
print("All grid elapsed time:", time.time() - start)
