from numba import jit
import numpy as np
from quadva import quadva
import astropy.units as u
import time

def test_x(tau1, 
            y, x, M, sigma, q):
    tau = np.reshape(tau1.copy(), (tau1.size,1))

   
    x_til = x/sigma
    y_til = y/sigma

   
    
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(x_til/(np.sqrt(1 - eta_sq*(tau)**2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    

    
    arr = aux*exp    
    return tau1*np.sum(arr, 1)

def test_y(tau1,
            y, x, M, sigma, q):
    tau = np.reshape(tau1.copy(), (tau1.size,1))
    
    x_til = x/sigma
    y_til = y/sigma
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(y_til/(np.power(1 - eta_sq*(tau)**2, 3/2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    
    
    arr = aux*exp
    
    return tau1*np.sum(arr, 1)

from numba import njit, double

test_x_nb = njit(double[:](double[:],double,double,double[:],double[:],double[:]))(test_x)
test_y_nb = njit(double[:](double[:],double,double,double[:],double[:],double[:]))(test_y)

M, sigma, q = np.loadtxt("Input.txt", unpack=True)
grid = np.loadtxt("grid.txt")
grid = (grid*u.arcsec).to(u.rad).value

result_x = np.zeros([len(grid), 3])              #Onde ficarão salvos os resultados da deflexão em x
result_y = np.zeros([len(grid), 3])              #Onde ficarão salvos os resultados da deflexão em y


start = time.time()
quadva(test_x_nb, [0., 1.], args=(grid[0][0], grid[0][1], M, sigma, q))   #Integral em x
print("alpha_x elapsed time:", time.time() - start)


start = time.time()
quadva(test_y_nb, [0., 1.], args=(grid[0][0], grid[0][1], M, sigma, q))   #Integral em x
print("alpha_y elapsed time:", time.time() - start)


start = time.time()
for i in range(len(grid)):                      #Começo do loop
    result_x[i] = quadva(test_x_nb, [0., 1.], args=(grid[i][0], grid[i][1], M, sigma, q))   #Integral em x
    result_y[i] = quadva(test_y_nb, [0., 1.], args=(grid[i][0], grid[i][1], M, sigma, q))   #Integral em y
print("All grid elapsed time:", time.time() - start)
