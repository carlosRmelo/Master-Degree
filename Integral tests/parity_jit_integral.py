from numba import jit
import numpy as np
from quadva import quadva
import astropy.units as u
import time

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def alpha_x(tau1, 
            y, x, M, sigma, q):

    tau = tau1.reshape(tau1.size, 1)
   
    x_til = x/sigma
    y_til = y/sigma

   
    
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(x_til/(np.sqrt(1 - eta_sq*(tau)**2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    

    
    arr = aux*exp
    return tau1*np.sum(arr, 1)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def alpha_y(tau1,
            y, x, M, sigma, q):
    

    tau = tau1.reshape(tau1.size, 1)
    x_til = x/sigma
    y_til = y/sigma
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(y_til/(np.power(1 - eta_sq*(tau)**2, 3/2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    
    
    arr = aux*exp
    
    return tau1*np.sum(arr, 1)

M, sigma, q = np.loadtxt("Input.txt", unpack=True)
grid = np.loadtxt("Grid_sdp.txt")
grid = (grid*u.arcsec).to(u.rad).value

grid_result = np.zeros_like(grid)                                      #Onde salvamos os resultados

start = time.time()
quadva(alpha_x, [0., 1.], args=(grid[0][0], grid[0][1], M, sigma, q))   #Integral em x
print("alpha_x elapsed time:", time.time() - start)


start = time.time()
quadva(alpha_y, [0., 1.], args=(grid[0][0], grid[0][1], M, sigma, q))   #Integral em x
print("alpha_y elapsed time:", time.time() - start)


start = time.time()
#Calculo em y

for i in range(len(grid)):
    if grid_result[i, 0] == 0:
        result_y = quadva(alpha_y, [0., 1.], args=(grid[i][0], grid[i][1], M, sigma, q))  #Integral in y
        grid_result[i, 0] = result_y[0]
        
        index = np.where( (grid[:,0] == grid[i][0]) & (grid[:,1] == -grid[i][1])  ) #Fix y and change x
        grid_result[index, 0] = result_y[0]
        
        index = np.where( (grid[:,0] == -grid[i][0]) & (grid[:,1] == grid[i][1])  ) #Fix x and change y
        grid_result[index, 0] = -result_y[0]
        
        index = np.where( (grid[:,0] == -grid[i][0]) & (grid[:,1] == -grid[i][1])  ) #Change both
        grid_result[index, 0] = -result_y[0]
    
    
#Calculo em x

for i in range(len(grid)):
    if grid_result[i, 1] == 0:
        result_x = quadva(alpha_x, [0., 1.], args=(grid[i][0], grid[i][1], M, sigma, q))  #Integral in x
        grid_result[i, 1] = result_x[0]
        
        index = np.where( (grid[:,0] == -grid[i][0]) & (grid[:,1] == grid[i][1])  ) #Fix x and change y
        grid_result[index, 1] = result_x[0]
        
        index = np.where( (grid[:,0] == grid[i][0]) & (grid[:,1] == -grid[i][1])  ) #Fix y and change x
        grid_result[index, 1] = -result_x[0]
        
        index = np.where( (grid[:,0] == -grid[i][0]) & (grid[:,1] == -grid[i][1]) ) #Change both 
        grid_result[index, 1] = -result_x[0]

print("All grid elapsed time:", time.time() - start)

np.savetxt("parity_result.txt", grid_result, header="alpha_y \t\t alpha_x", fmt="%.10e \t %.10e")