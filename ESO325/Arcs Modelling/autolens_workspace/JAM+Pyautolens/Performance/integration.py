

from jampy.quadva import quadva
import astropy.units as u
import numpy as np
import time




# In[46]:


def alphax(tau1, 
            y, x, M, sigma, q):
    
    
    tau = tau1[:, None]
   
    x_til = x/sigma
    y_til = y/sigma

   
    
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(x_til/(np.sqrt(1 - eta_sq*(tau)**2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    

    
    arr = aux*exp    
    
    return tau1*arr.sum(1)



def alphay(tau1,
            y, x, M, sigma, q):
    

    tau = tau1[:, None]    
    x_til = x/sigma
    y_til = y/sigma
    eta = np.sqrt(1 - q**2)
    eta_sq = 1 - q**2
    
    aux = (M/sigma)*(y_til/(np.power(1 - eta_sq*(tau)**2, 3/2)))
    exp_arg = (tau**2/2)*(x_til**2 + y_til**2/(1-eta_sq*(tau)**2))
    exp = np.exp(-exp_arg)
    
    
    arr = aux*exp
    
    return tau1*arr.sum(1)


# In[47]:


def MGE_Grid_parameters(grid, quiet=True):
    y0 = np.array([0])
    x0 = np.array([0])
    initial = np.array([y0, x0, M, sigma, q])
    Grid_parameters = np.array([y0, x0, M, sigma, q])

        #Agora realizamos um loop para criar todas as posições:
    for i in range(len(grid)-1): #-1 pois já começamos com uma posição (initial)
        Grid_parameters = np.vstack([Grid_parameters, initial])

        #Agora atualizamos as posições (y,x) e convertemos suas unidades para rad
    Grid_parameters[:, 0] = (grid[:, 0]*u.arcsec).to(u.rad).value
    Grid_parameters[:, 1] = (grid[:, 1]*u.arcsec).to(u.rad).value


        #Class parameter
    Grid_parameters
        
    if quiet is False:
        print("Pyautolens MGE Class successfully initialized!!")
        

    return Grid_parameters


# In[48]:


M, sigma, q = np.loadtxt("Input.txt", unpack=True)


# In[51]:


grid = np.loadtxt("grid.txt")
Grid_parameter = MGE_Grid_parameters(grid)


# In[52]:


result_x = np.zeros([len(grid), 3])              #Onde ficarão salvos os resultados da deflexão em x
result_y = np.zeros([len(grid), 3])              #Onde ficarão salvos os resultados da deflexão em y

         
start = time.time()
for i in range(500):                      #Começo do loop
    result_x[i] = quadva(alphax, [0., 1.], args=(Grid_parameter[i]),epsrel=1e-10)   #Integral em x
    result_y[i] = quadva(alphay, [0., 1.], args=(Grid_parameter[i]),epsrel=1e-10)   #Integral em y
print(time.time()-start)


# In[ ]:




