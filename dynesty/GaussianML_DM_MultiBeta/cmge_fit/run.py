#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mge1d_fit import mge1d_fit
from time import time

def ngauss(mge,x):
  y=x*0.
  for i in range(mge.shape[0]):
    y+=mge[i,0]/np.sqrt(2*np.pi)/mge[i,1]*np.exp(-0.5*(x/mge[i,1])**2)
  return y

def nfw(r,rho,rs,gamma):
  rst=rho*(r/rs)**gamma*(0.5+0.5*r/rs)**(-gamma-3)
  return rst

rho=0.01306
rs=10.3*1e3
gamma=-0.82541
tx=np.logspace(np.log10(0.5*1e3), np.log10(100*1e3), 200)
ty=nfw(tx,rho,rs,gamma)
terr=tx*0.+0.0001


start_time=time()
mge=mge1d_fit(tx,ty,terr,imax=15,rbound=[0.2*1e3,200*1e3],ngauss=8)
print ('time elapse: %.3f'%(time()-start_time))

yy=ngauss(mge,tx)

print (mge)
fig=plt.figure()
ax=fig.add_subplot(2,1,1)
ax.plot(np.log10(tx),np.log10(ty))
ax.plot(np.log10(tx),np.log10(yy),'r')
ax.set_xlabel('logR pc')
ax.set_ylabel('log rho')
ax1=fig.add_subplot(2,1,2)
ax1.plot(np.log10(tx),(yy/ty-1.)*100,'r')
ax1.set_xlabel('logR pc')
ax1.set_ylabel('error %')


plt.show()
#print 'time elapse: %.2f'%(time()-start_time)

