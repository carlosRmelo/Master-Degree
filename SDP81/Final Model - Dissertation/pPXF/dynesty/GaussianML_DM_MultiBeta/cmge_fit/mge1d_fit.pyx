cdef extern from 'MGE1D.h':
  struct multigaussexp1d:
    double *tot_counts
    double *sigma
    int ntotal

  multigaussexp1d mge_fit(double *x, double *y,\
        double *error, int num_data, int num_gauss,\
        int imax,double rbound_min, double rbound_max)

from libc.stdlib cimport malloc, free
import numpy as np
#cimport numpy as np

def mge1d_fit(x,y,err,ngauss=10,imax=5,rbound=None):
  if rbound is None:
    rbound=[min(x),max(x)]
  if len(x)!=len(y) or len(x)!=len(err):
    print 'x y err must have the same dimension'
    exit(1)
  cdef unsigned i,num_data=len(x),num_gauss=ngauss
  cdef multigaussexp1d mge
  cdef double *cx=<double *> malloc(num_data*sizeof(double))
  cdef double *cy=<double *> malloc(num_data*sizeof(double))
  cdef double *cerr=<double *> malloc(num_data*sizeof(double))
  if (cx==NULL or cy==NULL or cerr==NULL):
    print 'allocate array error'
    exit()
  #cdef np.ndarry MGE
  for i in range(num_data):
    cx[i]=x[i]
    cy[i]=y[i]
    cerr[i]=err[i]
  mge=mge_fit(cx,cy,cerr,int(num_data),int(num_gauss),int(imax),\
              float(rbound[0]),float(rbound[1]))
  MGE=np.zeros([ngauss,2])
  for i in range(num_gauss):
    MGE[i,0]=mge.tot_counts[i]
    MGE[i,1]=mge.sigma[i]
  good=MGE[:,0]!=0
  MGE=MGE[good,:]
  free(mge.tot_counts)
  free(mge.sigma)
  free(cx)
  free(cy)
  return MGE

