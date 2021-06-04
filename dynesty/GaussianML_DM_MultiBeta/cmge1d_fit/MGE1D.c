#include<stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nnls.c"
#include "mpfit.h"

#define SQRT_TOW_PI  2.5066282746310002

struct multigaussexp1d {
    double *tot_counts;
    double *sigma;
    int ntotal;
};

struct data {
  double *x;
  double *y;
  double *error;
  double *L;
};

double* range( double lo, double hi, int n, int open ) {
    
    int i;
    double *returnval;
    
    returnval = (double *) malloc( n * sizeof( double ) );
    
    if ( open ) {
        
        // create n bins between lo and hi, return the centre of the bins
        for ( i = 0; i < n; i++ ) {
            returnval[i] = lo + ( hi - lo ) * ( 0.5 + i ) / n;
        }
    
    } else {
        
        // for n < 0, create an integer array
        if ( n < 0 ) n = (int) ( hi - lo );
        
        // return n element array with limits included
        for ( i = 0; i < n; i++ ) {
            returnval[i] = lo + ( hi - lo ) * i / ( n - 1 );
        }
        
    }
    
    return (double *) returnval;    
}

double *Ngauss(double *x, int num, struct multigaussexp1d *mge){
  int i,j;
  double *y=(double *) malloc(num*sizeof(double));
  for(i=0; i < num; i++){
    y[i]=0.0;
    for(j=0; j < mge->ntotal; j++ ){
      y[i]+=mge->tot_counts[j]/SQRT_TOW_PI/mge->sigma[j]* \
            exp(-0.5/(mge->sigma[j]*mge->sigma[j])*x[i]*x[i] );
    }
  }
  return y;
}

int myfunc(int m, int n, double *p, double *dy, double **dvec, void *data){
  double *xx=((struct data *)data)->x;
  double *yy=((struct data *)data)->y;
  double *error=((struct data *)data)->error;
  double *b=(double *) malloc(m*sizeof(double));
  double *L=((struct data *)data)->L;
  double *Y;
  double *X=(double *) malloc(n*sizeof(double));
  void *rnorm=NULL,*wp=NULL,*zzp=NULL,*indexp=NULL; 
  struct multigaussexp1d mge;
  int i,j,status; 
  mge.ntotal=n;
  mge.sigma=(double *) malloc(n*sizeof(double));
  mge.tot_counts=(double *) malloc(n*sizeof(double));

  // set mge parameters  
  for(i=0;i<n;i++){
    mge.sigma[i]=pow(10.0,p[i]);
    //mge->tot_counts[i]=1.
  }
  //inatiallize A b(i.e. yy) for NNLS
  double **A=(double **) malloc(n*sizeof(double*));
  for(i=0;i<n;i++){
    A[i]=(double *) malloc(m*sizeof(double));
  }
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[j]=1.;
      A[i][j]=exp(-0.5/(mge.sigma[i]*mge.sigma[i])*xx[j]*xx[j])\
        /SQRT_TOW_PI/mge.sigma[i]/yy[j];
    }
  }
  //for(i=0;i<m;i++)
  //  printf("%.2f %.4e %.2f\n",xx[i],yy[i],error[i]);
  status=nnls(A,m,n,b,X,rnorm,wp,zzp,indexp);
  if (status){
    printf("NNLS error");
    exit(1);
  }
  for(i=0;i<n;i++){
    mge.tot_counts[i]=X[i];
    L[i]=X[i];
    //printf("total counts %d: %.2f\n",i,X[i]);
    //printf("MGE %d: %.3e %.3e\n",i,X[i],mge.sigma[i]);
  }
  // calculate residual 
  Y=Ngauss(xx,m,&mge);
  for(i=0;i<m;i++){
    dy[i]=(log10(Y[i])-log10(yy[i]))/error[i];
  }
  free(Y);
  free(b);
  free(X);
  free(mge.sigma);
  free(mge.tot_counts);
  for(i=0;i<n;i++) free(A[i]);
  free(A);
  return 0;
}

/* The function fitting a profile y=f(x) using multi-Gaussians
   Input value:
     x: float pointer to a n element array which contain x values 
     y: float pointer to a n element array which contain y values
     error: float pointer to a n element array which contain error of y 
     num_data: int number n, the lenght of the input arrays 
     num_gauss: int N, the number of Gaussians used in the fitting 
     imax:  int number m, maximum number for the iteration, usually 10-20 is enough
     rbound_min,rbound_max: the min and max value of the possible sigma of the
                            Gaussians in the fitting 
   Output value:
     mge: structure which contain the fitting results (coefficients of the Gaussians)
*/
struct multigaussexp1d mge_fit(double *x, double *y,\
        double *error, int num_data, int num_gauss,\
        int imax,double rbound_min, double rbound_max){

  int i,status,j;
  //the final returned mge structure
  struct multigaussexp1d mge;
  mge.tot_counts=(double *) malloc(num_gauss*sizeof(double));
  mge.sigma=(double *) malloc(num_gauss*sizeof(double));
  //solver type, solver and fitted funtion
  
  double *p0;
  double *L=(double *) malloc(num_gauss*sizeof(double));
  p0=range(log10(rbound_min)+0.1,log10(rbound_max)-0.1,num_gauss,0);
  //for(i=0;i<num_gauss;i++)
  //  printf("%f\n",p0[i]);
  //constrains on the paramters
  mp_par pars[num_gauss];
  memset(&pars[0],0,sizeof(pars));
  for(i=0;i<num_gauss;i++){
    pars[i].limited[0]=1;
    pars[i].limited[1]=1;
    pars[i].limits[0]=log10(rbound_min);
    pars[i].limits[1]=log10(rbound_max);
  }
  //structure contain result
  mp_result result;
  memset(&result,0,sizeof(result));
  struct data d={x,y,error,L};
  //for(i=0;i<num_data;i++)
  //  printf("%.2f %.2f %.2f\n",x[i],y[i],error[i]);
  
  mp_config config;
  memset(&config, 0, sizeof(config));
  config.maxiter = imax;
  //call mpifit
  status = mpfit(myfunc, num_data , num_gauss, p0, pars,\
           &config, (void *) &d, &result);
  mge.ntotal=num_gauss;
  for(i=0;i<num_gauss;i++){
    mge.tot_counts[i]=L[i];
    mge.sigma[i]=pow(10,p0[i]);
  }
  free(p0);
  free(L);
  return mge;
}
