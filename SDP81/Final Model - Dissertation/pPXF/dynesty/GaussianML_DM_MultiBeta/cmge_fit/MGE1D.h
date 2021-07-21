struct multigaussexp1d {
    double *tot_counts;
    double *sigma;
    int ntotal;
};
struct multigaussexp1d mge_fit(double *x, double *y,\
        double *error, int num_data, int num_gauss,\
        int imax,double rbound_min, double rbound_max);
