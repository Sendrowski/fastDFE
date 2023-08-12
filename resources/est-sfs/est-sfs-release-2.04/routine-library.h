#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


extern void monitorinput();

extern void getseedquick(char *seedfile);

extern void terminate_uniform_generator();

extern void writeseed(char *seedfile);

extern double uniform();

extern FILE *openforreadsilent(char *str);

extern FILE *openforwritesilent(char *str, char *mode);

extern void gabort(char *s, int r);

extern int skiptoendofline(FILE *fptr);

extern int odd(int i);

extern void nrerror(char error_txt[]);

extern double *dvector(long nl, long nh);

extern double **dmatrix(long nrl,long nrh,long ncl,long nch);

extern void free_dvector(double *v,long nl,long nh);

extern void free_dmatrix(double **m,long nrl,long nrh,long ncl,long nch);

extern void amoeba(double **p,double y[],int ndim, double ftol, double(*funk)(double []),int *nfunk, int limit);

extern double amotry(double **p,double y[],double psum[],int ndim,double (*funk)(double []),int ihi,double fac);

double golden(double ax, double bx, double cx, double (*f)(double),
   double tol, double *xmin, int *neval);
