/* 
MEX interface for TQLB. Matlab calling sequence:  
  [lambda,top,bot,err] = tqlb(alpha,beta)   
*/

/* TODO: Transfer TQLB from fortran to C */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
/*
#include <algorithm>
*/

#include <mkl.h>
#include <omp.h>
#include "mex.h"
#include "matrix.h"
#include "../useblas.h"
#include "../sparse_classification.h""

/* Template for tqlb: */
/*
void tqlb_(INT *n, DOUBLE *d__, DOUBLE *e, DOUBLE *bnd, 
	   DOUBLE *bnd2, INT *ierr);
*/

/* Here comes the gateway function to be called by Matlab: */
void mexFunction(int nlhs, mxArray *plhs[], 
		 int nrhs, const mxArray *prhs[])
{
  INT m, n,i, ierr;
  DOUBLE x, *tmp;

  if (nrhs != 2)
     mexErrMsgTxt("tqlb requires two input arguments");
  else if  (nlhs != 4)
     mexErrMsgTxt("tqlb requires four output arguments");

  for (i=0; i<2; i++) { 
    m = mxGetM(prhs[i]); /* get the dimensions of the input */
    n = mxGetN(prhs[i]);
    
    /* make sure input is m x 1 */
    if (n != 1) 
      mexErrMsgTxt("Input must be a m x 1 vectors");
  }

  /* Create/allocate return argument, a 1x1 real-valued Matrix */
  for (i=0; i<3; i++) { 
    plhs[i]=mxCreateDoubleMatrix(m,1,mxREAL); 
  }
  plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);   
  tmp = mxCalloc(m,sizeof(DOUBLE));
  
  memcpy(mxGetPr(plhs[0]), mxGetPr(prhs[0]),m*sizeof(DOUBLE));
  memcpy(tmp,mxGetPr(prhs[1]), m*sizeof(DOUBLE));
  TQLB(&m,mxGetPr(plhs[0]),tmp,mxGetPr(plhs[1]),
	mxGetPr(plhs[2]),&ierr);
  
  *(mxGetPr(plhs[3])) = (DOUBLE) ierr;
  mxFree(tmp);
}
