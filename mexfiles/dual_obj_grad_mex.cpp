/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <mkl.h>
#include <omp.h>
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"
	
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs != 6) {
		ERROR("Six input arguments are required.");
    }
	
	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *dualLambda = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *SSt = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *SXt = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *SXtXSt = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE trXXt = *(DOUBLE*) mxGetData(prhs[4]);
	DOUBLE c = *(DOUBLE*) mxGetData(prhs[5]);
	
	INT K = (INT) mxGetM(prhs[2]);
	INT N = (INT) mxGetN(prhs[2]);
	INT maxNK = IMAX(N, K);
	
	if ((INT) mxGetNumberOfElements(prhs[0]) != K) {
		ERROR("Length of dualLambda does not match number of atoms (first dimension of SXt matrix).");
	} else if ((INT) mxGetM(prhs[1]) != K) {
		ERROR("First dimension of SSt matrix does not match number of atoms (first dimension of SXt matrix).");
	} else if ((INT) mxGetN(prhs[1]) != K) {
		ERROR("Second dimension of SSt matrix does not match number of atoms (first dimension of SXt matrix).");
	} else if ((INT) mxGetM(prhs[3]) != K) {
		ERROR("First dimension of SXtXSt matrix does not match number of atoms (first dimension of SXt matrix).");
	} else if ((INT) mxGetN(prhs[3]) != K) {
		ERROR("Second dimension of SXtXSt matrix does not match number of atoms (first dimension of SXt matrix).");
	}
	
	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	INT derivFlag;
	if (nlhs == 2) {
		plhs[1] = mxCreateNumericMatrix(K, 1, MXPRECISION_CLASS, mxREAL);
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}	
	
	DOUBLE *SStLambda = (DOUBLE *) MALLOC(maxNK * K * sizeof(DOUBLE));
	DOUBLE *tempMatrix = (DOUBLE *) MALLOC(maxNK * K * sizeof(DOUBLE));
	
	dual_obj_grad(obj, deriv, dualLambda, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);
}
