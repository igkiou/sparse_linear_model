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
	if (nrhs > 4) {
		ERROR("Four or less input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }
	
	DOUBLE *D = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *S = (DOUBLE*) mxGetData(prhs[2]);
	CHAR familyName;
	if (nrhs == 4) {
		familyName = (CHAR)*(CHAR*) mxGetData(prhs[3]);
	} else {
		familyName = 'P';
	}
	
	INT N = (INT) mxGetM(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[1]);
	INT K = (INT) mxGetM(prhs[2]);
	
	if ((INT) mxGetNumberOfElements(prhs[0]) != N * K) {
		ERROR("Number of elements of dictionary does not match signal dimension (first dimension of data matrix) times dictionary size (first dimension of sparse code matrix).");
	} else if ((INT) mxGetN(prhs[2]) != numSamples) {
		ERROR("Second dimension of sparse code matrix does not match number of samples (second dimension of data matrix).");
	}
	
	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	INT derivFlag;
	
	if (nlhs == 2) {
		plhs[1] = mxCreateNumericMatrix(N, K, MXPRECISION_CLASS, mxREAL);
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}
	
	DOUBLE *DS = (DOUBLE *) MALLOC(N * numSamples * sizeof(DOUBLE));
	DOUBLE *aVal = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
	DOUBLE *aPrime = NULL;
	if (derivFlag == 1) {
		aPrime = (DOUBLE *) MALLOC(N * numSamples * sizeof(DOUBLE));
	}
	EXPONENTIAL_TYPE family = convertExponentialName(familyName);
	
	basis_exp_obj_grad(obj, deriv, D, X, S, N, K, numSamples, family, derivFlag, DS, aVal, aPrime);
	
	FREE(DS);
	FREE(aVal);
	if (derivFlag == 1) {
		FREE(aPrime);
	}
}
