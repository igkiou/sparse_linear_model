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
	if (nrhs > 5) {
		ERROR("Five or less input arguments are required.");
    } else if (nrhs < 4) {
		ERROR("At least four input arguments are required.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }
	
	DOUBLE *s = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *Dt = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *x = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE lambda = (DOUBLE)*(DOUBLE*) mxGetData(prhs[3]);
	CHAR familyName;
	if (nrhs >= 5) {
		if (!mxIsChar(prhs[4])) {
			ERROR("Second argument must be of type CHAR.");
		}
		familyName = (CHAR)*(CHAR*) mxGetData(prhs[4]);
	} else {
		familyName = 'P';
	}
	
	INT K = (INT) mxGetNumberOfElements(prhs[0]);
	INT N = (INT) mxGetNumberOfElements(prhs[2]);
	
	if ((INT) mxGetM(prhs[1]) != K) {
		ERROR("First dimension of transposed dictionary does not match sparse code dimension.");
	} else if ((INT) mxGetN(prhs[1]) != N) {
		ERROR("Second dimension of transposed dictionary does not match signal dimension.");
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
	
	DOUBLE *Ds = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *aPrime = NULL;
	if (derivFlag == 1) {
		aPrime = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	}
	EXPONENTIAL_TYPE family = convertExponentialName(familyName);
	
	l1exp_obj_subgrad(obj, deriv, s, Dt, x, N, K, lambda, family, derivFlag, Ds, aPrime);
	
	FREE(Ds);
	if (derivFlag == 1) {
		FREE(aPrime);
	}
}
