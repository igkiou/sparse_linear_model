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
	if (nrhs > 2) {
		ERROR("Two or less input arguments are required.");
    } else if (nrhs < 1) {
		ERROR("At least one input argument is required.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 3) {
		ERROR("Too many output arguments.");
    }
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	CHAR familyName;
	if (nrhs >= 2) {
		if (!mxIsChar(prhs[1])) {
			ERROR("Second argument must be of type CHAR.");
		}
		familyName = (CHAR)*(CHAR*) mxGetData(prhs[1]);
	} else {
		familyName = 'P';
	}
	
	INT N = (INT) mxGetM(prhs[0]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	
	plhs[0] = mxCreateNumericMatrix(1, numSamples, MXPRECISION_CLASS, mxREAL);
	DOUBLE *aVal = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *aPrime = NULL;
	DOUBLE *aDoublePrime = NULL;
	INT valFlag = 1;
	INT primeFlag = 0;
	INT doublePrimeFlag = 0;
	
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(N, numSamples, MXPRECISION_CLASS, mxREAL);
		aPrime = (DOUBLE *) mxGetData(plhs[1]);
		primeFlag = 1;
	}
	if (nlhs >= 3) {
		plhs[2] = mxCreateNumericMatrix(N, numSamples, MXPRECISION_CLASS, mxREAL);
		aDoublePrime = (DOUBLE *) mxGetData(plhs[2]);
		doublePrimeFlag = 1;
	}
	EXPONENTIAL_TYPE family = convertExponentialName(familyName);
	
	link_func_dual(aVal, aPrime, aDoublePrime, X, N, numSamples, family, \
			valFlag, primeFlag, doublePrimeFlag);
}
