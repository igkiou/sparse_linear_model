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
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs > 5) {
		ERROR("Five or less arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three arguments are required.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *S = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE l2norm = (DOUBLE)*(DOUBLE*) mxGetData(prhs[2]); 
	INT length;
	if (nrhs > 3) {
		length = (INT)*(DOUBLE*) mxGetData(prhs[3]); 
	} else {
		length = 100000;
	}

    INT N = (INT) mxGetM(prhs[0]);
	INT K = (INT) mxGetM(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	
	DOUBLE *Dorig;
	if (nrhs > 4) {
		Dorig = (DOUBLE *) MALLOC(N * K * sizeof(DOUBLE));
		datacpy(Dorig, (DOUBLE *) mxGetData(prhs[4]), N * K);
	} else {
		Dorig = NULL;
	}
	
	if ((INT) mxGetN(prhs[1]) != numSamples) {
		ERROR("Second dimension of code matrix does not match number of samples (second dimension of signal matrix).");
	} else if ((nrhs > 4) && ((INT) mxGetM(prhs[4]) != N)) {
		ERROR("First dimension of initial dictionary does not match signal dimension (first dimension of signal matrix).");
	} else if ((nrhs > 4) && ((INT) mxGetN(prhs[4]) != K)) {
		ERROR("Second dimension of initial dictionary does not match number of atoms (first dimension of code matrix).");
	}
	
	plhs[0] = mxCreateNumericMatrix(N, K, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Dopt = (DOUBLE *) mxGetData(plhs[0]);

	l2ls_learn_basis_dual(Dopt, Dorig, X, S, l2norm, length, N, K, numSamples);
	
	if (nrhs > 4) {
		FREE(Dorig);
	}
}
