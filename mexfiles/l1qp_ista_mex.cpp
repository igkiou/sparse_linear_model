/*
 * l1qp_ista_mex.c
 *
 *  Created on: Aug 24, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 6) {
		ERROR("Six or fewer input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *D = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *lambda = (DOUBLE*) mxGetData(prhs[2]);

	INT N = (INT) mxGetM(prhs[1]);
	INT K = (INT) mxGetN(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	INT numRepeats = (INT) mxGetNumberOfElements(prhs[2]);

	DOUBLE tolerance;
	if ((nrhs >= 4) && (mxGetNumberOfElements(prhs[3]) != 0)) {
		tolerance = *(DOUBLE*) mxGetData(prhs[3]);
	} else {
		tolerance = 0.00000001;
	}

	DOUBLE delta;
	if ((nrhs >= 5) && (mxGetNumberOfElements(prhs[4]) != 0)) {
		delta = *(DOUBLE*) mxGetData(prhs[4]);
	} else {
		delta = 0.001;
	}

	INT numIters;
	if ((nrhs >= 6) && (mxGetNumberOfElements(prhs[5]) != 0)) {
		numIters = (INT)*(DOUBLE*) mxGetData(prhs[5]);
	} else {
		numIters = 50000;
	}

	if ((INT) mxGetM(prhs[0]) != N) {
		ERROR("First dimension of X matrix does not match dimension size (first dimension of D matrix).");
	}

	if (delta < 0) {
		qp_lipschitz(&delta, D, N, K, NULL, 0);
	}

	/*DOUBLE *Sin;
	if ((nrhs < 4) || (mxGetNumberOfElements(prhs[3]) == 0)) {
		Sin = MALLOC(K * numSamples * sizeof(DOUBLE));
		memset((DOUBLE *) Sin, 0, K * numSamples * sizeof(DOUBLE));
	} else {
		Sin = (DOUBLE*) mxGetData(prhs[3]);
		if ((INT) mxGetM(prhs[3]) != K) {
			ERROR("First dimension of Sin matrix does not match dictionary size (second dimension of D matrix).");
		} else if ((INT) mxGetN(prhs[3]) != numSamples) {
			ERROR("Second dimension of Sin matrix does not match number of samples (second dimension of X matrix).");
		}
	}*/

	plhs[0] = mxCreateNumericMatrix(K, numSamples, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *A = (DOUBLE *) mxGetData(plhs[0]);

	l1qp_ista(A, X, D, lambda, delta, numIters, tolerance, N, K, numSamples, numRepeats);
}


