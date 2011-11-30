/*
 * matrix_dictionary_obj_grad_mex.c
 *
 *  Created on: May 3, 2011
 *      Author: igkiou
 */

/*
#define __DEBUG__
*/

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

	DOUBLE *D = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *A = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE mu = * (DOUBLE*) mxGetData(prhs[3]);
	INT M = (INT) * (DOUBLE*) mxGetData(prhs[4]);
	INT N = (INT) * (DOUBLE*) mxGetData(prhs[5]);

	INT K = (INT) mxGetM(prhs[2]);
	INT numSamples = (INT) mxGetN(prhs[1]);

	if ((INT) mxGetM(prhs[0]) != M * N) {
		ERROR("First dimension of matrix dictionary does not match signal dimension.");
	} else if ((INT) mxGetN(prhs[0]) != K) {
		ERROR("Second dimension of matrix dictionary does not match code dimension (first dimension of A matrix).");
	} else if ((INT) mxGetN(prhs[2]) != numSamples) {
		ERROR("Second dimension of code matrix does not match number of samples (first dimension of X matrix).");
	}

	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	INT derivFlag;
	DOUBLE *deriv;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(M * N * K, 1, MXPRECISION_CLASS, mxREAL); /* x */
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}
	DOUBLE *res = (DOUBLE *) MALLOC(M * N * numSamples * sizeof(DOUBLE));

	matrix_dictionary_obj_grad(obj, deriv, D, X, A, mu, M, N, K, numSamples, derivFlag, res);

	FREE(res);
}
