/*
 * matrix_dictionary_kernel_obj_grad_mex.c
 *
 *  Created on: May 6, 2011
 *      Author: igkiou
 */

/*
#define __DEBUG__
*/

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 7) {
		ERROR("Seven input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *D = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *A = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *Ksq = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE mu = * (DOUBLE*) mxGetData(prhs[4]);
	INT M = (INT) * (DOUBLE*) mxGetData(prhs[5]);
	INT N = (INT) * (DOUBLE*) mxGetData(prhs[6]);

	INT K = (INT) mxGetM(prhs[2]);
	INT numSamples = (INT) mxGetN(prhs[1]);

	if ((INT) mxGetM(prhs[0]) != M * N) {
		ERROR("First dimension of matrix dictionary does not match signal dimension.");
	} else if ((INT) mxGetN(prhs[0]) != K) {
		ERROR("Second dimension of matrix dictionary does not match code dimension (first dimension of A matrix).");
	} else if ((INT) mxGetN(prhs[2]) != numSamples) {
		ERROR("Second dimension of code matrix does not match number of samples (first dimension of X matrix).");
	} else if ((INT) mxGetM(prhs[3]) != N) {
		ERROR("First dimension of kernel matrix does not match number of frequency samples (second dimension of dictionary atoms).");
	} else if ((INT) mxGetN(prhs[3]) != N) {
		ERROR("Second dimension of kernel matrix does not match number of frequency samples (second dimension of dictionary atoms).");
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
	DOUBLE *Dt = (DOUBLE *) MALLOC(M * N * K * sizeof(DOUBLE));
	DOUBLE *derivTemp;
	if (derivFlag == 1) {
		derivTemp = (DOUBLE *) MALLOC(M * N * K * sizeof(DOUBLE));
	} else {
		derivTemp = NULL;
	}

	matrix_dictionary_kernel_obj_grad(obj, deriv, D, X, A, Ksq, mu, M, N, K, numSamples, derivFlag, res, Dt, derivTemp);

	FREE(res);
	FREE(Dt);
	if (derivFlag == 1) {
		FREE(derivTemp);
	}
}
