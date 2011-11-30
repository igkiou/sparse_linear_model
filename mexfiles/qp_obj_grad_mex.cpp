/*
 * qp_obj_grad_mex.c
 *
 *  Created on: Apr 19, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 3) {
		ERROR("Three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *A = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *D = (DOUBLE*) mxGetData(prhs[2]);

	INT N = (INT) mxGetM(prhs[1]);
	INT K = (INT) mxGetN(prhs[2]);

	if ((INT) mxGetNumberOfElements(prhs[0]) != K) {
		ERROR("Length of sparse code does not match dictionary size (second dimension of D matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[1]) != N) {
		ERROR("Length of signal does not match dictionary size (first dimension of D matrix).");
	}

	if ((INT) mxGetNumberOfElements(prhs[0]) != K) {
		ERROR("Length of sparse code does not match dictionary size (second dimension of D matrix).");
	} else if ((INT) mxGetN(prhs[1]) != 1) {
		ERROR("Number of elements in KXX is not equal to 1 (single sample).");
	} else if ((INT) mxGetM(prhs[2]) != N) {
		ERROR("First dimension of D matrix does not match signal size (first dimension of X matrix).");
	}

	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(K, 1, MXPRECISION_CLASS, mxREAL); /* x */
		deriv = (DOUBLE *) mxGetData(plhs[1]);
	} else {
		deriv = NULL;
	}
	qp_obj_grad(obj, deriv, A, X, D, N, K, NULL);
}
