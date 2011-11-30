/*
 * kernel_obj_grad_mex.c
 *
 *  Created on: Aug 24, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 4) {
		ERROR("Four input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *A = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *KXX = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *KDX = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *KDD = (DOUBLE*) mxGetData(prhs[3]);

	INT K = (INT) mxGetM(prhs[2]);

	if ((INT) mxGetNumberOfElements(prhs[0]) != K) {
		ERROR("Length of sparse code does not match dictionary size (first dimension of KDX matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[1]) != 1) {
		ERROR("Number of elements in KXX is not equal to 1 (single sample).");
	} else if ((INT) mxGetN(prhs[2]) != 1) {
		ERROR("Second dimension of KDX matrix is not equal to 1 (single sample).");
	} else if ((INT) mxGetM(prhs[3]) != K) {
		ERROR("First dimension of KDD matrix does not match dictionary size (first dimension of KDX matrix).");
	} else if ((INT) mxGetN(prhs[3]) != K) {
		ERROR("Second dimension of KDD matrix does not match dictionary size (first dimension of KDX matrix).");
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
	kernel_obj_grad(obj, deriv, A, KXX, KDX, KDD, K, NULL);
}
