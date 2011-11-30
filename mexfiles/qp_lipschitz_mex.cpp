/*
 * qp_lipschitz_mex.c
 *
 *  Created on: Aug 24, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 2) {
		ERROR("Two or fewer input arguments are required.");
	} else if (nrhs < 1) {
		ERROR("At least one input argument is required.");
	}

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *D = (DOUBLE*) mxGetData(prhs[0]);
	INT precomputedKernelFlag;
	if ((nrhs >= 2) && (mxGetNumberOfElements(prhs[1]) != 0)) {
		precomputedKernelFlag = (INT)*(DOUBLE*) mxGetData(prhs[1]);
	} else {
		precomputedKernelFlag = 0;
	}

	INT N = (INT) mxGetM(prhs[0]);
	INT K = (INT) mxGetN(prhs[0]);

	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *Lf = (DOUBLE *) mxGetData(plhs[0]);

	if (precomputedKernelFlag == 1) {
		qp_lipschitz(Lf, NULL, N, K, D, 1);
	} else {
		qp_lipschitz(Lf, D, N, K, NULL, 0);
	}
}
