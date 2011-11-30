/*
 * kernel_ball_projection_mex.cpp
 *
 *  Created on: Nov 14, 2011
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
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *kernelMat = (DOUBLE *) mxGetData(prhs[0]);
	INT K = (INT) mxGetM(prhs[0]);

	if ((INT) mxGetN(prhs[0]) != K) {
		ERROR("Kernel Grammian is not square.");
	}

	DOUBLE radius;
	if ((nrhs >= 2) && (mxGetNumberOfElements(prhs[1]) != 0)) {
		radius = * (DOUBLE *) mxGetData(prhs[1]);
	} else {
		radius = 1;
	}

	plhs[0] = mxCreateNumericMatrix(K, K, MXPRECISION_CLASS, mxREAL);
	DOUBLE *kernelMatNorm = (DOUBLE *) mxGetData(plhs[0]);
	datacpy(kernelMatNorm, kernelMat, K * K);

	DOUBLE *normVec;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(1, K, MXPRECISION_CLASS, mxREAL); /* x */
		normVec = (DOUBLE *) mxGetData(plhs[1]);
	} else {
		normVec = NULL;
	}
	kernel_ball_projection_batch(kernelMatNorm, normVec, radius, K);
}
