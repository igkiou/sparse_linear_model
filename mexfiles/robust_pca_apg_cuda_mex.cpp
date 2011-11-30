/*
 * robust_pca_apg_cuda_mex.cpp
 *
 *  Created on: Nov 13, 2011
 *      Author: igkiou
 */

/*
 * matrix_dictionary_learning_lowrank_apg_cuda_mex.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 8) {
		ERROR("Eight or fewer input arguments are required.");
	} else if (nrhs < 1) {
		ERROR("At least one input argument is required.");
	}

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
	}

	DOUBLE *h_D = (DOUBLE *) mxGetData(prhs[0]);

	INT M = (INT) mxGetM(prhs[0]);
	INT N = (INT) mxGetN(prhs[0]);

	DOUBLE mu;
	if ((nrhs >= 2) && (mxGetNumberOfElements(prhs[1]) != 0)) {
		mu = * (DOUBLE *) mxGetData(prhs[1]);
	} else {
		qp_lipschitz(&mu, h_D, M, N, NULL, 0);
		mu = 0.99 * 0.00001 * SQRT(mu / 2.0);
	}

	DOUBLE lambda;
	if ((nrhs >= 3) && (mxGetNumberOfElements(prhs[2]) != 0)) {
		lambda = * (DOUBLE *) mxGetData(prhs[2]);
	} else {
		lambda = 1.0 / SQRT((DOUBLE) IMAX(M, N));
	}

	DOUBLE gamma;
	if ((nrhs >= 4) && (mxGetNumberOfElements(prhs[3]) != 0)) {
		gamma = * (DOUBLE *) mxGetData(prhs[3]);
	} else {
		gamma = 0;
	}

	DOUBLE tolerance;
	if ((nrhs >= 5) && (mxGetNumberOfElements(prhs[4]) != 0)) {
		tolerance = * (DOUBLE *) mxGetData(prhs[4]);
	} else {
		tolerance = 0.000001;
	}

	DOUBLE delta;
	if ((nrhs >= 6) && (mxGetNumberOfElements(prhs[5]) != 0)) {
		delta = *(DOUBLE*) mxGetData(prhs[5]);
	} else {
		delta = 100000;
	}

	INT numIters;
	if ((nrhs >= 7) && (mxGetNumberOfElements(prhs[6]) != 0)) {
		numIters = (INT)*(DOUBLE*) mxGetData(prhs[6]);
	} else {
		numIters = 50000;
	}

	DOUBLE eta;
	if ((nrhs >= 8) && (mxGetNumberOfElements(prhs[7]) != 0)) {
		eta = *(DOUBLE*) mxGetData(prhs[7]);
	} else {
		eta = 0.9;
	}

	INT initFlag = 0;
	plhs[0] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *h_B = (DOUBLE *) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *h_A = (DOUBLE *) mxGetData(plhs[1]);

	DOUBLE kappa = gamma * mu;

	culaInitialize();
	CUHANDLE handle;
	cublasInitialize(&handle);
	robust_pca_apg_cuda(handle, h_B, h_A, h_D, mu, lambda, kappa, numIters, \
						tolerance, delta, eta, initFlag, M, N);
	culaShutdown();
	cublasShutdown(handle);
}
