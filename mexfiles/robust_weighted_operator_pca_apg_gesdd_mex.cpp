/*
 * robust_weighted_operator_pca_apg_gesdd_mex.cpp
 *
 *  Created on: Nov 16, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 10) {
		ERROR("Ten or fewer input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	/* D is M x N */
	DOUBLE *D = (DOUBLE *) mxGetData(prhs[0]);
	/* Y is N x K */
	DOUBLE *Y = (DOUBLE *) mxGetData(prhs[1]);
	/* W is N x N */
	DOUBLE *W = (DOUBLE *) mxGetData(prhs[2]);

	INT M = (INT) mxGetM(prhs[0]);
	INT N = (INT) mxGetN(prhs[0]);
	INT K = (INT) mxGetN(prhs[1]);

	if ((INT) mxGetM(prhs[1]) != N) {
		ERROR("First dimension of kernel factor matrix Y does not match second modality dimension (second dimension of D matrix).");
	} else if ((INT) mxGetM(prhs[2]) != N) {
		ERROR("First dimension of weight matrix W second modality dimension (second dimension of D matrix).");
	} else if ((INT) mxGetN(prhs[2]) != N) {
		ERROR("First dimension of weight matrix W second modality dimension (second dimension of D matrix).");
	}

	DOUBLE mu;
	if ((nrhs >= 4) && (mxGetNumberOfElements(prhs[3]) != 0)) {
		mu = * (DOUBLE *) mxGetData(prhs[3]);
	} else {
		qp_lipschitz(&mu, D, M, N, NULL, 0);
		mu = 0.99 * 0.00001 * SQRT(mu / 2.0);
	}

	DOUBLE lambda;
	if ((nrhs >= 5) && (mxGetNumberOfElements(prhs[4]) != 0)) {
		lambda = * (DOUBLE *) mxGetData(prhs[4]);
	} else {
		lambda = 1.0 / SQRT((DOUBLE) IMAX(M, N));
	}

	DOUBLE gamma;
	if ((nrhs >= 6) && (mxGetNumberOfElements(prhs[5]) != 0)) {
		gamma = * (DOUBLE *) mxGetData(prhs[5]);
	} else {
		gamma = 0;
	}

	DOUBLE tolerance;
	if ((nrhs >= 7) && (mxGetNumberOfElements(prhs[6]) != 0)) {
		tolerance = * (DOUBLE *) mxGetData(prhs[6]);
	} else {
		tolerance = 0.000001;
	}

	DOUBLE delta;
	if ((nrhs >= 8) && (mxGetNumberOfElements(prhs[7]) != 0)) {
		delta = *(DOUBLE*) mxGetData(prhs[7]);
	} else {
		delta = 100000;
	}

	INT numIters;
	if ((nrhs >= 9) && (mxGetNumberOfElements(prhs[8]) != 0)) {
		numIters = (INT)*(DOUBLE*) mxGetData(prhs[8]);
	} else {
		numIters = 50000;
	}

	DOUBLE eta;
	if ((nrhs >= 10) && (mxGetNumberOfElements(prhs[9]) != 0)) {
		eta = *(DOUBLE*) mxGetData(prhs[9]);
	} else {
		eta = 0.9;
	}

	DOUBLE kappa = gamma * mu;

	INT initFlag = 0;
	plhs[0] = mxCreateNumericMatrix(M, K, MXPRECISION_CLASS, mxREAL);
	DOUBLE *B = (DOUBLE *) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *A = (DOUBLE *) mxGetData(plhs[1]);

	omp_set_num_threads(16);
	mkl_set_num_threads(16);
	robust_weighted_operator_pca_apg_gesdd(B, A, D, Y, W, mu, lambda, kappa, \
							numIters, tolerance, delta, eta, initFlag, M, N, K);
}
