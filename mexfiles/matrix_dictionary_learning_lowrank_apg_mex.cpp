/*
 * matrix_dictionary_learning_lowrank_apg_mex.c
 *
 *  Created on: Oct 19, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 11) {
		ERROR("Eleven or fewer input arguments are required.");
    } else if (nrhs < 6) {
		ERROR("At least six input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *Binit = (DOUBLE *) mxGetData(prhs[0]);
	DOUBLE *XAt = (DOUBLE *) mxGetData(prhs[1]);
	DOUBLE *AAt = (DOUBLE *) mxGetData(prhs[2]);

	INT M = (INT) * (DOUBLE *) mxGetData(prhs[3]);
	INT numSamples = (INT) * (DOUBLE *) mxGetData(prhs[4]);
	INT MN = (INT) mxGetM(prhs[1]);
	INT K = (INT) mxGetN(prhs[1]);
	INT N = MN / M;

	if (((INT) mxGetNumberOfElements(prhs[0]) != 0) && ((INT) mxGetM(prhs[0]) != MN)) {
		ERROR("First dimension of matrix B does not match total signal dimension (first dimension of XAt matrix).");
	} else if (((INT) mxGetNumberOfElements(prhs[0]) != 0) && ((INT) mxGetN(prhs[0]) != K)) {
		ERROR("Second dimension of matrix B does not match number of dictionary atoms (second dimension of XAt matrix).");
	} else if ((INT) mxGetM(prhs[2]) != K) {
		ERROR("First dimension of matrix AAt does not match number of dictionary atoms (second dimension of XAt matrix).");
	} else if ((INT) mxGetN(prhs[2]) != K) {
		ERROR("Second dimension of matrix AAt does not match number of dictionary atoms (second dimension of XAt matrix).");
	}

	DOUBLE mu = * (DOUBLE *) mxGetData(prhs[5]);

	DOUBLE kappa;
	if ((nrhs >= 7) && (mxGetNumberOfElements(prhs[6]) != 0)) {
		kappa = * (DOUBLE *) mxGetData(prhs[6]);
	} else {
		kappa = 0;
	}

	DOUBLE tolerance;
	if ((nrhs >= 8) && (mxGetNumberOfElements(prhs[7]) != 0)) {
		tolerance = * (DOUBLE *) mxGetData(prhs[7]);
	} else {
		tolerance = 0.000001;
	}

	DOUBLE delta;
	if ((nrhs >= 9) && (mxGetNumberOfElements(prhs[8]) != 0)) {
		delta = *(DOUBLE*) mxGetData(prhs[8]);
	} else {
		delta = 0.00001;
	}

	INT numIters;
	if ((nrhs >= 10) && (mxGetNumberOfElements(prhs[9]) != 0)) {
		numIters = (INT)*(DOUBLE*) mxGetData(prhs[9]);
	} else {
		numIters = 50000;
	}

	DOUBLE eta;
	if ((nrhs >= 11) && (mxGetNumberOfElements(prhs[10]) != 0)) {
		eta = *(DOUBLE*) mxGetData(prhs[10]);
	} else {
		eta = 0.9;
	}

	plhs[0] = mxCreateNumericMatrix(MN, K, MXPRECISION_CLASS, mxREAL);
	DOUBLE *B = (DOUBLE *) mxGetData(plhs[0]);
	INT initFlag = 0;
	if ((INT) mxGetNumberOfElements(prhs[0]) != 0) {
		datacpy(B, Binit, MN * K);
		initFlag = 1;
	} else {
		initFlag = 0;
	}
	omp_set_num_threads(16);
	mkl_set_num_threads(16);
	matrix_dictionary_learning_lowrank_apg_parallel(B, XAt, AAt, mu, kappa, tolerance, \
					delta, numIters, eta, initFlag, M, N, K, numSamples);
}
