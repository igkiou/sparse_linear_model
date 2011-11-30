/*
 * matrix_completion_apg_mex.cpp
 *
 *  Created on: Nov 20, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 9) {
		ERROR("Nine or fewer input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	INT M = (INT) * (DOUBLE *) mxGetData(prhs[0]);
	INT N = (INT) * (DOUBLE *) mxGetData(prhs[1]);
	DOUBLE *matlabObservationMat = (DOUBLE*) mxGetData(prhs[2]);

	INT numObservations = (INT) mxGetM(prhs[2]);
	INT *observedInds = (INT *) MALLOC(numObservations * 2 * sizeof(INT));
	DOUBLE *observedVals = (DOUBLE *) MALLOC(numObservations * 1 * sizeof(DOUBLE));
	if (mxGetN(prhs[2]) != 3) {
		ERROR("Observation matrix does not meet specified format: second dimension not equal to three.");
	}
	convertObservationMat(observedInds, observedVals, matlabObservationMat, \
						numObservations, M, N);

	DOUBLE mu;
	if ((nrhs >= 4) && (mxGetNumberOfElements(prhs[3]) != 0)) {
		mu = * (DOUBLE *) mxGetData(prhs[3]);
	} else {
		DOUBLE *dummy = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
		INT iterObserved;
		INT currentI;
		INT currentJ;
		memset((void *) dummy, 0, M * N * sizeof(DOUBLE));
		for (iterObserved = 0; iterObserved < numObservations; ++iterObserved) {
			currentI = observedInds[iterObserved];
			currentJ = observedInds[numObservations + iterObserved];
			dummy[currentJ * M + currentI] = observedVals[iterObserved];
		}
		qp_lipschitz(&mu, dummy, M, N, NULL, 0);
		mu = 0.99 * 0.00001 * SQRT(mu / 2.0);
		FREE(dummy);
	}

	DOUBLE gamma;
	if ((nrhs >= 5) && (mxGetNumberOfElements(prhs[4]) != 0)) {
		gamma = * (DOUBLE *) mxGetData(prhs[4]);
	} else {
		gamma = 0;
	}

	DOUBLE tolerance;
	if ((nrhs >= 6) && (mxGetNumberOfElements(prhs[5]) != 0)) {
		tolerance = * (DOUBLE *) mxGetData(prhs[5]);
	} else {
		tolerance = 0.000001;
	}

	DOUBLE delta;
	if ((nrhs >= 7) && (mxGetNumberOfElements(prhs[6]) != 0)) {
		delta = *(DOUBLE*) mxGetData(prhs[6]);
	} else {
		delta = 100000;
	}

	INT numIters;
	if ((nrhs >= 8) && (mxGetNumberOfElements(prhs[7]) != 0)) {
		numIters = (INT)*(DOUBLE*) mxGetData(prhs[7]);
	} else {
		numIters = 50000;
	}

	DOUBLE eta;
	if ((nrhs >= 9) && (mxGetNumberOfElements(prhs[8]) != 0)) {
		eta = *(DOUBLE*) mxGetData(prhs[8]);
	} else {
		eta = 0.9;
	}

	DOUBLE kappa = gamma * mu;

	INT initFlag = 0;
	plhs[0] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *B = (DOUBLE *) mxGetData(plhs[0]);

	omp_set_num_threads(16);
	mkl_set_num_threads(16);
	matrix_completion_apg(B, observedInds, observedVals, mu, kappa, numIters, \
						tolerance, delta, eta, initFlag, M, N, numObservations);
	FREE(observedInds);
	FREE(observedVals);
}
