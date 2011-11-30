/*
 * itml_mex.c
 *
 *  Created on: Jul 15, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 7) {
		ERROR("Seven or fewer input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *A0 = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *matlabConstraintMat = (DOUBLE*) mxGetData(prhs[2]);

	INT N = (INT) mxGetM(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[1]);
	INT numConstraints = (INT) mxGetM(prhs[2]);

	DOUBLE tolerance;
	if (nrhs >= 4) {
		tolerance = *(DOUBLE*) mxGetData(prhs[3]);
	} else {
		tolerance = 0.00001;
	}
	DOUBLE gamma;
	if (nrhs >= 5) {
		gamma = *(DOUBLE*) mxGetData(prhs[4]);
	} else {
		gamma = 1.0;
	}
	INT maxEpochs;
	if (nrhs >= 6) {
		maxEpochs = (INT)*(DOUBLE*) mxGetData(prhs[5]);
	} else {
		maxEpochs = ((INT) 100000) / numConstraints;
	}
	INT randomizationFlag;
	if (nrhs >= 7) {
		randomizationFlag = (INT)*(DOUBLE*) mxGetData(prhs[6]);
		if (randomizationFlag > 0) {
			randomizationFlag = 1;
		}
	} else {
		randomizationFlag = 0;
	}

	if (mxGetM(prhs[0]) != N) {
		ERROR("First dimension of initialization matrix does not match signal dimension (first dimension of the sample matrix).");
	} else if (mxGetN(prhs[0]) != N) {
		ERROR("Second dimension of initialization matrix does not match signal dimension (first dimension of the sample matrix).");
	} else if (mxGetN(prhs[2]) != 4) {
		ERROR("Constraint matrix does not meet specified format: second dimension not equal to four.");
	}

	plhs[0] = mxCreateNumericMatrix(N, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *A = (DOUBLE *) mxGetData(plhs[0]);
	matcpy(A, A0, N, N, 'A');

	INT *constraintMat  = (INT *) MALLOC(numConstraints * 3 * sizeof(DOUBLE));
	DOUBLE *boundVec = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));

	DIST_LABEL_TYPE labelType = DIST_LABEL_BOUNDS;
	convertDistanceLabelMat(constraintMat, boundVec, labelType, \
					matlabConstraintMat, numConstraints, numSamples);

	INT *constraintPerm = (INT *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
	DOUBLE *lambda = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
	DOUBLE *lambdaOld = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
	DOUBLE *vec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *Avec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));

	itml(A, X, constraintMat, boundVec, tolerance, gamma, maxEpochs, \
		randomizationFlag, N, numConstraints, constraintPerm, lambda, lambdaOld, \
		vec, Avec);

	FREE(constraintMat);
	FREE(boundVec);
	FREE(constraintPerm);
	FREE(lambda);
	FREE(lambdaOld);
	FREE(vec);
	FREE(Avec);
}
