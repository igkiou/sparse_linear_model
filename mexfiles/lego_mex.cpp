/*
 * lego_mex.c
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
	if (nrhs > 8) {
		ERROR("Eight or fewer input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *A0 = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *matlabPairMat = (DOUBLE*) mxGetData(prhs[2]);

	INT N = (INT) mxGetM(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[1]);
	INT numPairs = (INT) mxGetM(prhs[2]);

	DOUBLE tolerance;
	if (nrhs >= 4) {
		tolerance = *(DOUBLE*) mxGetData(prhs[3]);
	} else {
		tolerance = 0.00001;
	}
	DOUBLE eta0;
	if (nrhs >= 5) {
		eta0 = *(DOUBLE*) mxGetData(prhs[4]);
	} else {
		eta0 = 1.0;
	}
	INT maxEpochs;
	if (nrhs >= 6) {
		maxEpochs = (INT)*(DOUBLE*) mxGetData(prhs[5]);
	} else {
		maxEpochs = ((INT) 100000) / numPairs;
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
	INT continuationFlag;
	if (nrhs >= 8) {
		continuationFlag = (INT)*(DOUBLE*) mxGetData(prhs[6]);
		if (continuationFlag > 0) {
			continuationFlag = 1;
		}
	} else {
		continuationFlag = 0;
	}

	if (mxGetM(prhs[0]) != N) {
		ERROR("First dimension of initialization matrix does not match signal dimension (first dimension of the sample matrix).");
	} else if (mxGetN(prhs[0]) != N) {
		ERROR("Second dimension of initialization matrix does not match signal dimension (first dimension of the sample matrix).");
	} else if (mxGetN(prhs[2]) != 3) {
		ERROR("Pair matrix does not meet specified format: second dimension not equal to three.");
	}

	plhs[0] = mxCreateNumericMatrix(N, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *A = (DOUBLE *) mxGetData(plhs[0]);
	matcpy(A, A0, N, N, 'A');

	INT *pairMat  = (INT *) MALLOC(numPairs * 2 * sizeof(DOUBLE));
	DOUBLE *distVec = (DOUBLE *) MALLOC(numPairs * 1 * sizeof(DOUBLE));

	DIST_LABEL_TYPE labelType = DIST_LABEL_TARGETS;
	convertDistanceLabelMat(pairMat, distVec, labelType, \
					matlabPairMat, numPairs, numSamples);

	DOUBLE *AOld = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT *pairPerm = (INT *) MALLOC(numPairs * 1 * sizeof(DOUBLE));
	DOUBLE *vec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *Avec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));

	lego(A, X, pairMat, distVec, tolerance, eta0, maxEpochs, randomizationFlag, \
		continuationFlag, N, numPairs, AOld, pairPerm, vec, Avec);

	FREE(pairMat);
	FREE(distVec);
	FREE(AOld);
	FREE(pairPerm);
	FREE(vec);
	FREE(Avec);
}
