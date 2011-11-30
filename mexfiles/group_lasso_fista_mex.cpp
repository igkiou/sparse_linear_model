/*
 * group_lasso_fista_mex.c
 *
 *  Created on: Aug 29, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 10) {
		ERROR("Ten or fewer input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *D = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *lambda = (DOUBLE*) mxGetData(prhs[2]);

	INT N = (INT) mxGetM(prhs[1]);
	INT K = (INT) mxGetN(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	INT numRepeats = (INT) mxGetNumberOfElements(prhs[2]);
	INT numSets = (INT) mxGetNumberOfElements(prhs[3]);

	DOUBLE *setWeights;
	INT setWeightsFlag = 0;
	if ((nrhs >= 5) && (mxGetNumberOfElements(prhs[4]) != 0)) {
		setWeights = (DOUBLE*) mxGetData(prhs[4]);
	} else {
		setWeights = NULL;
		setWeightsFlag = 1;
	}

	DOUBLE *indOrig = (DOUBLE *) MALLOC(K * sizeof(DOUBLE));
	INT *setSizes = (INT *) MALLOC(numSets * sizeof(INT));
	if (setWeightsFlag == 1) {
		setWeights = (DOUBLE *) MALLOC(numSets * sizeof(DOUBLE));
	}
	INT iterSet;
	INT currCount = 0;
	for (iterSet = 0; iterSet < numSets; ++iterSet) {
		setSizes[iterSet] = mxGetNumberOfElements(mxGetCell(prhs[3], iterSet));
		datacpy(&indOrig[currCount], (DOUBLE *) mxGetData(mxGetCell(prhs[3], iterSet)), \
				setSizes[iterSet]);
		currCount += setSizes[iterSet];
		if (setWeightsFlag == 1) {
			setWeights[iterSet] = 1;
		}
	}

	DOUBLE tolerance;
	if ((nrhs >= 6) && (mxGetNumberOfElements(prhs[5]) != 0)) {
		tolerance = *(DOUBLE*) mxGetData(prhs[5]);
	} else {
		tolerance = 0.00000001;
	}

	DOUBLE delta;
	if ((nrhs >= 7) && (mxGetNumberOfElements(prhs[6]) != 0)) {
		delta = *(DOUBLE*) mxGetData(prhs[6]);
	} else {
		delta = 1000;
	}

	INT numIters;
	if ((nrhs >= 8) && (mxGetNumberOfElements(prhs[7]) != 0)) {
		numIters = (INT)*(DOUBLE*) mxGetData(prhs[7]);
	} else {
		numIters = 50000;
	}

	INT lineSearchFlag;
	if ((nrhs >= 9) && (mxGetNumberOfElements(prhs[8]) != 0)) {
		lineSearchFlag = (INT)*(DOUBLE*) mxGetData(prhs[8]);
	} else {
		lineSearchFlag = 0;
	}

	DOUBLE eta;
	if ((nrhs >= 10) && (mxGetNumberOfElements(prhs[9]) != 0)) {
		eta = *(DOUBLE*) mxGetData(prhs[9]);
	} else {
		eta = 1.1;
	}

	if ((INT) mxGetM(prhs[0]) != N) {
		ERROR("First dimension of X matrix does not match dimension size (first dimension of D matrix).");
	} else if ((nrhs >= 5) && (!mxIsEmpty(prhs[4])) && \
		((INT) mxGetNumberOfElements(prhs[4]) != numSets)) {
		ERROR("Number of weights does not agree with number of sets.");
	}

	if (delta < 0) {
		qp_lipschitz(&delta, D, N, K, NULL, 0);
	}


	INT *indLocality = (INT *) MALLOC(K * sizeof(INT));
	convertSetLabelMat(indLocality, indOrig, setSizes, K, numSets);

	plhs[0] = mxCreateNumericMatrix(K, numSamples, MXPRECISION_CLASS, mxREAL);
	DOUBLE *A = (DOUBLE *) mxGetData(plhs[0]);

	DOUBLE *DTemp = (DOUBLE *) MALLOC(N * K * sizeof(DOUBLE));
	copyDictionaryToLocality(DTemp, D, indLocality, N, K);
	DOUBLE *ATemp = (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
	group_lasso_fista(ATemp, X, DTemp, lambda, setSizes, setWeights, delta, \
					numIters, tolerance, lineSearchFlag, eta, N, K, numSamples, \
					numSets, numRepeats);
	copyCodeToOrig(A, ATemp, indLocality, K, numSamples);

	FREE(indOrig);
	FREE(setSizes);
	if (setWeightsFlag == 1) {
		FREE(setWeights);
	}
	FREE(DTemp);
	FREE(ATemp);
}
