/*
 * group_lasso_proximal_mex.c
 *
 *  Created on: Aug 29, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 4) {
		ERROR("Four or fewer input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *tau = (DOUBLE*) mxGetData(prhs[1]);
//	mxArray *indexSets = (mxArray *) mxGetCell(prhs[2]);
	DOUBLE *setWeights;
	INT setWeightsFlag = 0;
	if ((nrhs >= 4) && (mxGetNumberOfElements(prhs[3]) != 0)) {
		setWeights = (DOUBLE*) mxGetData(prhs[3]);
	} else {
		setWeights = NULL;
		setWeightsFlag = 1;
	}

	INT K = (INT) mxGetNumberOfElements(prhs[0]);
	INT numSets = (INT) mxGetNumberOfElements(prhs[2]);
	if ((nrhs >= 4) && (!mxIsEmpty(prhs[3])) && \
		((INT) mxGetNumberOfElements(prhs[3]) != numSets)) {
		ERROR("Number of weights does not agree with number of sets.");
	}

	DOUBLE *indOrig = (DOUBLE *) MALLOC(K * sizeof(DOUBLE));
	INT *setSizes = (INT *) MALLOC(numSets * sizeof(INT));
	if (setWeightsFlag == 1) {
		setWeights = (DOUBLE *) MALLOC(numSets * sizeof(DOUBLE));
	}
	INT iterSet;
	INT currCount = 0;
	for (iterSet = 0; iterSet < numSets; ++iterSet) {
		setSizes[iterSet] = mxGetNumberOfElements(mxGetCell(prhs[2], iterSet));
		datacpy(&indOrig[currCount], (DOUBLE *) mxGetData(mxGetCell(prhs[2], iterSet)), \
				setSizes[iterSet]);
		currCount += setSizes[iterSet];
		if (setWeightsFlag == 1) {
			setWeights[iterSet] = 1;
		}
	}

	INT *indLocality = (INT *) MALLOC(K * sizeof(INT));
	convertSetLabelMat(indLocality, indOrig, setSizes, K, numSets);

	plhs[0] = mxCreateNumericMatrix(K, 1, MXPRECISION_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Xr = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *norm = (DOUBLE *) mxGetData(plhs[1]);

	DOUBLE *XTemp = (DOUBLE *) MALLOC(K * sizeof(DOUBLE));
	copyCodeToLocality(XTemp, X, indLocality, K, 1);
	group_lasso_proximal(XTemp, norm, *tau, setSizes, setWeights, numSets);
	copyCodeToOrig(Xr, XTemp, indLocality, K, 1);

	FREE(indOrig);
	FREE(setSizes);
	if (setWeightsFlag == 1) {
		FREE(setWeights);
	}
	FREE(XTemp);
}
