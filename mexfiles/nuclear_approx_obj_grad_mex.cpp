/*
 * nuclear_approx_obj_grad_mex.c
 *
 *  Created on: Mar 25, 2011
 *      Author: igkiou
 */

/*
#define __DEBUG__
*/

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 4) {
		ERROR("Two input arguments are required.");
	}

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *rp = (DOUBLE*) mxGetData(prhs[1]);
	INT M = (INT) * (DOUBLE *) mxGetData(prhs[2]);
	INT N = (INT) * (DOUBLE *) mxGetData(prhs[3]);
//  INT M = (INT) mxGetM(prhs[0]);
//	INT N = (INT) mxGetN(prhs[0]);

	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	INT derivFlag;

	if (nlhs == 2) {
		plhs[1] = mxCreateNumericMatrix(M * N, 1, MXPRECISION_CLASS, mxREAL);
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}

	INT MN = IMIN(M, N);
	DOUBLE *svdVec = (DOUBLE *) MALLOC(1 * MN * sizeof(DOUBLE));
	DOUBLE *dataBuffer = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *derivVec = NULL;
	DOUBLE *vtMat = NULL;
	if (derivFlag == 1) {
		derivVec = (DOUBLE *) MALLOC(1 * MN * sizeof(DOUBLE));
		vtMat = (DOUBLE *) MALLOC(MN * N * sizeof(DOUBLE));
	}

	nuclear_approx_obj_grad(obj, deriv, X, rp, M, N, derivFlag, svdVec, vtMat, \
						dataBuffer, derivVec, NULL, 0);

	FREE(svdVec);
	FREE(dataBuffer);
	if (derivFlag == 1) {
		FREE(derivVec);
		FREE(vtMat);
	}
}
