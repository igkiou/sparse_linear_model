/*
 * nuclear_proximal_mex.c
 *
 *  Created on: Apr 19, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 2) {
		ERROR("Two input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *tau = (DOUBLE*) mxGetData(prhs[1]);

	INT M = (INT) mxGetM(prhs[0]);
	INT N = (INT) mxGetN(prhs[0]);

	plhs[0] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Xr = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *norm;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
		norm = (DOUBLE *) mxGetData(plhs[1]);
	} else {
		norm = NULL;
	}

	datacpy(Xr, X, M * N);
	INT MINMN = IMIN(M, N);
	INT MAXMN = IMAX(M, N);
	DOUBLE *sv = (DOUBLE *) MALLOC(MINMN * 1 * sizeof(DOUBLE));
	DOUBLE *svecsmall = (DOUBLE *) MALLOC(MINMN * MINMN * sizeof(DOUBLE));
	DOUBLE *sveclarge = (DOUBLE *) MALLOC(MAXMN * MINMN * sizeof(DOUBLE));
	INT lwork = 0;

	nuclear_proximal(Xr, norm, *tau, M, N, sv, svecsmall, sveclarge, NULL, lwork);

	FREE(sv);
	FREE(svecsmall);
	FREE(sveclarge);
}
