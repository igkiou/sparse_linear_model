/*
 * nuclear_psd_proximal_mex.c
 *
 *  Created on: Apr 26, 2011
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
	DOUBLE tau = * (DOUBLE*) mxGetData(prhs[1]);

	INT M = (INT) mxGetM(prhs[0]);

	if ((INT) mxGetN(prhs[0]) != M) {
		ERROR("Input matrix is not symmetric, as second dimension does not match first dimension.");
	}

	plhs[0] = mxCreateNumericMatrix(M, M, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Xr = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *norm;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
		norm = (DOUBLE *) mxGetData(plhs[1]);
	} else {
		norm = NULL;
	}

	datacpy(Xr, X, M * M);
	DOUBLE *eigv = (DOUBLE *) MALLOC(M * 1 * sizeof(DOUBLE));
	DOUBLE *eigvec = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
	INT lwork = 0;

	nuclear_psd_proximal(Xr, norm, tau, M, eigv, eigvec, NULL, lwork);

	FREE(eigv);
	FREE(eigvec);
}
