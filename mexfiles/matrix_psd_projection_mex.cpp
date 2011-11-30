/*
 * matrix_psd_projection_mex.c
 *
 *  Created on: Oct 24, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 1) {
		ERROR("One input argument is required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);

	INT M = (INT) mxGetM(prhs[0]);

	if ((INT) mxGetN(prhs[0]) != M) {
		ERROR("Input matrix is not symmetric, as second dimension does not match first dimension.");
	}

	plhs[0] = mxCreateNumericMatrix(M, M, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Xr = (DOUBLE *) mxGetData(plhs[0]);

	datacpy(Xr, X, M * M);
	DOUBLE *eigv = (DOUBLE *) MALLOC(M * 1 * sizeof(DOUBLE));
	DOUBLE *eigvec = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
	INT lwork = 0;

	matrix_psd_projection(Xr, M, eigv, eigvec, NULL, lwork);

	FREE(eigv);
	FREE(eigvec);
}
