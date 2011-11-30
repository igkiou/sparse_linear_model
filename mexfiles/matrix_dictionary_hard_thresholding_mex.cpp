/*
 * matrix_dictionary_hard_thresholding_mex.c
 *
 *  Created on: May 3, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 4) {
		ERROR("Four input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *Dc = (DOUBLE*) mxGetData(prhs[0]);
	INT rank = (INT) * (DOUBLE*) mxGetData(prhs[1]);
	INT M = (INT) * (DOUBLE*) mxGetData(prhs[2]);
	INT N = (INT) * (DOUBLE*) mxGetData(prhs[3]);

	INT K = (INT) mxGetN(prhs[0]);

	if ((INT) mxGetM(prhs[0]) != M * N) {
		ERROR("First dimension of Dc matrix does not match size of dictionary atoms.");
	} else if (rank > IMIN(M, N)) {
		ERROR("Input rank is larger than minimum dimension of signal matrix.");
	}

	plhs[0] = mxCreateNumericMatrix(M * N, K, MXPRECISION_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Dcr = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *norm = (DOUBLE *) mxGetData(plhs[1]);

	datacpy(Dcr, Dc, M * N * K);
//	INT MINMN = IMIN(M, N);
//	INT MAXMN = IMAX(M, N);
//	DOUBLE *sv = (DOUBLE *) MALLOC(MINMN * 1 * sizeof(DOUBLE));
//	DOUBLE *svecsmall = (DOUBLE *) MALLOC(MINMN * MINMN * sizeof(DOUBLE));
//	DOUBLE *sveclarge = (DOUBLE *) MALLOC(MAXMN * MINMN * sizeof(DOUBLE));
//	DOUBLE *normvec = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
//	INT lwork = 0;
//	matrix_dictionary_proximal(Dcr, norm, tau, M, N, K, sv, svecsmall, sveclarge, normvec, \
			NULL, lwork);

	matrix_dictionary_hard_thresholding_parallel(Dcr, norm, rank, M, N, K);

//	FREE(sv);
//	FREE(svecsmall);
//	FREE(sveclarge);
//	FREE(normvec);
}
