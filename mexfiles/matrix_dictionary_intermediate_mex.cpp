/* Verdict: Slower than all device-side GPU implementations. */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include <math.h>

typedef double DOUBLE;
typedef MKL_INT INT;
typedef char CHAR;
#define mxPRECISION_CLASS mxDOUBLE_CLASS
#define GEMM dgemm
#define MXISDOUBLE mxIsDouble

/* Main */
void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {

	if (nrhs != 4) {
		mexErrMsgTxt("matrix_dictionary_intermediate requires 4 input arguments");
	} else if (nlhs != 0) {
		mexErrMsgTxt("dgemm has no output arguments");
	}

	if ( !MXISDOUBLE(prhs[0]) ||
		!MXISDOUBLE(prhs[1]) ||
		!MXISDOUBLE(prhs[2]) ||
		!MXISDOUBLE(prhs[3]))   {
		mexErrMsgTxt("Input arrays must be single precision.");
	}

	
	DOUBLE *X = (DOUBLE *) mxGetData(prhs[0]);
	DOUBLE *A = (DOUBLE *) mxGetData(prhs[1]);
	DOUBLE *XAt = (DOUBLE *) mxGetData(prhs[2]);
	DOUBLE *AAt = (DOUBLE *) mxGetData(prhs[3]);

	INT MN = (INT) mxGetM(prhs[0]);
	INT S = (INT) mxGetN(prhs[0]);
	INT K = (INT) mxGetM(prhs[1]);

	omp_set_num_threads(16);
	mkl_set_num_threads(16);

	CHAR transa = 'N';
	CHAR transb = 'T';
	INT GEMMM = MN;
	INT GEMMN = K;
	INT GEMMK = S;
	DOUBLE alpha = 1.0;
	INT GEMMLDA = MN;
	INT GEMMLDB = K;
	DOUBLE beta = 0.0;
	INT GEMMLDC = MN;
	dgemm(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X, &GEMMLDA, A, &GEMMLDB, &beta, XAt, &GEMMLDC);

	CHAR uplo = 'U';
	CHAR trans = 'N';
	INT SYRKN = K;
	INT SYRKK = S;
	alpha = 1.0;
	INT SYRKLDA = K;
	beta = 0.0;
	INT SYRKLDC = K;
	dsyrk(&uplo, &trans, &SYRKN, &SYRKK, &alpha, A, &SYRKLDA, &beta, AAt, &SYRKLDC);
	INT iterK1;
	INT iterK2;
	#pragma omp parallel for private(iterK1, iterK2) shared(AAt) \
			firstprivate(K)
	for (iterK1 = 0; iterK1 < K; ++iterK1) {
		for (iterK2 = iterK1 + 1; iterK2 < K; ++iterK2) {
			AAt[iterK1 * K + iterK2] = AAt[iterK2 * K + iterK1];
		}
	}
}
