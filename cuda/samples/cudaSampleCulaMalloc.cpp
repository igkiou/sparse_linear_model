/* Verdict: does not work. */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include <math.h>
#include "cula_lapack_device.h"
#include "cuda_runtime.h"
#include "cublas.h"

typedef float DOUBLE;
#define mxPRECISION_CLASS mxSINGLE_CLASS
#define CUGEMM culaDeviceSgemm
#define MXISDOUBLE mxIsSingle

/* Main */
void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {

	if (nrhs != 7) {
		mexErrMsgTxt("sgemm requires 7 input arguments");
	} else if (nlhs != 1) {
		mexErrMsgTxt("sgemm requires 1 output argument");
	}

	if ( !MXISDOUBLE(prhs[4]) ||
		!MXISDOUBLE(prhs[5]) ||
		!MXISDOUBLE(prhs[6]))   {
		mexErrMsgTxt("Input arrays must be single precision.");
	}

	int ta = (int) mxGetScalar(prhs[0]);
	int tb = (int) mxGetScalar(prhs[1]);
	DOUBLE alpha = (DOUBLE) mxGetScalar(prhs[2]);
	DOUBLE beta = (DOUBLE) mxGetScalar(prhs[3]);
	DOUBLE *h_A = (DOUBLE*) mxGetData(prhs[4]);
	DOUBLE *h_B = (DOUBLE*) mxGetData(prhs[5]);
	DOUBLE *h_C = (DOUBLE*) mxGetData(prhs[6]);

	int M = mxGetM(prhs[4]);   /* gets number of rows of A */
	int K = mxGetN(prhs[4]);   /* gets number of columns of A */
	int L = mxGetM(prhs[5]);   /* gets number of rows of B */
	int N = mxGetN(prhs[5]);   /* gets number of columns of B */

	char transa, transb;
	int MM, KK, NN;
	if (ta == 0) {
		transa = 'N';
		MM=M;
		KK=K;
	} else {
		transa = 'T';
		MM=K;
		KK=M;
	}

	if (tb == 0) {
		transb = 'N';
		NN=N;
	} else {
		transb = 'T';
		NN=L;
	}


/* Left hand side matrix set up */
	mwSize dims0[2];
	dims0[0]=MM;
	dims0[1]=NN;
	plhs[0] = mxCreateNumericArray(2,dims0,mxPRECISION_CLASS,mxREAL);
	DOUBLE *h_C_out = (DOUBLE*) mxGetData(plhs[0]);

	cublasStatus status;
	culaInitialize();
	cublasInit();

	DOUBLE* d_A = 0;
	DOUBLE* d_B = 0;
	DOUBLE* d_C = 0;

	int LDA;
	int LDB;
	int LDC;
	culaDeviceMalloc((void**)&d_A, &LDA, M, K, sizeof(d_A[0]));
	culaDeviceMalloc((void**)&d_B, &LDB, L, M, sizeof(d_B[0]));
	culaDeviceMalloc((void**)&d_C, &LDC, MM, NN, sizeof(d_C[0]));

	/* Initialize the device matrices with the host matrices */
    status = cublasSetVector(M * K, sizeof(h_A[0]), h_A, 1, d_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		mexErrMsgTxt("!!!! device access error (write A)\n");

	}
	status = cublasSetVector(L * N, sizeof(h_B[0]), h_B, 1, d_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		mexErrMsgTxt("!!!! device access error (write B)\n");
	}
	status = cublasSetVector(MM * NN, sizeof(h_C[0]), h_C, 1, d_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		mexErrMsgTxt("!!!! device access error (write C)\n");
	}

	int iter;
	for (iter = 0; iter < 100000; ++iter) {
	/* Performs operation using cublas */
		CUGEMM(transa, transb, MM, NN, KK, alpha, d_A, LDA, d_B, LDB, beta, d_C, LDC);
	}

	/* Read the result back */
	status = cublasGetVector(MM * NN, sizeof(h_C[0]), d_C, 1, h_C_out, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		mexErrMsgTxt("!!!! device access error (read C)\n");

	}

	if (cudaFree(d_A) != cudaSuccess) {
		mexErrMsgTxt("!!!! memory free error (A)\n");
	}
	if (cudaFree(d_B) != cudaSuccess) {
		mexErrMsgTxt("!!!! memory free error (B)\n");
	}
	if (cudaFree(d_C) != cudaSuccess) {
		mexErrMsgTxt("!!!! memory free error (C)\n");
	}

	/* Shutdown */
	cublasShutdown();
	culaShutdown();
}
