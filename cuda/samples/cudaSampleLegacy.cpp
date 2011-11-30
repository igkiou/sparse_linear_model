/* Verdict: works OK. */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include <math.h>
#include "cublas.h"

typedef float DOUBLE;
#define mxPRECISION_CLASS mxSINGLE_CLASS
#define CUGEMM cublasSgemm
#define MXISDOUBLE mxIsSingle

//typedef double DOUBLE;
//#define mxPRECISION_CLASS mxDOUBLE_CLASS
//#define CUGEMM cublasDgemm
//#define MXISDOUBLE mxIsDouble

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
	cublasInit();

	DOUBLE* d_A = 0;
	DOUBLE* d_B = 0;
	DOUBLE* d_C = 0;

	/* Allocate device memory for the matrices */
	if (cudaMalloc((void**)&d_A, M * K * sizeof(d_A[0])) != cudaSuccess) {
		mexErrMsgTxt("!!!! device memory allocation error (allocate A)\n");
	}
	if (cudaMalloc((void**)&d_B, L * N * sizeof(d_B[0])) != cudaSuccess) {
		mexErrMsgTxt("!!!! device memory allocation error (allocate B)\n");

	}
	if (cudaMalloc((void**)&d_C, MM * NN * sizeof(d_C[0])) != cudaSuccess) {
		mexErrMsgTxt("!!!! device memory allocation error (allocate C)\n");
	}

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
		CUGEMM(transa, transb, MM, NN, KK, alpha, d_A, M, d_B, L, beta, d_C, MM);
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
}
