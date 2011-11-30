/* Verdict: this interface is not supported by CULA. */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

/* Main */
void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {

	if (nrhs != 7) {
		mexErrMsgTxt("sgemm requires 7 input arguments");
	} else if (nlhs != 1) {
		mexErrMsgTxt("sgemm requires 1 output argument");
	}

	if ( !mxIsSingle(prhs[4]) || 
		!mxIsSingle(prhs[5]) ||
		!mxIsSingle(prhs[6]))   {
		mexErrMsgTxt("Input arrays must be single precision.");
	}

	int ta = (int) mxGetScalar(prhs[0]);
	int tb = (int) mxGetScalar(prhs[1]);
	float alpha = (float) mxGetScalar(prhs[2]);
	float beta = (float) mxGetScalar(prhs[3]);
	float *h_A = (float*) mxGetData(prhs[4]);
	float *h_B = (float*) mxGetData(prhs[5]);
	float *h_C = (float*) mxGetData(prhs[6]);

	int M = mxGetM(prhs[4]);   /* gets number of rows of A */
	int K = mxGetN(prhs[4]);   /* gets number of columns of A */
	int L = mxGetM(prhs[5]);   /* gets number of rows of B */
	int N = mxGetN(prhs[5]);   /* gets number of columns of B */

	cublasOperation_t transa, transb;
	int MM, KK, NN;
	if (ta == 0) {
		transa = CUBLAS_OP_N;
		MM=M;
		KK=K;
	} else {
		transa = CUBLAS_OP_T;
		MM=K;
		KK=M;
	}

	if (tb == 0) {
		transb = CUBLAS_OP_N;
		NN=N;
	} else {
		transb = CUBLAS_OP_T;
		NN=L;
	}

/*	printf("transa=%c\n",transa); 
	printf("transb=%c\n",transb); 
	printf("alpha=%f\n",alpha); 
	printf("beta=%f\n",beta);	*/

/* Left hand side matrix set up */
	mwSize dims0[2];
	dims0[0]=MM;
	dims0[1]=NN;
	plhs[0] = mxCreateNumericArray(2,dims0,mxSINGLE_CLASS,mxREAL);
	float *h_C_out = (float*) mxGetData(plhs[0]);
	
	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		mexErrMsgTxt("!!!! CUBLAS initialization error\n");
	}

	float* d_A = 0;
	float* d_B = 0;
	float* d_C = 0;
	
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

	/* Performs operation using cublas */
	status = cublasSgemm(handle, transa, transb, MM, NN, KK, &alpha, d_A, M, d_B, L, &beta, d_C, MM);
	if (status != CUBLAS_STATUS_SUCCESS) {
		mexErrMsgTxt("!!!! kernel execution error.\n");
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
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		mexErrMsgTxt("!!!! shutdown error (A)\n");
	}
}
