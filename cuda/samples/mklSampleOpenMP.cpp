/* Verdict: Can speedup simple MKL, still slower than best GPU. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include <math.h>

typedef float DOUBLE;
#define mxPRECISION_CLASS mxSINGLE_CLASS
#define GEMM sgemm
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

	MKL_INT M = (MKL_INT) mxGetM(prhs[4]);   /* gets number of rows of A */
	MKL_INT K = (MKL_INT) mxGetN(prhs[4]);   /* gets number of columns of A */
	MKL_INT L = (MKL_INT) mxGetM(prhs[5]);   /* gets number of rows of B */
	MKL_INT N = (MKL_INT) mxGetN(prhs[5]);   /* gets number of columns of B */

	char transa, transb;
	MKL_INT MM, KK, NN;
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

/*	printf("transa=%c\n",transa);
	printf("transb=%c\n",transb);
	printf("alpha=%f\n",alpha);
	printf("beta=%f\n",beta);	*/

/* Left hand side matrix set up */
	mwSize dims0[2];
	dims0[0]=MM;
	dims0[1]=NN;
	plhs[0] = mxCreateNumericArray(2,dims0,mxPRECISION_CLASS,mxREAL);
	DOUBLE *h_C_out = (DOUBLE*) mxGetData(plhs[0]);
	DOUBLE *h_C_dummy = (DOUBLE *) malloc(MM * NN * 1000 * sizeof(DOUBLE));

	/* Performs operation using cublas */
	int iter;
	#pragma omp parallel for private(iter) shared(h_A, h_B, h_C_dummy) \
		firstprivate(transa, transb, MM, NN, KK, alpha, M, L, beta)
	for (iter = 0; iter < 1000; ++iter) {
		GEMM(&transa, &transb, &MM, &NN, &KK, &alpha, h_A, &M, h_B, &L, &beta, &h_C_dummy[iter * MM * NN], &MM);
	}
	memcpy((void *) h_C_out, (void *) h_C_dummy, MM * NN * sizeof(h_C[0]));
	free(h_C_dummy);
}
