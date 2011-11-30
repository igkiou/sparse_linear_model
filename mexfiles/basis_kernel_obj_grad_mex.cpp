/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <mkl.h>
#include <omp.h>
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs > 6) {
		ERROR("Six or less arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three arguments are required.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *D = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *A = (DOUBLE*) mxGetData(prhs[2]);
	CHAR kernelName;
	if (nrhs >= 4) {
		if (!mxIsChar(prhs[3])) {
			ERROR("Fourth argument must be of type CHAR.");
		}
		kernelName = (CHAR)*(CHAR*) mxGetData(prhs[3]);
	} else {
		kernelName = 'G';
	}
	DOUBLE *param1;
	DOUBLE *param2;
	if (nrhs >= 5) {
		param1 = (DOUBLE*) mxGetData(prhs[4]);
	} else {
		param1 = NULL;
	}
	if (nrhs >= 6) {
		param2 = (DOUBLE*) mxGetData(prhs[5]);
	} else {
		param2 = NULL;
	}
	
    INT N = (INT) mxGetM(prhs[1]);
	INT K = (INT) mxGetM(prhs[2]);
	INT numSamples = (INT) mxGetN(prhs[1]);
	
	if ((INT) mxGetNumberOfElements(prhs[0]) != N * K) {
		ERROR("Number of elements of dictionary does not match signal dimension (first dimension of data matrix) times dictionary size (first dimension of sparse code matrix).");
	} else if ((INT) mxGetN(prhs[2]) != numSamples) {
		ERROR("Second dimension of code matrix does not match number of samples (second dimension of signal matrix).");
	} else if ((kernelName != 'G') && (kernelName != 'g')) {
		ERROR("Function basis_kernelobj_grad is only implemented for Gaussian kernel.");
	}
	
	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	INT derivFlag;
	
	if (nlhs == 2) {
		plhs[1] = mxCreateNumericMatrix(N, K, MXPRECISION_CLASS, mxREAL);
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}
	
	DOUBLE *KDD = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
	DOUBLE *KDX = (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
	DOUBLE *KDDDX = (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
	DOUBLE *normMat1 = (DOUBLE *) MALLOC(IMAX(K, numSamples) * 1 * sizeof(DOUBLE));
	DOUBLE *oneVec = (DOUBLE *) MALLOC(IMAX(K, numSamples) * 1 * sizeof(DOUBLE));
	DOUBLE *ak;
	DOUBLE *tempMat1;
	DOUBLE *tempMat2;
	if (derivFlag == 1) {
		ak = (DOUBLE *) MALLOC(numSamples * 1 * sizeof(DOUBLE));
		tempMat1 = (DOUBLE *) MALLOC(numSamples * K * sizeof(DOUBLE));
		tempMat2 = (DOUBLE *) MALLOC(numSamples * N * sizeof(DOUBLE));
	} else {
		ak = NULL;
		tempMat1 = NULL;
		tempMat2 = NULL;
	}
	KERNEL_TYPE kernelType = convertKernelName(kernelName);
	basis_kernel_obj_grad(obj, deriv, D, X, A, N, K, numSamples, kernelType, param1, param2,\
				derivFlag, KDD, KDX, KDDDX, ak, normMat1, oneVec, tempMat1, tempMat2);
	
	FREE(KDD);
	FREE(KDX);
	FREE(KDDDX);
	FREE(normMat1);
	FREE(oneVec);
	if (derivFlag == 1) {
		FREE(ak);
		FREE(tempMat1);
		FREE(tempMat2);
	}
}
