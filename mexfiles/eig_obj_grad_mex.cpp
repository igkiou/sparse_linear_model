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
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs != 5) {
		ERROR("Five input arguments are required.");
    }
	
	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *DDt2 = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *DDt3 = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *VL = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *L = (DOUBLE*) mxGetData(prhs[4]);
	
	INT N = (INT) mxGetN(prhs[1]);
	INT M = ((INT) mxGetNumberOfElements(prhs[0])) / N;
	
	if (M != ((DOUBLE) mxGetNumberOfElements(prhs[0])) / (DOUBLE) N) {
		ERROR("Number of elements of Phi is not compatible with dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetN(prhs[1]) != N) {
		ERROR("Second dimension of DDt2 matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetM(prhs[2]) != N) {
		ERROR("First dimension of DDt3 matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetN(prhs[2]) != N) {
		ERROR("Second dimension of DDt3 matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetM(prhs[3]) != N) {
		ERROR("First dimension of VL matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetN(prhs[3]) != N) {
		ERROR("Second dimension of VL matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[4]) != N) {
		ERROR("Number of elements of L matrix does not match dimension size (first dimension of DDt2 matrix).");
	}	

	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	INT derivFlag;
	
	if (nlhs == 2) {
		plhs[1] = mxCreateNumericMatrix(M * N, 1, MXPRECISION_CLASS, mxREAL);
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}
	
	DOUBLE *Gamma = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *PhiDDt2;
	DOUBLE *DDt3temp;
	
	if (derivFlag == 1) {
		PhiDDt2 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
		DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	} else {
		PhiDDt2 = NULL;
		DDt3temp = NULL;
	}
	
	eig_obj_grad(obj, deriv, X, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
	
	FREE(Gamma);
	FREE(ObjMat);
	
	if (derivFlag == 1) {
		FREE(PhiDDt2);
		FREE(DDt3temp);
	}
}
