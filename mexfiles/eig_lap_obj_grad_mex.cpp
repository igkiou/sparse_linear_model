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
	
	if (nrhs != 10) {
		ERROR("Ten input arguments are required.");
    }

	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *Phi = (DOUBLE*) mxGetData(prhs[0]);
	INT M = (INT)*(DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *XLXt = (DOUBLE*) mxGetData(prhs[2]);
	INT numSamples = (INT)*(DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *DDt2 = (DOUBLE*) mxGetData(prhs[4]);
	DOUBLE *DDt3 = (DOUBLE*) mxGetData(prhs[5]);
	DOUBLE *VL = (DOUBLE*) mxGetData(prhs[6]);
	DOUBLE *L = (DOUBLE*) mxGetData(prhs[7]);
	DOUBLE alphaReg = *(DOUBLE*) mxGetData(prhs[8]);
	DOUBLE betaReg = *(DOUBLE*) mxGetData(prhs[9]);
	
	INT N = (INT) mxGetM(prhs[2]);
	
	if ((INT) mxGetNumberOfElements(prhs[0]) != M * N) {
		ERROR("Number of elements of Phi does not match product of reduced dimension number (second argument) times dimension number (first dimension of XLXt).");
	} else if ((INT) mxGetN(prhs[2]) != N) {
		ERROR("Second dimension of XLXt matrix does not match dimension number (first dimension of XLXt).");
	} else if ((INT) mxGetM(prhs[4]) != N) {
		ERROR("First dimension of DDt2 matrix does not match dimension number (first dimension of XLXt).");
	} else if ((INT) mxGetN(prhs[4]) != N) {
		ERROR("Second dimension of DDt2 matrix does not match dimension number (first dimension of XLXt).");
	} else if ((INT) mxGetM(prhs[5]) != N) {
		ERROR("First dimension of DDt3 matrix does not match dimension number (first dimension of XLXt).");
	} else if ((INT) mxGetN(prhs[5]) != N) {
		ERROR("Second dimension of DDt3 matrix does not match dimension number (first dimension of XLXt).");
	} else if ((INT) mxGetM(prhs[6]) != N) {
		ERROR("First dimension of VL matrix does not match dimension number (first dimension of XLXt).");
	} else if ((INT) mxGetN(prhs[6]) != N) {
		ERROR("Second dimension of VL matrix does not match dimension number (first dimension of XLXt).");
	} else if ((INT) mxGetNumberOfElements(prhs[7]) != N) {
		ERROR("Number of elements of L matrix does not match dimension number (first dimension of XLXt).");
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
	
	DOUBLE *PhiXLXt = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *PhiXLXtPhit = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
	DOUBLE *Gamma;
	DOUBLE *ObjMat;
	DOUBLE *PhiDDt2;
	DOUBLE *DDt3temp;
	
	if (betaReg > 0) {
		Gamma = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
		ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		if (derivFlag == 1) {
			PhiDDt2 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
			DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		} else {
			PhiDDt2 = NULL;
			DDt3temp = NULL;
		}
	} else {
		Gamma = NULL;
		ObjMat = NULL;
		PhiDDt2 = NULL;
		DDt3temp = NULL;
	}
	
	eig_lap_obj_grad(obj, deriv, Phi, XLXt, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, numSamples, derivFlag,\
					PhiXLXt, PhiXLXtPhit, Gamma, ObjMat, PhiDDt2, DDt3temp);
	
	FREE(PhiXLXt);
	FREE(PhiXLXtPhit);
	if (betaReg > 0) {
		FREE(Gamma);
		FREE(ObjMat);
		if (derivFlag == 1) {
			FREE(PhiDDt2);
			FREE(DDt3temp);
		}
	}
}
