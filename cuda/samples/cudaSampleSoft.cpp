/*
 * cudaSampleKernel.c
 *
 *  Created on: Nov 9, 2011
 *      Author: igkiou
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include <math.h>
#include <cuda_runtime.h>

//typedef double DOUBLE;
//#define mxPRECISION_CLASS mxDOUBLE_CLASS
//#define CUGEMM cublasDgemm
//#define MXISDOUBLE mxIsDouble
#include "cudaSampleKernel_sub.h"

void nuclear_proximal_cuda(cublasHandle_t handle, CUDOUBLE *X, CUDOUBLE *h_norm, \
							CUDOUBLE tau, CUINT M, CUINT N, CUDOUBLE *sv, \
							CUDOUBLE *svecsmall, CUDOUBLE *sveclarge) {

	CUINT MINMN = IMIN(M, N);
	CUINT MAXMN = IMAX(M, N);

	CUINT svFlag = 0;
	if (sv == NULL) {
		cumalloc((void**)&sv, MINMN * 1 * sizeof(CUDOUBLE));
		svFlag = 1;
	}

	CUINT svecsmallFlag = 0;
	if (svecsmall == NULL) {
		cumalloc((void**)&svecsmall, MINMN * MINMN * sizeof(CUDOUBLE));
		svecsmallFlag = 1;
	}

	CUINT sveclargeFlag = 0;
	if (sveclarge == NULL) {
		cumalloc((void**)&sveclarge, MAXMN * MINMN * sizeof(CUDOUBLE));
		sveclargeFlag = 1;
	}

	CUCHAR jobu = 'S';
	CUCHAR jobvt = 'S';
	CUDOUBLE *u;
	CUDOUBLE *vt;
	if (MAXMN == M) {
		u = sveclarge;
		vt = svecsmall;
	} else {
		u = svecsmall;
		vt = sveclarge;
	}
	CUINT GESVDM = M;
	CUINT GESVDN = N;
	CUINT GESVDLDA = M;
	CUINT GESVDLDU = M;
	CUINT GESVDLDVT = MINMN;

	culaStatus status = CUGESVD(jobu, jobvt, GESVDM, GESVDN, X, GESVDLDA, sv, u, \
								GESVDLDU, vt, GESVDLDVT);
	checkCulaStatus(status);

//	cuSoftThreshold(sv, tau, MINMN);
	if (h_norm != NULL) {
		CUINT ASUMN = MINMN;
		INT incx = 1;
		CUASUM(handle, MINMN, sv, incx, h_norm);
	}

	/*
	 * TODO: Only multiply for singular vectors corresponding to non-zero singular values.
	 */
	if (MAXMN == M) {
		CUINT SCALN = M;
		CUINT incx = 1;
		INT iterMN;
		cublasSetPointerModeDevice(handle);
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			CUSCAL(handle, SCALN, &sv[iterMN], &u[iterMN * M], incx);
		}
		cublasSetPointerModeHost(handle);

		CUCHAR transa = 'N';
		CUCHAR transb = 'N';
		CUINT GEMMM = M;
		CUINT GEMMN = N;
		CUINT GEMMK = MINMN;
		CUDOUBLE alpha = 1;
		CUINT GEMMLDA = M;
		CUINT GEMMLDB = MINMN;
		CUDOUBLE beta = 0;
		CUINT GEMMLDC = M;

		CUGEMM(handle, getCublasOperation(transa), getCublasOperation(transb), \
				GEMMM, GEMMN, GEMMK, &alpha, u, GEMMLDA, vt, GEMMLDB, &beta, X, \
				GEMMLDC);
	} else {
		CUINT SCALN = M;
		CUINT incx = 1;
		INT iterMN;
		cublasSetPointerModeDevice(handle);
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			CUSCAL(handle, SCALN, &sv[iterMN], &u[iterMN * M], incx);
		}
		cublasSetPointerModeHost(handle);

		CUCHAR transa = 'N';
		CUCHAR transb = 'N';
		CUINT GEMMM = M;
		CUINT GEMMN = N;
		CUINT GEMMK = MINMN;
		CUDOUBLE alpha = 1;
		CUINT GEMMLDA = M;
		CUINT GEMMLDB = MINMN;
		CUDOUBLE beta = 0;
		CUINT GEMMLDC = M;

		CUGEMM(handle, getCublasOperation(transa), getCublasOperation(transb), \
				GEMMM, GEMMN, GEMMK, &alpha, u, GEMMLDA, vt, GEMMLDB, &beta, X, \
				GEMMLDC);
	}

	if (svFlag == 1) {
		cufree(sv);
	}

	if (svecsmallFlag == 1) {
		cufree(svecsmall);
	}

	if (sveclargeFlag == 1) {
		cufree(sveclarge);
	}
}


/* Main */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 2) {
		ERROR("Two input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	CUDOUBLE *h_X = (CUDOUBLE*) mxGetData(prhs[0]);
	CUDOUBLE tau = *(CUDOUBLE*) mxGetData(prhs[1]);

	CUINT M = (CUINT) mxGetM(prhs[0]);
	CUINT N = (CUINT) mxGetN(prhs[0]);

	plhs[0] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
	CUDOUBLE *h_Xr = (CUDOUBLE *) mxGetData(plhs[0]);
	CUDOUBLE *h_norm;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
		h_norm = (CUDOUBLE *) mxGetData(plhs[1]);
	} else {
		h_norm = NULL;
	}

	culaInitialize();
	cublasHandle_t handle;
	cublasInitialize(&handle);
	CUDOUBLE *Xr = NULL;
	cumalloc((void**)&Xr, M * N * sizeof(CUDOUBLE));
	cuh2dcpy(Xr, h_X, M * N);
	nuclear_proximal_cuda(handle, Xr, h_norm, tau, M, N, NULL, NULL, NULL);
	cud2hcpy(h_Xr, Xr, M * N);
	cufree(Xr);
	culaShutdown();
	cublasDestroy(handle);
}
