/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useinterfaces.h"
#include "useblas.h"
#include "projection.h"
#include "exponential.h"
#include "l2exp_learn_basis.h"

void basis_exp_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *S, INT N, \
					INT K, INT numSamples, EXPONENTIAL_TYPE family, INT derivFlag, DOUBLE *DS, DOUBLE *aVal, \
					DOUBLE *aPrime) {
	
	INT DSFlag = 0;
	if (DS == NULL) {
		DS = (DOUBLE *) MALLOC(N * numSamples * sizeof(DOUBLE));
		DSFlag = 1;
	}
	
	INT aValFlag = 0;
	if (aVal == NULL) {
		aVal = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		aValFlag = 1;
	}
	
	INT aPrimeFlag = 0;
	if (derivFlag == 1) {
		if (aPrime == NULL) {
			aPrime = (DOUBLE *) MALLOC(N * numSamples * sizeof(DOUBLE));
			aPrimeFlag = 1;
		}
	}
	
	CHAR transa = 'N';
	CHAR transb = 'N';
	INT GEMMM = N;
	INT GEMMN = numSamples;
	INT GEMMK = K;
	DOUBLE alpha = 1;
	INT GEMMLDA = N;
	INT GEMMLDB = K;
	DOUBLE beta = 0;
	INT GEMMLDC = N;
	
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, D, &GEMMLDA, S, &GEMMLDB, &beta, DS, &GEMMLDC);

	INT incx = 1;
	INT incy = 1;
		
	if (derivFlag == 1) {
		link_func(aVal, aPrime, NULL, DS, N, numSamples, family, 1, 1, 0);
		
		INT AXPYN = N * numSamples;
		alpha = -1;

		AXPY(&AXPYN, &alpha, X, &incx, aPrime, &incy);
	} else if (derivFlag == 0) {
		link_func(aVal, aPrime, NULL, DS, N, numSamples, family, 1, 0, 0);
	}

	INT iterX;
	DOUBLE objtemp = 0;
	DOUBLE objp = 0;
	
	INT DOTN = N;
	
	INT GERM = N;
	INT GERN = K;
	INT GERLDA = N;
	alpha = 1;
	
	if (derivFlag == 1) {
		memset((void *) deriv, 0, N * K * sizeof(DOUBLE));
	}

	#pragma omp parallel for private(iterX, objtemp) shared(X, aVal, aPrime, DS, S, deriv) \
			reduction(+: objp) firstprivate(N, numSamples, incx, incy, DOTN, GERM, GERN, alpha, GERLDA)
	for (iterX = 0; iterX < numSamples; ++iterX) {
		
		objtemp = - DOT(&DOTN, &DS[iterX * N], &incx, &X[iterX * N], &incy);
		objtemp += aVal[iterX];
		
		if (derivFlag == 1) {
			#pragma omp critical
			GER(&GERM, &GERN, &alpha, &aPrime[iterX * N], &incx, &S[iterX * K], &incy, deriv, &GERLDA);
		}
		
		objp += objtemp;
	}

	*obj = objp;
	if (DSFlag == 1) {
		FREE(DS);
	}
	
	if (aValFlag == 1) {
		FREE(aVal);
	}
	
	if (derivFlag == 1) {
		if (aPrimeFlag == 1) {
			FREE(aPrime);
		}
	}
}

void l2exp_learn_basis_gradient_projection_backtracking(DOUBLE *D, DOUBLE *X, DOUBLE *S, DOUBLE *Dinit, \
							INT N, INT K, INT numSamples, EXPONENTIAL_TYPE family) {
	
	DOUBLE BETA = 0.9;
	DOUBLE ALPHA = 0.3;
	DOUBLE EPS = POW(10, -12);
	INT MAXITER = 200;
	INT MAXITEROUT = 100;
	DOUBLE tf;
	DOUBLE tAdd;
	DOUBLE p;
	
	DOUBLE preObjVal;
	DOUBLE postObjVal;
	INT iterBT;
	INT iterOut;
	
	INT NK = N * K;	
	DOUBLE *stp = (DOUBLE *) MALLOC(NK * sizeof(DOUBLE));
	DOUBLE *Dnorm = (DOUBLE *) MALLOC(NK * sizeof(DOUBLE));
	DOUBLE *DS = (DOUBLE *) MALLOC(N * numSamples * sizeof(DOUBLE));
	DOUBLE *aVal = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
	DOUBLE *aPrime = (DOUBLE *) MALLOC(N * numSamples * sizeof(DOUBLE));
	
	INT AXPYN = NK;
	INT NRM2N = NK;
	INT incx = 1;
	INT incy = 1;
	
	datacpy(D, Dinit, NK);
	iterOut = 0;
	while (1) {
		basis_exp_obj_grad(&preObjVal, stp, D, X, S, N, K, numSamples, family, 1, DS, aVal, aPrime);
		tf = 1.0;
		tAdd = - 1.0;
		p = NRM2(&NRM2N, stp, &incx);
		p = - SQR(p);
		for (iterBT = 0; iterBT < MAXITER; ++iterBT) {
			AXPY(&AXPYN, &tAdd, stp, &incx, D, &incy);
			l2_ball_projection_batch(Dnorm, NULL, 1, N, K);
			basis_exp_obj_grad(&postObjVal, NULL, Dnorm, X, S, N, K, numSamples, family, 0, DS, aVal, aPrime);
			if (postObjVal < preObjVal + ALPHA * tf * p) {
				break;
			} else {
				tAdd = (1 - BETA) * tf;
				tf = BETA * tf;	
			}
		}
		
		datacpy(D, Dnorm, NK);
		if (ABS(postObjVal - preObjVal) < EPS) {
			break;
		}
		if (++iterOut > MAXITEROUT) {
			break;
		}
	}
				
	FREE(Dnorm);
	FREE(DS);
	FREE(aVal);
	FREE(aPrime);
}

void l2exp_learn_basis_gradient_backtracking(DOUBLE *D, DOUBLE *X, DOUBLE *S, DOUBLE *Dinit, \
							INT N, INT K, INT numSamples, EXPONENTIAL_TYPE family) {
	
	DOUBLE BETA = 0.9;
	DOUBLE ALPHA = 0.3;
	DOUBLE EPS = POW(10, -12);
	INT MAXITER = 200;
	INT MAXITEROUT = 200;
	DOUBLE tf;
	DOUBLE tAdd;
	DOUBLE p;
	
	DOUBLE preObjVal;
	DOUBLE postObjVal;
	INT iterBT;
	INT iterOut;
	
	INT NK = N * K;	
	DOUBLE *stp = (DOUBLE *) MALLOC(NK * sizeof(DOUBLE));
/*
	DOUBLE *Dnorm = (DOUBLE *) MALLOC(NK * sizeof(DOUBLE));
*/
	DOUBLE *DS = (DOUBLE *) MALLOC(N * numSamples * sizeof(DOUBLE));
	DOUBLE *aVal = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
	DOUBLE *aPrime = (DOUBLE *) MALLOC(N * numSamples * sizeof(DOUBLE));
	
	INT AXPYN = NK;
	INT NRM2N = NK;
	INT incx = 1;
	INT incy = 1;
	
	datacpy(D, Dinit, NK);
	iterOut = 0;
	while (1) {
		basis_exp_obj_grad(&preObjVal, stp, D, X, S, N, K, numSamples, family, 1, DS, aVal, aPrime);
		tf = 1.0;
		tAdd = - 1.0;
		p = NRM2(&NRM2N, stp, &incx);
		p = - SQR(p);
		for (iterBT = 0; iterBT < MAXITER; ++iterBT) {
			AXPY(&AXPYN, &tAdd, stp, &incx, D, &incy);
			basis_exp_obj_grad(&postObjVal, NULL, D, X, S, N, K, numSamples, family, 0, DS, aVal, aPrime);
			if (postObjVal < preObjVal + ALPHA * tf * p) {
				break;
			} else {
				tAdd = (1 - BETA) * tf;
				tf = BETA * tf;	
			}
		}
		
/*
		datacpy(D, Dnorm, NK);
*/
		if (ABS(postObjVal - preObjVal) < EPS) {
			break;
		}
		if (++iterOut > MAXITEROUT) {
			break;
		}
	}
				
/*
	FREE(Dnorm);
*/
	FREE(DS);
	FREE(aVal);
	FREE(aPrime);
}
