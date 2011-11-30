/*
 * l1_fista.c
 *
 *  Created on: Apr 19, 2011
 *      Author: igkiou
 *  Entirely rewritten and renamed on: Aug 24, 2011
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "utils.h"
#ifdef USE_CUDA
#include "usecuda.h"
#include "cudakernels.h"
#endif
#include "l1_proximal.h"
#include "useinterfaces.h"
#include "useblas.h"

// TODO: Write version with lineSearch.
// TODO: Write to add initialization of A argument.
// TODO: Change to add weighted l1-norm.
void l1qp_ista(DOUBLE *A, DOUBLE *X, DOUBLE *D, DOUBLE *lambda, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, INT N, INT K, \
			INT numSamples, INT numRepeats) {

	DOUBLE *AOld;
	DOUBLE *KDX =  (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
	DOUBLE *KDD =  (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
	DOUBLE *KDDA;

	CHAR uplo = 'U';
	CHAR trans = 'T';
	INT SYRKN = K;
	INT SYRKK = N;
	DOUBLE alpha = 1.0;
	INT SYRKLDA = N;
	DOUBLE beta = 0;
	INT SYRKLDC = K;
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, D, &SYRKLDA, &beta, KDD, &SYRKLDC);

	CHAR transa = 'T';
	CHAR transb = 'N';
	INT GEMMM = K;
	INT GEMMN = numSamples;
	INT GEMMK = N;
	alpha = 1.0;
	INT GEMMLDA = N;
	INT GEMMLDB = N;
	beta = 0;
	INT GEMMLDC = K;
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, D, &GEMMLDA, X, &GEMMLDB, &beta, KDX, &GEMMLDC);

	memset((void *) A, 0, K * numSamples * sizeof(DOUBLE));
	INT iterSamples;
	INT iterRepeat;

	#pragma omp parallel private(iterSamples, iterRepeat, AOld, KDDA) \
				shared(X, A, KDX, KDD, lambda) \
	firstprivate(delta, numIters, tolerance, K, numSamples, numRepeats)
	{
		AOld = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		KDDA = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));

		#pragma omp for
		for (iterSamples = 0; iterSamples < numSamples; ++iterSamples) {
			for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
	//			printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, numRepeats, lambda[iterRepeat]);
				l1_ista_inner(&A[iterSamples * K], &KDX[iterSamples * K], KDD, \
						lambda[iterRepeat], delta, numIters, tolerance, K, \
						AOld, KDDA);
			}
		}
		CFREE(AOld);
		CFREE(KDDA);
	}
	FREE(KDX);
	FREE(KDD);
}

void l1kernel_ista(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE *lambda, \
			DOUBLE delta, INT numIters, DOUBLE tolerance, INT K, \
			INT numSamples, INT numRepeats) {

	DOUBLE *AOld;
	DOUBLE *KDDA;

	memset((void *) A, 0, K * numSamples * sizeof(DOUBLE));
	INT iterSamples;
	INT iterRepeat;

	#pragma omp parallel private(iterSamples, iterRepeat, AOld, KDDA) \
					shared(A, KDX, KDD, lambda) \
					firstprivate(delta, numIters, tolerance, K, numSamples, \
								numRepeats)
	{
		AOld = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		KDDA = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));

		#pragma omp for
		for (iterSamples = 0; iterSamples < numSamples; ++iterSamples) {
			for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
	//			printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, numRepeats, lambda[iterRepeat]);
				l1_ista_inner(&A[iterSamples * K], &KDX[iterSamples * K], KDD, \
						lambda[iterRepeat], delta, numIters, tolerance, K, \
						AOld, KDDA);
			}
		}
		CFREE(AOld);
		CFREE(KDDA);
	}
}

void l1_ista_inner(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE tau, \
			DOUBLE delta, INT numIters, DOUBLE tolerance, INT K, DOUBLE *AOld, \
			DOUBLE *KDDA) {

	INT AOldFlag = 0;
	if (AOld == NULL) {
		AOld = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		AOldFlag = 1;
	}

	INT KDDAFlag = 0;
	if (KDDA == NULL) {
		KDDA = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		KDDAFlag = 1;
	}

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	DOUBLE normAOld;
	DOUBLE normADiff;
	DOUBLE normA;

	INT NRM2N = K;
	INT AXPBYN = K;
	INT AXPYN = K;
	DOUBLE alpha;
	DOUBLE beta;
	INT incx = 1;
	INT incy = 1;

	INT iter = 0;
	while (1) {
		++iter;
//		if (iter % 10000 == 0) {
//			printf("Now running iter %d\n", iter);
//		}

		datacpy(AOld, A, K);
		normAOld = NRM2(&NRM2N, AOld, &incx);

		kernel_alt_obj_grad(NULL, A, AOld, KDX, KDD, K, KDDA);

		alpha = 1.0;
		beta = - delta;
		AXPBY(&AXPBYN, &alpha, AOld, &incx, &beta, A, &incy);

		l1_proximal(A, NULL, tau * delta, K);

		if (iter == numIters) {
			break;
		}

		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, A, &incx, AOld, &incy);
		normADiff = NRM2(&NRM2N, AOld, &incx);
		normA = NRM2(&NRM2N, A, &incx);

		if ((normA > eps) && (normADiff / IMAX(1.0, normAOld) < tolerance)) {
			break;
		}
	}

	if (AOldFlag == 1) {
		FREE(AOld);
	}

	if (KDDAFlag == 1) {
		FREE(KDDA);
	}
}

void l1qp_fista(DOUBLE *A, DOUBLE *X, DOUBLE *D, DOUBLE *lambda, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, \
			INT N, INT K, INT numSamples, INT numRepeats) {

	DOUBLE *AOld;
	DOUBLE *KDDA;
	DOUBLE *L;
	DOUBLE *LfGrad;
	DOUBLE *LfGD;
	DOUBLE *ALfGDDiff;
	DOUBLE *KDX = (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
	DOUBLE *KDD = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));

	CHAR uplo = 'U';
	CHAR trans = 'T';
	INT SYRKN = K;
	INT SYRKK = N;
	DOUBLE alpha = 1.0;
	INT SYRKLDA = N;
	DOUBLE beta = 0;
	INT SYRKLDC = K;
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, D, &SYRKLDA, &beta, KDD, &SYRKLDC);

	CHAR transa = 'T';
	CHAR transb = 'N';
	INT GEMMM = K;
	INT GEMMN = numSamples;
	INT GEMMK = N;
	alpha = 1.0;
	INT GEMMLDA = N;
	INT GEMMLDB = N;
	beta = 0;
	INT GEMMLDC = K;
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, D, &GEMMLDA, X, &GEMMLDB, &beta, KDX, &GEMMLDC);

	memset((void *) A, 0, K * numSamples * sizeof(DOUBLE));
	DOUBLE delta;
	INT iterSamples;
	INT iterRepeat;

//	#pragma omp parallel private(iterSamples, iterRepeat, delta, AOld, KDDA, L, \
//								LfGrad, LfGD, ALfGDDiff) \
//				shared(X, A, KDX, KDD, lambda) \
//				firstprivate(delta0, numIters, tolerance, K, numSamples, \
//								numRepeats, NRM2N, incx, lineSearchFlag, eta)
//	{
		AOld = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		KDDA = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		L = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		LfGrad = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		LfGD = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		ALfGDDiff = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));

//		#pragma omp for
		for (iterSamples = 0; iterSamples < numSamples; ++iterSamples) {
			for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
	//			printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, numRepeats, lambda[iterRepeat]);
				delta = delta0;
				l1_fista_inner(&A[iterSamples * K], &KDX[iterSamples * K], \
						KDD, lambda[iterRepeat], delta, numIters, tolerance, \
						lineSearchFlag, eta, K, AOld, KDDA, L, LfGrad, LfGD, \
						ALfGDDiff);
			}
		}
		CFREE(AOld);
		CFREE(KDDA);
		CFREE(L);
		CFREE(LfGrad);
		CFREE(LfGD);
		CFREE(ALfGDDiff);
//	}
	FREE(KDX);
	FREE(KDD);

}

void l1kernel_fista(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE *lambda, \
			DOUBLE delta0, INT numIters, DOUBLE tolerance, INT lineSearchFlag, \
			DOUBLE eta, INT K, INT numSamples, INT numRepeats) {

	DOUBLE *AOld;
	DOUBLE *KDDA;
	DOUBLE *L;
	DOUBLE *LfGrad;
	DOUBLE *LfGD;
	DOUBLE *ALfGDDiff;

	memset((void *) A, 0, K * numSamples * sizeof(DOUBLE));
	DOUBLE delta;
	INT iterSamples;
	INT iterRepeat;

	#pragma omp parallel private(iterSamples, iterRepeat, delta, AOld, KDDA, \
								L, LfGrad, LfGD, ALfGDDiff) \
				shared(A, KDX, KDD, lambda) \
				firstprivate(delta0, numIters, tolerance, K, numSamples, \
								numRepeats, lineSearchFlag, eta)
	{
		AOld = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		KDDA = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		L = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		LfGrad = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		LfGD = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		ALfGDDiff = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));

		for (iterSamples = 0; iterSamples < numSamples; ++iterSamples) {
			for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
		//			printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, numRepeats, lambda[iterRepeat]);
				delta = delta0;
				l1_fista_inner(&A[iterSamples * K], &KDX[iterSamples * K], \
						KDD, lambda[iterRepeat], delta, numIters, tolerance, \
						lineSearchFlag, eta, K, AOld, KDDA, L, LfGrad, LfGD, \
						ALfGDDiff);
			}
		}
		CFREE(AOld);
		CFREE(KDDA);
		CFREE(L);
		CFREE(LfGrad);
		CFREE(LfGD);
		CFREE(ALfGDDiff);
	}
}

// TODO: Urgent, differs from MATLAB version, probably problem in convergence criterion.
void l1_fista_inner(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE tau, \
			DOUBLE delta0, INT numIters, DOUBLE tolerance, INT lineSearchFlag, \
			DOUBLE eta, INT K, DOUBLE *AOld, DOUBLE *KDDA, DOUBLE *L, \
			DOUBLE *LfGrad, DOUBLE *LfGD, DOUBLE *ALfGDDiff) {

	INT AOldFlag = 0;
	if (AOld == NULL) {
		AOld = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		AOldFlag = 1;
	}

	INT KDDAFlag = 0;
	if (KDDA == NULL) {
		KDDA = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		KDDAFlag = 1;
	}

	INT LFlag = 0;
	if (L == NULL) {
		L = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		LFlag = 1;
	}

	INT LfGradFlag = 0;
	if (LfGrad == NULL) {
		LfGrad = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		LfGradFlag = 1;
	}

	INT LfGDFlag = 0;
	if (LfGD == NULL) {
		LfGD = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		LfGDFlag = 1;
	}

	INT ALfGDDiffFlag = 0;
	if (ALfGDDiff == NULL) {
		ALfGDDiff = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		ALfGDDiffFlag = 1;
	}

	DOUBLE alpha;
	DOUBLE beta;
	INT incx = 1;
	INT incy = 1;
	INT NRM2N = K;
	INT AXPYN = K;
	INT AXPBYN = K;

	DOUBLE t = 1;
	DOUBLE tOld = 1;
	DOUBLE LfObj;
	DOUBLE delta = delta0;
	DOUBLE normADiff;
	DOUBLE normAOld;
	DOUBLE normA;

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	INT iter = 0;
	while (1) {
		++iter;
//		if (iter % 10000 == 0) {
//			printf("Now running iter %d\n", iter);
//		}

		datacpy(L, A, K);
		alpha = - (tOld - 1.0) / t;
		beta = 1 - alpha;
 		AXPBY(&AXPBYN, &alpha, AOld, &incx, &beta, L, &incy);
 		datacpy(AOld, A, K);
 		kernel_alt_obj_grad(&LfObj, LfGrad, L, KDX, KDD, K, KDDA);

 		l1_line_search(&delta, A, L, LfObj, LfGrad, tau, lineSearchFlag, \
 				eta, KDX, KDD, K, LfGD, ALfGDDiff, KDDA);

		tOld = t;
		t = (1 + SQRT(1 + 4 * SQR(tOld))) * 0.5;

		if (iter == numIters) {
			break;
		}

		datacpy(ALfGDDiff, A, K);
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, AOld, &incx, ALfGDDiff, &incy);
		normADiff = NRM2(&NRM2N, ALfGDDiff, &incx);
		normAOld = NRM2(&NRM2N, AOld, &incx);
		normA = NRM2(&NRM2N, A, &incx);

		if ((normA > eps) && (normADiff / IMAX(1.0, normAOld) < tolerance)) {
			break;
		}
	}

	if (AOldFlag == 1) {
		FREE(AOld);
	}

	if (KDDAFlag == 1) {
		FREE(KDDA);
	}

	if (LFlag == 1) {
		FREE(L);
	}

	if (LfGradFlag == 1) {
		FREE(LfGrad);
	}

	if (LfGDFlag == 1) {
		FREE(LfGD);
	}

	if (ALfGDDiffFlag == 1) {
		FREE(ALfGDDiff);
	}
}

void l1_line_search(DOUBLE *delta, DOUBLE *LfGDShrink, DOUBLE *L, \
		DOUBLE LfObj, DOUBLE *LfGrad, DOUBLE tau, INT lineSearchFlag, \
		DOUBLE eta, DOUBLE *KDX, DOUBLE *KDD, INT K, DOUBLE *LfGD, \
		DOUBLE *ALfGDDiff, DOUBLE *KDDA) {

	INT ALfGDDiffFlag = 0;
	if (ALfGDDiff == NULL) {
		ALfGDDiff = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		ALfGDDiffFlag = 1;
	}

	INT KDDAFlag = 0;
	if (KDDA == NULL) {
		KDDA = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		KDDAFlag = 1;
	}

	INT AXPYN = K;
	DOUBLE alpha;
	INT incx = 1;
	INT incy = 1;

	DOUBLE LfGDShrinkfObj;
	DOUBLE LQmodObj;
	DOUBLE deltatemp = *delta;
	while (1) {
		datacpy(LfGD, L, K);
		alpha = - 1.0 / deltatemp;
		AXPY(&AXPYN, &alpha, LfGrad, &incx, LfGD, &incy);
		l1_shrinkage(LfGDShrink, NULL, LfGD, tau / deltatemp, K);
		if (lineSearchFlag == 0) {
			break;
		}
		kernel_alt_obj_grad(&LfGDShrinkfObj, NULL, LfGDShrink, KDX, KDD, K, \
						KDDA);
		l1_Q_func_mod(&LQmodObj, LfGDShrink, LfGD, LfObj, LfGrad, \
						deltatemp, K, ALfGDDiff);
		if (LfGDShrinkfObj <= LQmodObj) {
			break;
		}
		deltatemp = deltatemp * eta;
	}

	*delta = deltatemp;
	if (ALfGDDiffFlag == 1) {
		FREE(ALfGDDiff);
	}

	if (KDDAFlag == 1) {
		FREE(KDDA);
	}
}

void l1_shrinkage(DOUBLE *AShrink, DOUBLE *AShrinkNorm, DOUBLE *A, \
				DOUBLE tau, INT K) {

	datacpy(AShrink, A, K);
	l1_proximal(AShrink, AShrinkNorm, tau, K);

}

void l1_proximal(DOUBLE *X, DOUBLE *norm, DOUBLE tau, INT N) {

	INT iterN;
	DOUBLE normTemp = 0;
	DOUBLE threshTemp;
	for (iterN = 0; iterN < N; ++iterN) {
		if ((threshTemp = ABS(X[iterN]) - tau) > 0) {
			X[iterN] = SIGN(X[iterN]) * threshTemp;
			normTemp += threshTemp;
		} else {
			X[iterN] = 0;
		}
	}
	if (norm != NULL) {
		*norm = normTemp;
	}
}

#ifdef USE_CUDA
void l1_proximal_cuda(CUHANDLE handle, CUDOUBLE *X, CUDOUBLE *h_norm, \
						CUDOUBLE tau, CUINT N) {

	cuSoftThreshold(X, tau, N);
	if (h_norm != NULL) {
		CUINT ASUMN = N;
		CUINT incx = 1;
		CUASUM(handle, ASUMN, X, incx, h_norm);
	}
}
#endif

void l1_Q_func_mod(DOUBLE *QObj, DOUBLE *A, DOUBLE *LfGD, DOUBLE LfObj, \
			DOUBLE *LfGrad, DOUBLE delta, INT K, DOUBLE *ALfGDDiff) {

	INT ALfGDDiffFlag = 0;
	if (ALfGDDiff == NULL) {
		ALfGDDiff = (DOUBLE *) MALLOC(K * sizeof(DOUBLE));
		ALfGDDiffFlag = 1;
	}

	datacpy(ALfGDDiff, A, K);
	INT AXPYN = K;
	DOUBLE alpha = - 1.0;
	INT incx = 1;
	INT incy = 1;
	AXPY(&AXPYN, &alpha, LfGD, &incx, ALfGDDiff, &incy);

	INT NRM2N = K;
	DOUBLE ALfGDDiffNorm = NRM2(&NRM2N, ALfGDDiff, &incx);
	DOUBLE LfGradNorm = NRM2(&NRM2N, LfGrad, &incx);

	*QObj = delta * 0.5 * SQR(ALfGDDiffNorm) - 0.5 / delta * SQR(LfGradNorm) \
			+ LfObj;

	if (ALfGDDiffFlag == 1) {
		FREE(ALfGDDiff);
	}
}

void qp_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *A, DOUBLE *X, DOUBLE *D, \
		INT N, INT K, DOUBLE *res) {

	INT resFlag = 0;
	if (res == NULL) {
		res = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		resFlag = 1;
	}

	datacpy(res, X, N);

	CHAR trans = 'N';
	INT GEMVM = N;
	INT GEMVN = K;
	DOUBLE alpha = - 1;
	INT GEMVLDA = N;
	DOUBLE beta = 1;
	INT incx = 1;
	INT incy = 1;
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, D, &GEMVLDA, A, &incx, &beta, res, &incy);

	if (obj != NULL) {
		INT NRM2N = N;
		*obj = NRM2(&NRM2N, res, &incx);
		*obj = SQR(*obj) * 0.5;
	}

	if (deriv != NULL) {
		trans = 'T';
		GEMVM = N;
		GEMVN = K;
		alpha = - 1;
		GEMVLDA = N;
		beta = 0;
		incx = 1;
		incy = 1;
		GEMV(&trans, &GEMVM, &GEMVN, &alpha, D, &GEMVLDA, res, &incx, &beta, deriv, &incy);
	}

	if (resFlag == 1) {
		FREE(res);
	}
}

void kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *A, DOUBLE *KXX, \
		DOUBLE *KDX, DOUBLE *KDD, INT K, DOUBLE *KDDA) {

	INT KDDAFlag = 0;
	if (KDDA == NULL) {
		KDDA = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		KDDAFlag = 1;
	}

	CHAR uplo = 'U';
	INT SYMVN = K;
	DOUBLE alpha = 1.0;
	INT SYMVLDA = K;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	SYMV(&uplo, &SYMVN, &alpha, KDD, &SYMVLDA, A, &incx, &beta, KDDA, &incy);

	if (obj != NULL) {
		INT DOTN = K;
		*obj = 0.5 * (*KXX - 2 * DOT(&DOTN, A, &incx, KDX, &incy) + DOT(&DOTN, A, &incx, KDDA, &incy));
	}

	if (deriv != NULL) {
		datacpy(deriv, KDDA, K);
		INT AXPYN = K;
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, KDX, &incx, deriv, &incy);
	}

	if (KDDAFlag == 1) {
		FREE(KDDA);
	}
}

void kernel_alt_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *A, DOUBLE *KDX, \
		DOUBLE *KDD, INT K, DOUBLE *KDDA) {

	INT KDDAFlag = 0;
	if (KDDA == NULL) {
		KDDA = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
		KDDAFlag = 1;
	}

	CHAR uplo = 'U';
	INT SYMVN = K;
	DOUBLE alpha = 1.0;
	INT SYMVLDA = K;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	SYMV(&uplo, &SYMVN, &alpha, KDD, &SYMVLDA, A, &incx, &beta, KDDA, &incy);

	if (obj != NULL) {
		INT DOTN = K;
		*obj = 0.5 * (- 2 * DOT(&DOTN, A, &incx, KDX, &incy) + DOT(&DOTN, A, &incx, KDDA, &incy));
	}

	if (deriv != NULL) {
		datacpy(deriv, KDDA, K);
		INT AXPYN = K;
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, KDX, &incx, deriv, &incy);
	}

	if (KDDAFlag == 1) {
		FREE(KDDA);
	}
}

void qp_lipschitz(DOUBLE *Lf, DOUBLE *D, INT N, INT K, DOUBLE *KDD, \
				INT precomputedKernelFlag) {

	INT KDDFlag = 0;
	if (precomputedKernelFlag == 0) {
		if (KDD == NULL) {
			KDD = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
			KDDFlag = 1;
		}
		CHAR uplo = 'U';
		CHAR trans = 'T';
		INT SYRKN = K;
		INT SYRKK = N;
		DOUBLE alpha = 1.0;
		INT SYRKLDA = N;
		DOUBLE beta = 0;
		INT SYRKLDC = K;
		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, D, &SYRKLDA, &beta, KDD, &SYRKLDC);
	}

	CHAR lamch_opt = 'S';
	DOUBLE sfmin = LAMCH(&lamch_opt);

	CHAR jobz = 'N';
	CHAR range = 'I';
	CHAR uplo = 'U';
	INT SYEVRN = K;
	INT SYEVLDA = K;
	INT IL = K;
	INT IU = K;
	DOUBLE abstol = sfmin;
	INT SYEVLDZ = 1;
	INT lwork = -1;
	INT liwork = -1;
	DOUBLE *work;
	INT *iwork;
	DOUBLE work_temp;
	INT iwork_temp;
	INT SYEVRM;
	INT INFO;
	INT SYEVRM_expected = IU - IL + 1;

	DOUBLE *lvec = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));

	SYEVR(&jobz, &range, &uplo, &SYEVRN, KDD, &SYEVLDA, NULL, NULL, &IL, &IU, &abstol, &SYEVRM, \
			lvec, NULL, &SYEVLDZ, NULL, &work_temp, &lwork, &iwork_temp, &liwork, &INFO);

	lwork = (INT) work_temp;
	work = (DOUBLE*) MALLOC(lwork * sizeof(DOUBLE));
	liwork = (INT) iwork_temp;
	iwork = (INT*) MALLOC(liwork * sizeof(INT));

	SYEVR(&jobz, &range, &uplo, &SYEVRN, KDD, &SYEVLDA, NULL, NULL, &IL, &IU, &abstol, &SYEVRM, \
			lvec, NULL, &SYEVLDZ, NULL, work, &lwork, iwork, &liwork, &INFO);
	if (SYEVRM != SYEVRM_expected) {
		PRINTF("Error, only %d eigenvalues were found, when %d were expected. ", SYEVRM, SYEVRM_expected);
		ERROR("LAPACK execution error.");
	}

	*Lf = 2 * lvec[0];
	FREE(lvec);
	FREE(work);
	FREE(iwork);
	if (KDDFlag == 1) {
		FREE(KDD);
	}
}
