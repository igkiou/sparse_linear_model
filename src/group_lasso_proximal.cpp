/*
 * group_lasso_proximal.c
 *
 *  Created on: Aug 24, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "utils.h"
#include "group_lasso_proximal.h"
#include "l1_proximal.h"
#include "useinterfaces.h"
#include "useblas.h"

void convertSetLabelMat(INT *locality, DOUBLE *orig, INT *setSizes, INT K, \
						INT numSets) {

	INT iter;
	INT sumSetSizes = 0;
	for (iter = 0; iter < numSets; ++iter) {
		sumSetSizes += setSizes[iter];
	}
	if (sumSetSizes != K) {
		ERROR("Invalid set labels: total length of sets is not equal to number of features.");
	}

	for (iter = 0; iter < K; ++iter) {
		locality[iter] = - 1;
	}
	for (iter = 0; iter < K; ++iter) {
		if (locality[((INT) orig[iter]) - 1] != - 1) {
			ERROR("Invalid set labels: same index is used more than once.");
		} else if (orig[iter] > K) {
			ERROR("Invalid set labels: index exceeds length of feature vector.");
		} else if (orig[iter] < 1) {
			ERROR("Invalid set labels: index must be a positive integer.");
		}
		locality[((INT) orig[iter]) - 1] = iter;
	}
	for (iter = 0; iter < K; ++iter) {
		if (locality[iter] == - 1) {
			ERROR("Invalid set labels: index is not used.");
		}
	}
}

void copyDictionaryToLocality(DOUBLE *XOut, DOUBLE *XIn, INT *locality, INT N, \
							INT K) {
	INT iterK;
	for (iterK = 0; iterK < K; ++iterK) {
		datacpy(&XOut[N * locality[iterK]], \
				&XIn[N * iterK], N);
	}
}

void copyCodeToOrig(DOUBLE *XOut, DOUBLE *XIn, INT *locality, INT K, \
					INT numSamples) {

	INT iterK;
	INT iterSample;
	for (iterSample = 0; iterSample < numSamples; ++iterSample) {
		for (iterK = 0; iterK < K; ++iterK) {
			XOut[iterSample * K + iterK] = \
										XIn[iterSample * K + locality[iterK]];
		}
	}
}

void copyCodeToLocality(DOUBLE *XOut, DOUBLE *XIn, INT *locality, INT K, \
						INT numSamples) {

	INT iterK;
	INT iterSample;
	for (iterSample = 0; iterSample < numSamples; ++iterSample) {
		for (iterK = 0; iterK < K; ++iterK) {
			XOut[iterSample * K + locality[iterK]] = \
										XIn[iterSample * K + iterK];
		}
	}
}

void group_lasso_proximal(DOUBLE *X, DOUBLE *norm, DOUBLE tau, INT *setSizes,
						DOUBLE *setWeights, INT numSets) {

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	INT incx = 1;
	INT iterSet;
	INT currCount = 0;
	INT currSetSize;
	if (norm != NULL) {
		*norm = 0;
	}
	DOUBLE normSet;
	DOUBLE weightSet;
	DOUBLE tauSet;
	DOUBLE alpha;
	for (iterSet = 0; iterSet < numSets; ++iterSet) {
		currSetSize = setSizes[iterSet];
		normSet = NRM2(&currSetSize, &X[currCount], &incx);
		weightSet = setWeights[iterSet];
		tauSet = tau * weightSet;
		if ((normSet < eps) || (normSet - tauSet < 0)) {
			memset((void *) &X[currCount], 0, currSetSize * sizeof(DOUBLE));
		} else {
			alpha = (normSet - tauSet) / normSet;
			SCAL(&currSetSize, &alpha, &X[currCount], &incx);
			if (norm != NULL) {
				*norm += normSet - tauSet;
			}
		}
		currCount += currSetSize;
	}
}

// TODO: Write version with lineSearch.
// TODO: Write to add initialization of A argument.
void group_lasso_ista(DOUBLE *A, DOUBLE *X, DOUBLE *D, DOUBLE *lambda, \
			INT *setSizes, DOUBLE *setWeights, DOUBLE delta, INT numIters, \
			DOUBLE tolerance, INT N, INT K, INT numSamples, INT numSets, \
			INT numRepeats) {

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
	shared(X, A, KDX, KDD, lambda, setSizes, setWeights) \
	firstprivate(delta, numIters, tolerance, K, numSets, numSamples, numRepeats)
	{
		AOld = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		KDDA = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));

		#pragma omp for
		for (iterSamples = 0; iterSamples < numSamples; ++iterSamples) {
			for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
	//			printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, numRepeats, lambda[iterRepeat]);
				group_lasso_ista_inner(&A[iterSamples * K], \
						&KDX[iterSamples * K], KDD, lambda[iterRepeat], \
						setSizes, setWeights, delta, numIters, tolerance, K, \
						numSets, AOld, KDDA);
			}
		}

		CFREE(AOld);
		CFREE(KDDA);
	}
	FREE(KDX);
	FREE(KDD);
}

void group_lasso_ista_inner(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE tau, \
			INT *setSizes, DOUBLE *setWeights, DOUBLE delta, INT numIters, \
			DOUBLE tolerance, INT K, INT numSets, DOUBLE *AOld, DOUBLE *KDDA) {

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

		group_lasso_proximal(A, NULL, tau * delta, setSizes, setWeights, \
							numSets);

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

	if (KDDAFlag == 1) {		FREE(KDDA);
	}
}

void group_lasso_fista(DOUBLE *A, DOUBLE *X, DOUBLE *D, DOUBLE *lambda, \
			INT *setSizes, DOUBLE *setWeights, DOUBLE delta0, INT numIters, \
			DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, INT N, INT K, \
			INT numSamples, INT numSets, INT numRepeats) {

	DOUBLE *AOld;
	DOUBLE *KDDA;
	DOUBLE *L;
	DOUBLE *LfGrad;
	DOUBLE *LfGD;
	DOUBLE *ALfGDDiff;
	DOUBLE *KDX =  (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
	DOUBLE *KDD =  (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));

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

	#pragma omp parallel private(iterSamples, iterRepeat, delta, AOld, KDDA, \
							L, LfGrad, LfGD, ALfGDDiff) \
	shared(X, A, KDX, KDD, lambda, setSizes, setWeights) \
	firstprivate(delta0, numIters, tolerance, K, numSets, numSamples, \
							numRepeats, lineSearchFlag, eta)
	{
		AOld = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		KDDA = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		L = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		LfGrad = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		LfGD = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		ALfGDDiff = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));

		#pragma omp for
		for (iterSamples = 0; iterSamples < numSamples; ++iterSamples) {
			for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
	//			printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, numRepeats, lambda[iterRepeat]);
				delta = delta0;
				group_lasso_fista_inner(&A[iterSamples * K], \
						&KDX[iterSamples * K], KDD, lambda[iterRepeat], \
						setSizes, setWeights, delta, numIters, tolerance, \
						lineSearchFlag, eta, K, numSets, AOld, KDDA, L, LfGrad, \
						LfGD, ALfGDDiff);
			}
		}

		CFREE(AOld);
		CFREE(KDDA);
		CFREE(L);
		CFREE(LfGrad);
		CFREE(LfGD);
		CFREE(ALfGDDiff);
	}

	FREE(KDX);
	FREE(KDD);
}

void group_lasso_fista_inner(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE tau, \
			INT *setSizes, DOUBLE *setWeights, DOUBLE delta0, INT numIters, \
			DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, INT K, \
			INT numSets, DOUBLE *AOld, DOUBLE *KDDA, DOUBLE *L, DOUBLE *LfGrad, \
			DOUBLE *LfGD, DOUBLE *ALfGDDiff) {

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

 		group_lasso_line_search(&delta, A, L, LfObj, LfGrad, tau, setSizes, \
 				setWeights, lineSearchFlag, eta, KDX, KDD, K, numSets, LfGD, \
 				ALfGDDiff, KDDA);

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

void group_lasso_line_search(DOUBLE *delta, DOUBLE *LfGDShrink, DOUBLE *L, \
		DOUBLE LfObj, DOUBLE *LfGrad, DOUBLE tau, INT *setSizes, \
		DOUBLE *setWeights, INT lineSearchFlag, DOUBLE eta, DOUBLE *KDX, \
		DOUBLE *KDD, INT K, INT numSets, DOUBLE *LfGD, DOUBLE *ALfGDDiff, \
		DOUBLE *KDDA) {

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
		group_lasso_shrinkage(LfGDShrink, NULL, LfGD, tau / deltatemp, setSizes, \
								setWeights, K, numSets);
		if (lineSearchFlag == 0) {
			break;
		}
		kernel_alt_obj_grad(&LfGDShrinkfObj, NULL, LfGDShrink, KDX, KDD, K, \
						KDDA);
		group_lasso_Q_func_mod(&LQmodObj, LfGDShrink, LfGD, LfObj, LfGrad, \
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

void group_lasso_shrinkage(DOUBLE *AShrink, DOUBLE *AShrinkNorm, DOUBLE *A, \
				DOUBLE tau, INT *setSizes, DOUBLE *setWeights, INT K, \
				INT numSets) {

	datacpy(AShrink, A, K);
	group_lasso_proximal(AShrink, AShrinkNorm, tau, setSizes, setWeights, \
						numSets);

}

void group_lasso_Q_func_mod(DOUBLE *QObj, DOUBLE *A, DOUBLE *LfGD, DOUBLE LfObj, \
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
