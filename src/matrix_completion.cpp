/*
 * matrix_completion.cpp
 *
 *  Created on: Nov 20, 2011
 *      Author: igkiou
 */

/*
 * robust_pca.c
 *
 *  Created on: Oct 12, 2011
 *      Author: igkiou
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "utils.h"
#include "matrix_completion.h"
#include "l1_proximal.h"
#include "useinterfaces.h"
#include "useblas.h"
#include "matrix_proximal.h"

void convertObservationMat(INT *observedInds, DOUBLE *observedVals, \
						DOUBLE *matlabObservationMat, INT numObservations, \
						INT M, INT N) {
	INT iterObserved;
	INT currentI;
	INT currentJ;
	DOUBLE currentVal;
	for (iterObserved = 0; iterObserved < numObservations; ++iterObserved) {
		currentI = (INT) matlabObservationMat[iterObserved];
		currentJ = (INT) matlabObservationMat[numObservations + iterObserved];
		currentVal = matlabObservationMat[2 * numObservations + iterObserved];
		if ((currentI > M) || (currentI < 1) \
			|| (currentJ > N) || (currentJ < 1)) {
			PRINTF("iter %d, i %d, j %d val %lf.\n", iterObserved, currentI, currentJ, currentVal);
			ERROR("Invalid observation: index out of bounds");

		}
		observedInds[iterObserved] = currentI - 1;
		observedInds[numObservations + iterObserved] = currentJ - 1;
		observedVals[iterObserved] = currentVal;
	}
//	print_matrix_int(observedInds, numObservations, 2);
//	print_matrix(observedVals, numObservations, 1);
}


/*
 * Matrix completion with accelerated proximal gradient algorithm, using 2 * Lf
 * instead of Lf. Notation:
 * L = mu * ||B||_* + k / 2 * ||B||_Fro ^ 2 + 1 / 2 * ||B_Omega - b||_Fro ^ 2.
 */
/* TODO: Fix Lf in case kappa != 0. */
void matrix_completion_apg(DOUBLE *B, INT *observedInds, DOUBLE *observedVals,
						DOUBLE mu, DOUBLE kappa, INT numIters, DOUBLE tolerance, \
						DOUBLE delta, DOUBLE eta, INT initFlag, INT M, INT N, \
						INT numObserved) {

	DOUBLE *Bkm1 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *G = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *S = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));

	DOUBLE Lf = 1.0;
	DOUBLE invLf = 1.0;

	INT MINMN = IMIN(M, N);
	INT MAXMN = IMAX(M, N);
	DOUBLE *sv = (DOUBLE *) MALLOC(MINMN * 1 * sizeof(DOUBLE));
	DOUBLE *svecsmall = (DOUBLE *) MALLOC(MINMN * MINMN * sizeof(DOUBLE));
	DOUBLE *sveclarge = (DOUBLE *) MALLOC(MAXMN * MINMN * sizeof(DOUBLE));
	DOUBLE workTemp;
	INT lwork = -1;
	nuclear_proximal(B, NULL, mu, M, N, sv, svecsmall, sveclarge, &workTemp, \
					lwork);
	lwork = (INT) workTemp;
	DOUBLE *work = (DOUBLE *) MALLOC(lwork * 1 * sizeof(DOUBLE));

	INT AXPBYN = M * N;
	INT AXPYN = M * N;
	DOUBLE alpha;
	DOUBLE beta;
	INT incx = 1;
	INT incy = 1;
	DOUBLE normsum;
	CHAR NORM = 'F';
	INT LANGEM = M;
	INT LANGEN = N;
	INT LANGELDA = M;

	DOUBLE tk = 1.0;
	DOUBLE tkm1 = 1.0;
	DOUBLE muk = mu * delta;
	if (initFlag == 0) {
		memset((void *) B, 0, M * N * sizeof(DOUBLE));
	}
	memset((void *) Bkm1, 0, M * N * sizeof(DOUBLE));

	INT currentI;
	INT currentJ;
	INT iterObserved;
	INT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		datacpy(YB, B, M * N);
		AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);

		memset((void *) G, 0, M * N * sizeof(DOUBLE));
		for (iterObserved = 0; iterObserved < numObserved; ++iterObserved) {
			currentI = observedInds[iterObserved];
			currentJ = observedInds[numObserved + iterObserved];
			G[currentJ * M + currentI] = YB[currentJ * M + currentI] \
									- observedVals[iterObserved];
		}
//		print_matrix(G, M, N);

		datacpy(Bkm1, B, M * N);
		datacpy(B, YB, M * N);
		alpha = - invLf;
		beta = (1.0 - kappa * invLf);
		AXPBY(&AXPYN, &alpha, G, &incx, &beta, B, &incy);
		datacpy(S, B, M * N);
		nuclear_proximal(B, NULL, muk * invLf, M, N, sv, svecsmall, sveclarge, \
						work, lwork);
		alpha = - Lf;
		beta = Lf;
		AXPBY(&AXPBYN, &alpha, B, &incx, &beta, S, &incy);
		normsum = LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

		tkm1 = tk;
		tk = 0.5 * (1.0 + SQRT(4.0 * SQR(tkm1) + 1.0));
		muk = IMAX(eta * muk, mu);

		if (normsum < tolerance) {
			break;
		}

		++iter;
		if (iter % 100 == 0) {
			printf("%d\n", iter);
		}
		if (iter == numIters) {
			break;
		}
	}
	printf("%d\n", iter);
	FREE(Bkm1);
	FREE(YB);
	FREE(G);
	FREE(S);
	FREE(sv);
	FREE(sveclarge);
	FREE(svecsmall);
	FREE(work);
}

/*
 * Matrix completion with accelerated proximal gradient algorithm, using 2 * Lf
 * instead of Lf. Notation:
 * L = mu * ||B||_* + k / 2 * ||B||_Fro ^ 2 + 1 / 2 * ||B_Omega - b||_Fro ^ 2.
 */
/* TODO: Fix Lf in case kappa != 0. */
void operator_completion_apg(DOUBLE *B, INT *observedInds, DOUBLE *observedVals,
						DOUBLE *Y, DOUBLE mu, DOUBLE kappa, INT numIters, \
						DOUBLE tolerance, DOUBLE delta, DOUBLE eta, \
						INT initFlag, INT M, INT N, INT K, INT numObserved) {

	DOUBLE *Bkm1 = (DOUBLE *) MALLOC(M * K * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) MALLOC(M * K * sizeof(DOUBLE));
	DOUBLE *YBYt = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *PYBYt = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *S = (DOUBLE *) MALLOC(M * K * sizeof(DOUBLE));
	DOUBLE Lf;
	qp_lipschitz(&Lf, Y, N, K, NULL, 0);
	Lf = 0.5 * Lf;
	DOUBLE invLf = 1.0 / Lf;

	INT MINMN = IMIN(M, K);
	INT MAXMN = IMAX(M, K);
	DOUBLE *sv = (DOUBLE *) MALLOC(MINMN * 1 * sizeof(DOUBLE));
	DOUBLE *svecsmall = (DOUBLE *) MALLOC(MINMN * MINMN * sizeof(DOUBLE));
	DOUBLE *sveclarge = (DOUBLE *) MALLOC(MAXMN * MINMN * sizeof(DOUBLE));
	DOUBLE workTemp;
	INT lwork = -1;
	nuclear_proximal(B, NULL, mu, M, K, sv, svecsmall, sveclarge, &workTemp, \
					lwork);
	lwork = (INT) workTemp;
	DOUBLE *work = (DOUBLE *) MALLOC(lwork * 1 * sizeof(DOUBLE));

	INT AXPBYN = M * K;
	INT incx = 1;
	INT incy = 1;
	DOUBLE normsum;
	CHAR NORM = 'F';
	INT LANGEM = M;
	INT LANGEN = K;
	INT LANGELDA = M;
	CHAR transa;
	CHAR transb;
	INT GEMMM;
	INT GEMMN;
	INT GEMMK;
	DOUBLE alpha;
	INT GEMMLDA;
	DOUBLE beta;
	INT GEMMLDB;
	INT GEMMLDC;

	DOUBLE tk = 1.0;
	DOUBLE tkm1 = 1.0;
	DOUBLE muk = mu * delta;
	if (initFlag == 0) {
		memset((void *) B, 0, M * K * sizeof(DOUBLE));
	}
	memset((void *) Bkm1, 0, M * K * sizeof(DOUBLE));

	INT iterObserved;
	INT iter = 0;
	INT currentI;
	INT currentJ;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		datacpy(YB, B, M * K);
		AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);

		transa = 'N';
		transb = 'T';
		GEMMM = M;
		GEMMN = N;
		GEMMK = K;
		alpha = 1.0;
		GEMMLDA = M;
		beta = 0;
		GEMMLDB = N;
		GEMMLDC = M;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, YB, &GEMMLDA, Y, \
				&GEMMLDB, &beta, YBYt, &GEMMLDC);

//		memset((void *) Gt, 0, K * M * sizeof(DOUBLE));
//		for (iterObserved = 0; iterObserved < numObserved; ++iterObserved) {
//			currentI = observedInds[iterObserved];
//			currentJ = observedInds[numObserved + iterObserved];
//			AXPYN = K;
//			alpha = YBYt[currentJ * N + currentI] - observedVals[iterObserved];
//			AXPY(&AXPYN, &alpha, &Yt[K * currentJ], &incx, &Gt[K * currentI], \
//				&incy);
//		}
//		transpose(Gt, G, K, M);

		memset((void *) PYBYt, 0, M * N * sizeof(DOUBLE));
		for (iterObserved = 0; iterObserved < numObserved; ++iterObserved) {
			currentI = observedInds[iterObserved];
			currentJ = observedInds[numObserved + iterObserved];
			PYBYt[currentJ * M + currentI] = YBYt[currentJ * M + currentI] \
									- observedVals[iterObserved];
		}

		datacpy(Bkm1, B, M * K);
		datacpy(B, YB, M * K);
		transa = 'N';
		transb = 'N';
		GEMMM = M;
		GEMMN = K;
		GEMMK = N;
		alpha = - invLf;
		GEMMLDA = M;
		GEMMLDB = N;
		beta = 1.0 - invLf * kappa;
		GEMMLDC = M;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, PYBYt, \
				&GEMMLDA, Y, &GEMMLDB, &beta, B, &GEMMLDC);
		datacpy(S, B, M * K);
		nuclear_proximal(B, NULL, invLf * muk, M, K, sv, svecsmall, sveclarge, \
						work, lwork);
		alpha = - Lf;
		beta = Lf;
		AXPBY(&AXPBYN, &alpha, B, &incx, &beta, S, &incy);
		normsum = LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

		tkm1 = tk;
		tk = (1.0 + SQRT(4.0 * SQR(tkm1) + 1.0)) / 2.0;
		muk = IMAX(eta * muk, mu);

		if (normsum < tolerance) {
			break;
		}

		++iter;
		if (iter % 100 == 0) {
			printf("%d\n", iter);
		}
		if (iter == numIters) {
			break;
		}
	}
	printf("%d\n", iter);
	FREE(Bkm1);
	FREE(YB);
	FREE(YBYt);
	FREE(S);
	FREE(sv);
	FREE(sveclarge);
	FREE(svecsmall);
	FREE(work);
}
