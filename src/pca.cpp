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
#ifdef USE_CUDA
#include "usecuda.h"
#include "cudakernels.h"
#endif
#include "pca.h"
#include "useinterfaces.h"
#include "useblas.h"
#include "l1_proximal.h"
#include "matrix_proximal.h"

#ifdef USE_CUDA
void robust_pca_apg_cuda(CUHANDLE handle, CUDOUBLE *h_B, CUDOUBLE *h_A, \
				CUDOUBLE *h_D, CUDOUBLE mu, CUDOUBLE lambda, CUDOUBLE kappa, \
				CUINT numIters, CUDOUBLE tolerance, CUDOUBLE delta, \
				CUDOUBLE eta, CUINT initFlag, CUINT M, CUINT N) {

	CUDOUBLE *B = NULL;
	cumalloc(handle, (void**)&B, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *A = NULL;
	cumalloc(handle, (void**)&A, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *D = NULL;
	cumalloc(handle, (void**)&D, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *Bkm1 = NULL;
	cumalloc(handle, (void**)&Bkm1, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *Akm1 = NULL;
	cumalloc(handle, (void**)&Akm1, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *YB = NULL;
	cumalloc(handle, (void**)&YB, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *YA = NULL;
	cumalloc(handle, (void**)&YA, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *BAmD = NULL;
	cumalloc(handle, (void**)&BAmD, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *S = NULL;
	cumalloc(handle, (void**)&S, M * N * sizeof(CUDOUBLE));

	CUDOUBLE *Dt = (CUDOUBLE *) MALLOC(2 * N * 2 * N * sizeof(CUDOUBLE));
	memset((void *) Dt, 0, 2 * N * 2 * N * sizeof(CUDOUBLE));
	CUINT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		Dt[iterN * 2 * N + iterN] = 1;
		Dt[iterN * 2 * N + N + iterN] = 1;
		Dt[(N + iterN) * 2 * N + iterN] = SQRT(kappa);
	}
	CUDOUBLE Lf;
	qp_lipschitz(&Lf, Dt, 2 * N, 2 * N, NULL, 0);
	Lf = 0.5 * Lf;
	CUDOUBLE invLf = 1.0 / Lf;
	FREE(Dt);

	CUINT MINMN = IMIN(M, N);
	CUINT MAXMN = IMAX(M, N);
	CUDOUBLE *sv = NULL;
	cumalloc(handle, (void**)&sv, MINMN * 1 * sizeof(CUDOUBLE));
	CUDOUBLE *svecsmall = NULL;
	cumalloc(handle, (void**)&svecsmall, MINMN * MINMN * sizeof(CUDOUBLE));
	CUDOUBLE *sveclarge = NULL;
	cumalloc(handle, (void**)&sveclarge, MAXMN * MINMN * sizeof(CUDOUBLE));

	CUINT AXPBYN = M * N;
	CUINT AXPYN = M * N;
	CUINT NRM2N = M * N;
	CUDOUBLE alpha;
	CUDOUBLE beta;
	CUINT incx = 1;
	CUINT incy = 1;
	CUDOUBLE normtemp;
	CUDOUBLE normsum;

	CUDOUBLE tk = 1.0;
	CUDOUBLE tkm1 = 1.0;
	CUDOUBLE muk = mu * delta;

	if (initFlag == 0) {
		cumemset(handle, (void *) B, 0, M * N * sizeof(CUDOUBLE));
	} else {
		cuh2dcpy(handle, B, h_B, M * N);
	}
	cumemset(handle, (void *) Bkm1, 0, M * N * sizeof(CUDOUBLE));
	if (initFlag == 0) {
		cumemset(handle, (void *) A, 0, M * N * sizeof(CUDOUBLE));
	} else {
		cuh2dcpy(handle, A, h_A, M * N);
	}
	cumemset(handle, (void *) Akm1, 0, M * N * sizeof(CUDOUBLE));
	cuh2dcpy(handle, D, h_D, M * N);

	INT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		cud2dcpy(handle, YB, B, M * N);
		CUAXPBY(handle, AXPBYN, &alpha, Bkm1, incx, &beta, YB, incy);
		cud2dcpy(handle, YA, A, M * N);
		CUAXPBY(handle, AXPBYN, &alpha, Akm1, incx, &beta, YA, incy);
		cud2dcpy(handle, BAmD, D, M * N);
		alpha = 1.0;
		beta = - 1.0;
		CUAXPBY(handle, AXPBYN, &alpha, YB, incx, &beta, BAmD, incy);
		CUAXPY(handle, AXPYN, &alpha, YA, incx, BAmD, incy);

		cud2dcpy(handle, Akm1, A, M * N);
		cud2dcpy(handle, A, YA, M * N);
		alpha = - invLf;
		CUAXPY(handle, AXPYN, &alpha, BAmD, incx, A, incy);
		cud2dcpy(handle, S, A, M * N);
		l1_proximal_cuda(handle, A, NULL, invLf * muk * lambda, M * N);
		alpha = - Lf;
		beta = Lf;
		CUAXPBY(handle, AXPBYN, &alpha, A, incx, &beta, S, incy);
		CUNRM2(handle, NRM2N, S, incx, &normtemp);
		normsum = normtemp;

		cud2dcpy(handle, Bkm1, B, M * N);
		cud2dcpy(handle, B, YB, M * N);
		alpha = - invLf;
		beta = (1.0 - kappa * invLf);
		CUAXPBY(handle, AXPBYN, &alpha, BAmD, incx, &beta, B, incy);
		cud2dcpy(handle, S, B, M * N);
		nuclear_proximal_cuda(handle, B, NULL, muk * invLf, M, N, \
						sv, svecsmall, sveclarge);
		alpha = - Lf;
		beta = Lf;
		CUAXPBY(handle, AXPBYN, &alpha, B, incx, &beta, S, incy);
		CUNRM2(handle, NRM2N, S, incx, &normtemp);
		normsum += normtemp;

		tkm1 = tk;
		tk = 0.5 * (1.0 + SQRT(4.0 * SQR(tkm1) + 1.0));
		muk = IMAX(eta * muk, mu);

		if (normsum < tolerance) {
			break;
		}

		++iter;
		if (iter == numIters) {
			break;
		}
	}
	cud2hcpy(handle, h_B, B, M * N);
	cud2hcpy(handle, h_A, A, M * N);

	cufree(handle, B);
	cufree(handle, A);
	cufree(handle, D);
	cufree(handle, Bkm1);
	cufree(handle, Akm1);
	cufree(handle, YB);
	cufree(handle, YA);
	cufree(handle, S);
	cufree(handle, BAmD);
	cufree(handle, sv);
	cufree(handle, sveclarge);
	cufree(handle, svecsmall);
}
#endif

/*
 * Robust PCA with accelerated proximal gradient algorithm, using 2 * Lf
 * instead of Lf. Notation:
 * L = mu * ||B||_* + mu * lambda * ||A||_1 + k / 2 * ||B||_Fro ^ 2
 * 		+ 1 / 2 * ||D - B - A||_Fro ^ 2.
 */

void robust_pca_apg(DOUBLE *B, DOUBLE *A, DOUBLE *D, DOUBLE mu, DOUBLE lambda, \
				DOUBLE kappa, INT numIters, DOUBLE tolerance, DOUBLE delta, \
				DOUBLE eta, INT initFlag, INT M, INT N) {

	DOUBLE *Bkm1 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *Akm1 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *YA = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *BAmD = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *S = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));

	DOUBLE *Dt = (DOUBLE *) MALLOC(2 * N * 2 * N * sizeof(DOUBLE));
	memset((void *) Dt, 0, 2 * N * 2 * N * sizeof(DOUBLE));
	INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		Dt[iterN * 2 * N + iterN] = 1;
		Dt[iterN * 2 * N + N + iterN] = 1;
		Dt[(N + iterN) * 2 * N + iterN] = SQRT(kappa);
	}
	DOUBLE Lf;
	qp_lipschitz(&Lf, Dt, 2 * N, 2 * N, NULL, 0);
	Lf = 0.5 * Lf;
	DOUBLE invLf = 1.0 / Lf;

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
	if (initFlag == 0) {
		memset((void *) A, 0, M * N * sizeof(DOUBLE));
	}
	memset((void *) Akm1, 0, M * N * sizeof(DOUBLE));

	INT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		datacpy(YB, B, M * N);
		AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);
		datacpy(YA, A, M * N);
		AXPBY(&AXPBYN, &alpha, Akm1, &incx, &beta, YA, &incy);
		datacpy(BAmD, D, M * N);
		alpha = 1.0;
		beta = - 1.0;
		AXPBY(&AXPBYN, &alpha, YB, &incx, &beta, BAmD, &incy);
		AXPY(&AXPYN, &alpha, YA, &incx, BAmD, &incy);

		datacpy(Akm1, A, M * N);
		datacpy(A, YA, M * N);
		alpha = - invLf;
		AXPY(&AXPYN, &alpha, BAmD, &incx, A, &incy);
		datacpy(S, A, M * N);
		l1_proximal(A, NULL, invLf * muk * lambda, M * N);
		alpha = - Lf;
		beta = Lf;
		AXPBY(&AXPBYN, &alpha, A, &incx, &beta, S, &incy);
		normsum = LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

		datacpy(Bkm1, B, M * N);
		datacpy(B, YB, M * N);
		alpha = - invLf;
		beta = (1.0 - kappa * invLf);
		AXPBY(&AXPBYN, &alpha, BAmD, &incx, &beta, B, &incy);
		datacpy(S, B, M * N);
		nuclear_proximal(B, NULL, muk * invLf, M, N, sv, svecsmall, sveclarge, \
						work, lwork);
		alpha = - Lf;
		beta = Lf;
		AXPBY(&AXPBYN, &alpha, B, &incx, &beta, S, &incy);
		normsum += LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

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
	FREE(Akm1);
	FREE(YB);
	FREE(YA);
	FREE(S);
	FREE(Dt);
	FREE(BAmD);
	FREE(sv);
	FREE(sveclarge);
	FREE(svecsmall);
	FREE(work);
}

void robust_pca_apg_gesdd(DOUBLE *B, DOUBLE *A, DOUBLE *D, DOUBLE mu, DOUBLE lambda, \
				DOUBLE kappa, INT numIters, DOUBLE tolerance, DOUBLE delta, \
				DOUBLE eta, INT initFlag, INT M, INT N) {

	DOUBLE *Bkm1 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *Akm1 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *YA = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *BAmD = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *S = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));

	DOUBLE *Dt = (DOUBLE *) MALLOC(2 * N * 2 * N * sizeof(DOUBLE));
	memset((void *) Dt, 0, 2 * N * 2 * N * sizeof(DOUBLE));
	INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		Dt[iterN * 2 * N + iterN] = 1;
		Dt[iterN * 2 * N + N + iterN] = 1;
		Dt[(N + iterN) * 2 * N + iterN] = SQRT(kappa);
	}
	DOUBLE Lf;
	qp_lipschitz(&Lf, Dt, 2 * N, 2 * N, NULL, 0);
	Lf = 0.5 * Lf;
	DOUBLE invLf = 1.0 / Lf;

	INT MINMN = IMIN(M, N);
	INT MAXMN = IMAX(M, N);
	DOUBLE *sv = (DOUBLE *) MALLOC(MINMN * 1 * sizeof(DOUBLE));
	DOUBLE *svecsmall = (DOUBLE *) MALLOC(MINMN * MINMN * sizeof(DOUBLE));
	DOUBLE *sveclarge = (DOUBLE *) MALLOC(MAXMN * MINMN * sizeof(DOUBLE));
	INT *iwork = (INT *) MALLOC(16 * MINMN * sizeof(INT));
	DOUBLE workTemp;
	INT lwork = -1;
	nuclear_proximal_gesdd(B, NULL, mu, M, N, sv, svecsmall, sveclarge, &workTemp, \
			lwork, iwork);
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
	if (initFlag == 0) {
		memset((void *) A, 0, M * N * sizeof(DOUBLE));
	}
	memset((void *) Akm1, 0, M * N * sizeof(DOUBLE));

	INT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		datacpy(YB, B, M * N);
		AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);
		datacpy(YA, A, M * N);
		AXPBY(&AXPBYN, &alpha, Akm1, &incx, &beta, YA, &incy);
		datacpy(BAmD, D, M * N);
		alpha = 1.0;
		beta = - 1.0;
		AXPBY(&AXPBYN, &alpha, YB, &incx, &beta, BAmD, &incy);
		AXPY(&AXPYN, &alpha, YA, &incx, BAmD, &incy);

		datacpy(Akm1, A, M * N);
		datacpy(A, YA, M * N);
		alpha = - invLf;
		AXPY(&AXPYN, &alpha, BAmD, &incx, A, &incy);
		datacpy(S, A, M * N);
		l1_proximal(A, NULL, invLf * muk * lambda, M * N);
		alpha = - Lf;
		beta = Lf;
		AXPBY(&AXPBYN, &alpha, A, &incx, &beta, S, &incy);
		normsum = LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

		datacpy(Bkm1, B, M * N);
		datacpy(B, YB, M * N);
		alpha = - invLf;
		beta = (1.0 - kappa * invLf);
		AXPBY(&AXPYN, &alpha, BAmD, &incx, &beta, B, &incy);
		datacpy(S, B, M * N);
		nuclear_proximal_gesdd(B, NULL, muk * invLf, M, N, sv, svecsmall, sveclarge, \
						work, lwork, iwork);
		alpha = - Lf;
		beta = Lf;
		AXPBY(&AXPBYN, &alpha, B, &incx, &beta, S, &incy);
		normsum += LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

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
	FREE(Akm1);
	FREE(YB);
	FREE(YA);
	FREE(S);
	FREE(Dt);
	FREE(BAmD);
	FREE(sv);
	FREE(sveclarge);
	FREE(svecsmall);
	FREE(work);
	FREE(iwork);
}


/* TODO: Must debug CUDA versions. Also, weighted variants. */
#ifdef USE_CUDA
void robust_weighted_operator_pca_apg_cuda(CUHANDLE handle, CUDOUBLE *h_B, \
				CUDOUBLE *h_A, CUDOUBLE *h_D, CUDOUBLE *h_Y, CUDOUBLE *h_W, \
				CUDOUBLE mu, CUDOUBLE lambda, CUDOUBLE kappa, CUINT numIters, \
				CUDOUBLE tolerance, CUDOUBLE delta, CUDOUBLE eta, \
				CUINT initFlag, CUINT M, CUINT N, CUINT K) {

	CUDOUBLE *B = NULL;
	cumalloc(handle, (void**)&B, M * K * sizeof(CUDOUBLE));
	CUDOUBLE *A = NULL;
	cumalloc(handle, (void**)&A, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *D = NULL;
	cumalloc(handle, (void**)&D, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *Y = NULL;
	cumalloc(handle, (void**)&Y, N * K * sizeof(CUDOUBLE));
	CUDOUBLE *W = NULL;
	cumalloc(handle, (void**)&W, N * N * sizeof(CUDOUBLE));

	CUDOUBLE *Bkm1 = NULL;
	cumalloc(handle, (void**)&Bkm1, M * K * sizeof(CUDOUBLE));
	CUDOUBLE *Akm1 = NULL;
	cumalloc(handle, (void**)&Akm1, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *YB = NULL;
	cumalloc(handle, (void**)&YB, M * K * sizeof(CUDOUBLE));
	CUDOUBLE *YA = NULL;
	cumalloc(handle, (void**)&YA, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *BYtAmD = NULL;
	cumalloc(handle, (void**)&BYtAmD, M * N * sizeof(CUDOUBLE));
	CUDOUBLE *S = NULL;
	cumalloc(handle, (void**)&S, M * IMAX(N, K) * sizeof(CUDOUBLE));

	CUDOUBLE *h_YtW = (CUDOUBLE *) MALLOC(K * N * sizeof(CUDOUBLE));

	CHAR h_side = 'R';
	CHAR h_uplo = 'U';
	INT h_SYMMM = K;
	INT h_SYMMN = N;
	DOUBLE h_alpha = 1.0;
	INT h_SYMMLDA = N;
	INT h_SYMMLDB = N;
	DOUBLE h_beta = 0;
	INT h_SYMMLDC = K;
	SYMM(&h_side, &h_uplo, &h_SYMMM, &h_SYMMN, &h_alpha, h_W, &h_SYMMLDA, h_Y, &h_SYMMLDB, &h_beta, \
			h_YtW, &h_SYMMLDC);

	CUDOUBLE *h_Dt = (CUDOUBLE *) MALLOC(2 * N * 2 * N * sizeof(CUDOUBLE));
	memset((void *) h_Dt, 0, 2 * N * 2 * N * sizeof(CUDOUBLE));
	CUINT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		datacpy(&h_Dt[iterN * 2 * N], &h_YtW[iterN * N], N);
		datacpy(&h_Dt[iterN * 2 * N + N], &h_W[iterN * N], N);
		h_Dt[(N + iterN) * 2 * N + iterN] = SQRT(kappa);
	}
	CUDOUBLE Lf;
	qp_lipschitz(&Lf, h_Dt, 2 * N, 2 * N, NULL, 0);
	Lf = 0.5 * Lf;
	CUDOUBLE invLf = 1.0 / Lf;
	FREE(h_YtW);
	FREE(h_Dt);

	CUDOUBLE *Wsq = NULL;
	cumalloc(handle, (void**)&Wsq, N * N * sizeof(CUDOUBLE));
	CUDOUBLE *WsqY = NULL;
	cumalloc(handle, (void**)&WsqY, N * K * sizeof(CUDOUBLE));

	cuh2dcpy(handle, Y, h_Y, N * K);
	cuh2dcpy(handle, W, h_W, N * N);

	CUCHAR uplo = 'U';
	CUCHAR trans = 'N';
	CUINT SYRKN = N;
	CUINT SYRKK = N;
	CUDOUBLE alpha = 1.0;
	CUINT SYRKLDA = N;
	CUDOUBLE beta = 0;
	CUINT SYRKLDC = N;
	CUSYRK(handle, uplo, trans, SYRKN, SYRKK, &alpha, W, SYRKLDA, &beta, Wsq, \
		SYRKLDC);

	CUCHAR side = 'L';
	uplo = 'U';
	CUINT SYMMM = N;
	CUINT SYMMN = K;
	alpha = 1.0;
	CUINT SYMMLDA = N;
	CUINT SYMMLDB = N;
	beta = 0;
	CUINT SYMMLDC = N;
	CUSYMM(handle, side, uplo, SYMMM, SYMMN, &alpha, Wsq, SYMMLDA, Y, SYMMLDB, \
		&beta, WsqY, SYMMLDC);

	CUINT MINMN = IMIN(M, K);
	CUINT MAXMN = IMAX(M, K);
	CUDOUBLE *sv = NULL;
	cumalloc(handle, (void**)&sv, MINMN * 1 * sizeof(CUDOUBLE));
	CUDOUBLE *svecsmall = NULL;
	cumalloc(handle, (void**)&svecsmall, MINMN * MINMN * sizeof(CUDOUBLE));
	CUDOUBLE *sveclarge = NULL;
	cumalloc(handle, (void**)&sveclarge, MAXMN * MINMN * sizeof(CUDOUBLE));

	CUINT AXPBYN;
	CUINT incx = 1;
	CUINT incy = 1;
	CUDOUBLE normsum;
	CUDOUBLE normtemp;
	CUINT NRM2N;
	CUCHAR transa;
	CUCHAR transb;
	CUINT GEMMM;
	CUINT GEMMN;
	CUINT GEMMK;
	CUINT GEMMLDA;
	CUINT GEMMLDB;
	CUINT GEMMLDC;

	CUDOUBLE tk = 1.0;
	CUDOUBLE tkm1 = 1.0;
	CUDOUBLE muk = mu * delta;
	if (initFlag == 0) {
		cumemset(handle, (void *) B, 0, M * K * sizeof(CUDOUBLE));
	} else {
		cuh2dcpy(handle, h_B, B, M * K);
	}
	cumemset(handle, (void *) Bkm1, 0, M * K * sizeof(CUDOUBLE));
	if (initFlag == 0) {
		cumemset(handle, (void *) A, 0, M * N * sizeof(CUDOUBLE));
	} else {
		cuh2dcpy(handle, h_A, A, M * N);
	}
	cumemset(handle, (void *) Akm1, 0, M * N * sizeof(CUDOUBLE));
	cuh2dcpy(handle, D, h_D, M * N);

	CUINT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		cud2dcpy(handle, YB, B, M * K);
		AXPBYN = M * K;
		CUAXPBY(handle, AXPBYN, &alpha, Bkm1, incx, &beta, YB, incy);
		cud2dcpy(handle, YA, A, M * N);
		AXPBYN = M * N;
		CUAXPBY(handle, AXPBYN, &alpha, Akm1, incx, &beta, YA, incy);

		cud2dcpy(handle, BYtAmD, D, M * N);
		alpha = 1.0;
		beta = - 1.0;
		AXPBYN = M * N;
		CUAXPBY(handle, AXPBYN, &alpha, YA, incx, &beta, BYtAmD, incy);

		transa = 'N';
		transb = 'T';
		GEMMM = M;
		GEMMN = N;
		GEMMK = K;
		alpha = 1.0;
		GEMMLDA = M;
		GEMMLDB = N;
		beta = 1.0;
		GEMMLDC = M;
		CUGEMM(handle, transa, transb, GEMMM, GEMMN, GEMMK, &alpha, YB, GEMMLDA, \
			Y, GEMMLDB, &beta, BYtAmD, GEMMLDC);

		cud2dcpy(handle, Akm1, A, M * N);
		cud2dcpy(handle, A, YA, M * N);
		side = 'R';
		uplo = 'U';
		SYMMM = M;
		SYMMN = N;
		alpha = - invLf;
		SYMMLDA = N;
		SYMMLDB = M;
		beta = 1;
		SYMMLDC = M;
		CUSYMM(handle, side, uplo, SYMMM, SYMMN, &alpha, Wsq, SYMMLDA, BYtAmD, \
				SYMMLDB, &beta, A, SYMMLDC);
		cud2dcpy(handle, S, A, M * N);
		l1_proximal_cuda(handle, A, NULL, invLf * muk * lambda, M * N);
		alpha = - Lf;
		beta = Lf;
		AXPBYN = M * N;
		CUAXPBY(handle, AXPBYN, &alpha, A, incx, &beta, S, incy);
		NRM2N = M * N;
		normsum = 0;
		CUNRM2(handle, NRM2N, S, incx, &normtemp);
		normsum += normtemp;

		cud2dcpy(handle, Bkm1, B, M * K);
		cud2dcpy(handle, B, YB, M * K);
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
		CUGEMM(handle, transa, transb, GEMMM, GEMMN, GEMMK, &alpha, BYtAmD, \
				GEMMLDA, WsqY, GEMMLDB, &beta, B, GEMMLDC);
		cud2dcpy(handle, S, B, M * K);
		nuclear_proximal_cuda(handle, B, NULL, invLf * muk, M, K, sv, \
						svecsmall, sveclarge);
		alpha = - invLf;
		beta = Lf;
		AXPBYN = M * K;
		CUAXPBY(handle, AXPBYN, &alpha, B, incx, &beta, S, incy);
		NRM2N = M * K;
		CUNRM2(handle, NRM2N, S, incx, &normtemp);
		normsum += normtemp;

		tkm1 = tk;
		tk = (1.0 + SQRT(4.0 * SQR(tkm1) + 1.0)) / 2.0;
		muk = IMAX(eta * muk, mu);

		if (normsum < tolerance) {
			break;
		}

		++iter;
//		printf("%d\n", iter);
		if (iter == numIters) {
			break;
		}
	}

	cufree(handle, B);
	cufree(handle, A);
	cufree(handle, D);
	cufree(handle, Y);
	cufree(handle, W);
	cufree(handle, Bkm1);
	cufree(handle, Akm1);
	cufree(handle, YB);
	cufree(handle, YA);
	cufree(handle, S);
	cufree(handle, BYtAmD);
	cufree(handle, Wsq);
	cufree(handle, WsqY);
	cufree(handle, sv);
	cufree(handle, sveclarge);
	cufree(handle, svecsmall);
}
#endif


/*
 * Robust PCA with accelerated proximal gradient algorithm, using 2 * Lf
 * instead of Lf. Notation:
 * L = mu * ||B||_* + mu * lambda * ||A||_1 + k / 2 * ||B||_Fro ^ 2
 * 		+ 1 / 2 * ||(D - B * Y ^ T - A) * W||_Fro ^ 2.
 */

/*
 * TODO: Drop low-rank kernel matrix case.
 * TODO: Maybe use triangular Cholesky factors instead of Y = sqrt(K). Do not
 * expect to yield large speedups (this is a small multiplication, bottleneck
 * is in the SVD).
 */

void robust_weighted_operator_pca_apg(DOUBLE *B, DOUBLE *A, DOUBLE *D, \
				DOUBLE *Y, DOUBLE *W, DOUBLE mu, DOUBLE lambda, \
				DOUBLE kappa, INT numIters, DOUBLE tolerance, DOUBLE delta, \
				DOUBLE eta, INT initFlag, INT M, INT N, INT K) {

	DOUBLE *Bkm1 = (DOUBLE *) MALLOC(M * K * sizeof(DOUBLE));
	DOUBLE *Akm1 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) MALLOC(M * K * sizeof(DOUBLE));
	DOUBLE *YA = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *BYtAmD = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *S = (DOUBLE *) MALLOC(M * IMAX(N, K) * sizeof(DOUBLE));
	DOUBLE *YtW = (DOUBLE *) MALLOC(K * N * sizeof(DOUBLE));
	DOUBLE *Wsq = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *WsqY = (DOUBLE *) MALLOC(N * K * sizeof(DOUBLE));

	CHAR side = 'R';
	CHAR uplo = 'U';
	INT SYMMM = K;
	INT SYMMN = N;
	DOUBLE alpha = 1.0;
	INT SYMMLDA = N;
	INT SYMMLDB = N;
	DOUBLE beta = 0;
	INT SYMMLDC = K;
	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, W, &SYMMLDA, Y, &SYMMLDB, &beta, \
			YtW, &SYMMLDC);

	DOUBLE *Dt = (DOUBLE *) MALLOC(2 * N * 2 * N * sizeof(DOUBLE));
	memset((void *) Dt, 0, 2 * N * 2 * N * sizeof(DOUBLE));
	INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		datacpy(&Dt[iterN * 2 * N], &YtW[iterN * N], N);
		datacpy(&Dt[iterN * 2 * N + N], &W[iterN * N], N);
		Dt[(N + iterN) * 2 * N + iterN] = SQRT(kappa);
	}
	DOUBLE Lf;
	qp_lipschitz(&Lf, Dt, 2 * N, 2 * N, NULL, 0);
	Lf = 0.5 * Lf;
	DOUBLE invLf = 1.0 / Lf;

	uplo = 'U';
	CHAR trans = 'N';
	INT SYRKN = N;
	INT SYRKK = N;
	alpha = 1.0;
	INT SYRKLDA = N;
	beta = 0;
	INT SYRKLDC = N;
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, W, &SYRKLDA, &beta, Wsq, \
		&SYRKLDC);

	side = 'L';
	uplo = 'U';
	SYMMM = N;
	SYMMN = K;
	alpha = 1.0;
	SYMMLDA = N;
	SYMMLDB = N;
	beta = 0;
	SYMMLDC = N;
	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, Wsq, &SYMMLDA, Y, &SYMMLDB, \
		&beta, WsqY, &SYMMLDC);

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

	INT AXPBYN;
	INT incx = 1;
	INT incy = 1;
	DOUBLE normsum;
	CHAR NORM;
	INT LANGEM;
	INT LANGEN;
	INT LANGELDA;
	CHAR transa;
	CHAR transb;
	INT GEMMM;
	INT GEMMN;
	INT GEMMK;
	INT GEMMLDA;
	INT GEMMLDB;
	INT GEMMLDC;

	DOUBLE tk = 1.0;
	DOUBLE tkm1 = 1.0;
	DOUBLE muk = mu * delta;
	if (initFlag == 0) {
		memset((void *) B, 0, M * K * sizeof(DOUBLE));
	}
	memset((void *) Bkm1, 0, M * K * sizeof(DOUBLE));
	if (initFlag == 0) {
		memset((void *) A, 0, M * N * sizeof(DOUBLE));
	}
	memset((void *) Akm1, 0, M * N * sizeof(DOUBLE));

	INT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		datacpy(YB, B, M * K);
		AXPBYN = M * K;
		AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);
		datacpy(YA, A, M * N);
		AXPBYN = M * N;
		AXPBY(&AXPBYN, &alpha, Akm1, &incx, &beta, YA, &incy);

		datacpy(BYtAmD, D, M * N);
		alpha = 1.0;
		beta = - 1.0;
		AXPBYN = M * N;
		AXPBY(&AXPBYN, &alpha, YA, &incx, &beta, BYtAmD, &incy);

		transa = 'N';
		transb = 'T';
		GEMMM = M;
		GEMMN = N;
		GEMMK = K;
		alpha = 1.0;
		GEMMLDA = M;
		GEMMLDB = N;
		beta = 1.0;
		GEMMLDC = M;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, YB, &GEMMLDA, Y, \
				&GEMMLDB, &beta, BYtAmD, &GEMMLDC);

		datacpy(Akm1, A, M * N);
		datacpy(A, YA, M * N);
		side = 'R';
		uplo = 'U';
		SYMMM = M;
		SYMMN = N;
		alpha = - invLf;
		SYMMLDA = N;
		SYMMLDB = M;
		beta = 1;
		SYMMLDC = M;
		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, Wsq, &SYMMLDA, BYtAmD, \
				&SYMMLDB, &beta, A, &SYMMLDC);
		datacpy(S, A, M * N);
		l1_proximal(A, NULL, invLf * muk * lambda, M * N);
		alpha = - Lf;
		beta = Lf;
		AXPBYN = M * N;
		AXPBY(&AXPBYN, &alpha, A, &incx, &beta, S, &incy);
		NORM = 'F';
		LANGEM = M;
		LANGEN = N;
		LANGELDA = M;
		normsum = LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

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
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, BYtAmD, \
				&GEMMLDA, WsqY, &GEMMLDB, &beta, B, &GEMMLDC);
		datacpy(S, B, M * K);
		nuclear_proximal(B, NULL, invLf * muk, M, K, sv, \
						svecsmall, sveclarge, work, lwork);
		alpha = - Lf;
		beta = Lf;
		AXPBYN = M * K;
		AXPBY(&AXPBYN, &alpha, B, &incx, &beta, S, &incy);
		NORM = 'F';
		LANGEM = M;
		LANGEN = K;
		LANGELDA = M;
		normsum += LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

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
	FREE(Akm1);
	FREE(YB);
	FREE(YA);
	FREE(S);
	FREE(BYtAmD);
	FREE(Wsq);
	FREE(WsqY);
	FREE(YtW);
	FREE(Dt);
	FREE(sv);
	FREE(sveclarge);
	FREE(svecsmall);
	FREE(work);
}

void robust_weighted_operator_pca_apg_gesdd(DOUBLE *B, DOUBLE *A, DOUBLE *D, \
				DOUBLE *Y, DOUBLE *W, DOUBLE mu, DOUBLE lambda, \
				DOUBLE kappa, INT numIters, DOUBLE tolerance, DOUBLE delta, \
				DOUBLE eta, INT initFlag, INT M, INT N, INT K) {

	DOUBLE *Bkm1 = (DOUBLE *) MALLOC(M * K * sizeof(DOUBLE));
	DOUBLE *Akm1 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) MALLOC(M * K * sizeof(DOUBLE));
	DOUBLE *YA = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *BYtAmD = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *S = (DOUBLE *) MALLOC(M * IMAX(N, K) * sizeof(DOUBLE));
	DOUBLE *YtW = (DOUBLE *) MALLOC(K * N * sizeof(DOUBLE));
	DOUBLE *Wsq = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *WsqY = (DOUBLE *) MALLOC(N * K * sizeof(DOUBLE));

	CHAR side = 'R';
	CHAR uplo = 'U';
	INT SYMMM = K;
	INT SYMMN = N;
	DOUBLE alpha = 1.0;
	INT SYMMLDA = N;
	INT SYMMLDB = N;
	DOUBLE beta = 0;
	INT SYMMLDC = K;
	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, W, &SYMMLDA, Y, &SYMMLDB, &beta, \
			YtW, &SYMMLDC);

	DOUBLE *Dt = (DOUBLE *) MALLOC(2 * N * 2 * N * sizeof(DOUBLE));
	memset((void *) Dt, 0, 2 * N * 2 * N * sizeof(DOUBLE));
	INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		datacpy(&Dt[iterN * 2 * N], &YtW[iterN * N], N);
		datacpy(&Dt[iterN * 2 * N + N], &W[iterN * N], N);
		Dt[(N + iterN) * 2 * N + iterN] = SQRT(kappa);
	}
	DOUBLE Lf;
	qp_lipschitz(&Lf, Dt, 2 * N, 2 * N, NULL, 0);
	Lf = 0.5 * Lf;
	DOUBLE invLf = 1.0 / Lf;

	uplo = 'U';
	CHAR trans = 'N';
	INT SYRKN = N;
	INT SYRKK = N;
	alpha = 1.0;
	INT SYRKLDA = N;
	beta = 0;
	INT SYRKLDC = N;
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, W, &SYRKLDA, &beta, Wsq, \
		&SYRKLDC);

	side = 'L';
	uplo = 'U';
	SYMMM = N;
	SYMMN = K;
	alpha = 1.0;
	SYMMLDA = N;
	SYMMLDB = N;
	beta = 0;
	SYMMLDC = N;
	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, Wsq, &SYMMLDA, Y, &SYMMLDB, \
		&beta, WsqY, &SYMMLDC);

	INT MINMN = IMIN(M, K);
	INT MAXMN = IMAX(M, K);
	DOUBLE *sv = (DOUBLE *) MALLOC(MINMN * 1 * sizeof(DOUBLE));
	DOUBLE *svecsmall = (DOUBLE *) MALLOC(MINMN * MINMN * sizeof(DOUBLE));
	DOUBLE *sveclarge = (DOUBLE *) MALLOC(MAXMN * MINMN * sizeof(DOUBLE));
	INT *iwork = (INT *) MALLOC(16 * MINMN * sizeof(INT));
	DOUBLE workTemp;
	INT lwork = -1;
	nuclear_proximal_gesdd(B, NULL, mu, M, K, sv, svecsmall, sveclarge, &workTemp, \
			lwork, iwork);
	lwork = (INT) workTemp;
	DOUBLE *work = (DOUBLE *) MALLOC(lwork * 1 * sizeof(DOUBLE));

	INT AXPBYN;
	INT incx = 1;
	INT incy = 1;
	DOUBLE normsum;
	CHAR NORM;
	INT LANGEM;
	INT LANGEN;
	INT LANGELDA;
	CHAR transa;
	CHAR transb;
	INT GEMMM;
	INT GEMMN;
	INT GEMMK;
	INT GEMMLDA;
	INT GEMMLDB;
	INT GEMMLDC;

	DOUBLE tk = 1.0;
	DOUBLE tkm1 = 1.0;
	DOUBLE muk = mu * delta;
	if (initFlag == 0) {
		memset((void *) B, 0, M * K * sizeof(DOUBLE));
	}
	memset((void *) Bkm1, 0, M * K * sizeof(DOUBLE));
	if (initFlag == 0) {
		memset((void *) A, 0, M * N * sizeof(DOUBLE));
	}
	memset((void *) Akm1, 0, M * N * sizeof(DOUBLE));

	INT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		datacpy(YB, B, M * K);
		AXPBYN = M * K;
		AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);
		datacpy(YA, A, M * N);
		AXPBYN = M * N;
		AXPBY(&AXPBYN, &alpha, Akm1, &incx, &beta, YA, &incy);

		datacpy(BYtAmD, D, M * N);
		alpha = 1.0;
		beta = - 1.0;
		AXPBYN = M * N;
		AXPBY(&AXPBYN, &alpha, YA, &incx, &beta, BYtAmD, &incy);

		transa = 'N';
		transb = 'T';
		GEMMM = M;
		GEMMN = N;
		GEMMK = K;
		alpha = 1.0;
		GEMMLDA = M;
		GEMMLDB = N;
		beta = 1.0;
		GEMMLDC = M;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, YB, &GEMMLDA, Y, \
				&GEMMLDB, &beta, BYtAmD, &GEMMLDC);

		datacpy(Akm1, A, M * N);
		datacpy(A, YA, M * N);
		side = 'R';
		uplo = 'U';
		SYMMM = M;
		SYMMN = N;
		alpha = - invLf;
		SYMMLDA = N;
		SYMMLDB = M;
		beta = 1;
		SYMMLDC = M;
		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, Wsq, &SYMMLDA, BYtAmD, \
				&SYMMLDB, &beta, A, &SYMMLDC);
		datacpy(S, A, M * N);
		l1_proximal(A, NULL, invLf * muk * lambda, M * N);
		alpha = - Lf;
		beta = Lf;
		AXPBYN = M * N;
		AXPBY(&AXPBYN, &alpha, A, &incx, &beta, S, &incy);
		NORM = 'F';
		LANGEM = M;
		LANGEN = N;
		LANGELDA = M;
		normsum = LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

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
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, BYtAmD, \
				&GEMMLDA, WsqY, &GEMMLDB, &beta, B, &GEMMLDC);
		datacpy(S, B, M * K);
		nuclear_proximal_gesdd(B, NULL, invLf * muk, M, K, sv, \
						svecsmall, sveclarge, work, lwork, iwork);
		alpha = - Lf;
		beta = Lf;
		AXPBYN = M * K;
		AXPBY(&AXPBYN, &alpha, B, &incx, &beta, S, &incy);
		NORM = 'F';
		LANGEM = M;
		LANGEN = K;
		LANGELDA = M;
		normsum += LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

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
	FREE(Akm1);
	FREE(YB);
	FREE(YA);
	FREE(S);
	FREE(BYtAmD);
	FREE(Wsq);
	FREE(WsqY);
	FREE(YtW);
	FREE(Dt);
	FREE(sv);
	FREE(sveclarge);
	FREE(svecsmall);
	FREE(work);
	FREE(iwork);
}


void robust_operator_pca_apg(DOUBLE *B, DOUBLE *A, DOUBLE *D, DOUBLE *Y, \
				DOUBLE mu, DOUBLE lambda, DOUBLE kappa, INT numIters, \
				DOUBLE tolerance, DOUBLE delta, DOUBLE eta, INT initFlag, \
				INT M, INT N, INT K) {

	DOUBLE *Bkm1 = (DOUBLE *) MALLOC(M * K * sizeof(DOUBLE));
	DOUBLE *Akm1 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) MALLOC(M * K * sizeof(DOUBLE));
	DOUBLE *YA = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *BYtAmD = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *S = (DOUBLE *) MALLOC(M * IMAX(N, K) * sizeof(DOUBLE));
	DOUBLE *YtW = (DOUBLE *) MALLOC(K * N * sizeof(DOUBLE));
	DOUBLE *W = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	memset((void *) W, 0, N * N * sizeof(DOUBLE));

	INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		W[iterN * N + iterN] = 1;
	}

	CHAR side = 'R';
	CHAR uplo = 'U';
	INT SYMMM = K;
	INT SYMMN = N;
	DOUBLE alpha = 1.0;
	INT SYMMLDA = N;
	INT SYMMLDB = N;
	DOUBLE beta = 0;
	INT SYMMLDC = K;
	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, W, &SYMMLDA, Y, &SYMMLDB, &beta, \
			YtW, &SYMMLDC);

	DOUBLE *Dt = (DOUBLE *) MALLOC(2 * N * 2 * N * sizeof(DOUBLE));
	memset((void *) Dt, 0, 2 * N * 2 * N * sizeof(DOUBLE));
	for (iterN = 0; iterN < N; ++iterN) {
		datacpy(&Dt[iterN * 2 * N], &YtW[iterN * N], N);
		datacpy(&Dt[iterN * 2 * N + N], &W[iterN * N], N);
		Dt[(N + iterN) * 2 * N + iterN] = SQRT(kappa);
	}
	DOUBLE Lf;
	qp_lipschitz(&Lf, Dt, 2 * N, 2 * N, NULL, 0);
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

	INT AXPBYN;
	INT AXPYN;
	INT incx = 1;
	INT incy = 1;
	DOUBLE normsum;
	CHAR NORM;
	INT LANGEM;
	INT LANGEN;
	INT LANGELDA;
	CHAR transa;
	CHAR transb;
	INT GEMMM;
	INT GEMMN;
	INT GEMMK;
	INT GEMMLDA;
	INT GEMMLDB;
	INT GEMMLDC;

	DOUBLE tk = 1.0;
	DOUBLE tkm1 = 1.0;
	DOUBLE muk = mu * delta;
	if (initFlag == 0) {
		memset((void *) B, 0, M * K * sizeof(DOUBLE));
	}
	memset((void *) Bkm1, 0, M * K * sizeof(DOUBLE));
	if (initFlag == 0) {
		memset((void *) A, 0, M * N * sizeof(DOUBLE));
	}
	memset((void *) Akm1, 0, M * N * sizeof(DOUBLE));

	INT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		datacpy(YB, B, M * K);
		AXPBYN = M * K;
		AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);
		datacpy(YA, A, M * N);
		AXPBYN = M * N;
		AXPBY(&AXPBYN, &alpha, Akm1, &incx, &beta, YA, &incy);

		datacpy(BYtAmD, D, M * N);
		alpha = 1.0;
		beta = - 1.0;
		AXPBYN = M * N;
		AXPBY(&AXPBYN, &alpha, YA, &incx, &beta, BYtAmD, &incy);

		transa = 'N';
		transb = 'T';
		GEMMM = M;
		GEMMN = N;
		GEMMK = K;
		alpha = 1.0;
		GEMMLDA = M;
		GEMMLDB = N;
		beta = 1.0;
		GEMMLDC = M;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, YB, &GEMMLDA, Y, \
				&GEMMLDB, &beta, BYtAmD, &GEMMLDC);

		datacpy(Akm1, A, M * N);
		datacpy(A, YA, M * N);
		alpha = - invLf;
		AXPYN = M * N;
		AXPY(&AXPYN, &alpha, BYtAmD, &incx, A, &incy);
		datacpy(S, A, M * N);
		l1_proximal(A, NULL, invLf * muk * lambda, M * N);
		alpha = - Lf;
		beta = Lf;
		AXPBYN = M * N;
		AXPBY(&AXPBYN, &alpha, A, &incx, &beta, S, &incy);
		NORM = 'F';
		LANGEM = M;
		LANGEN = N;
		LANGELDA = M;
		normsum = LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

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
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, BYtAmD, \
				&GEMMLDA, Y, &GEMMLDB, &beta, B, &GEMMLDC);
		datacpy(S, B, M * K);
		nuclear_proximal(B, NULL, invLf * muk, M, K, sv, svecsmall, sveclarge, \
						work, lwork);
		alpha = - Lf;
		beta = Lf;
		AXPBYN = M * K;
		AXPBY(&AXPBYN, &alpha, B, &incx, &beta, S, &incy);
		NORM = 'F';
		LANGEM = M;
		LANGEN = K;
		LANGELDA = M;
		normsum += LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);

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
	FREE(Akm1);
	FREE(YB);
	FREE(YA);
	FREE(S);
	FREE(BYtAmD);
	FREE(Dt);
	FREE(sv);
	FREE(sveclarge);
	FREE(svecsmall);
	FREE(work);
	FREE(W);
}
