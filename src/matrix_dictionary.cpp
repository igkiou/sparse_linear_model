/*
 * matrix_dictionary.c
 *
 *  Created on: May 3, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useblas.h"
#ifdef USE_CUDA
#include "usecuda.h"
#include "cudakernels.h"
#endif
#include "useinterfaces.h"
#include "matrix_proximal.h"
#include "l1_proximal.h"
#include "matrix_dictionary.h"
#include "utils.h"

/*
 * TODO: Transfer SVD to GPU?
 * TODO: Add sparse matrix multiplication?
 * TODO: Change convergence criterion here and in PCA to be the same as in
 * kernel/metric learning versions.
*/
#ifdef USE_CUDA
void matrix_dictionary_learning_lowrank_apg_cuda(CUHANDLE handle, \
				CUDOUBLE *h_B, CUDOUBLE *h_XAt, CUDOUBLE *h_AAt, CUDOUBLE mu, \
				CUDOUBLE kappa, CUDOUBLE tolerance, CUDOUBLE delta,	\
				CUINT numIters, CUDOUBLE eta, CUINT initFlag, CUINT M, CUINT N, \
				CUINT K, CUINT numSamples) {

	CUINT MN = M * N;
	CUDOUBLE *B = NULL;
	cumalloc(handle, (void**)&B, MN * K * sizeof(CUDOUBLE));
	CUDOUBLE *XAt = NULL;
	cumalloc(handle, (void**)&XAt, MN * K * sizeof(CUDOUBLE));
	CUDOUBLE *AAt = NULL;
	cumalloc(handle, (void**)&AAt, K * K * sizeof(CUDOUBLE));
	CUDOUBLE *Bkm1 = NULL;
	cumalloc(handle, (void**)&Bkm1, MN * K * sizeof(CUDOUBLE));
	CUDOUBLE *YB = NULL;
	cumalloc(handle, (void**)&YB, MN * K * sizeof(CUDOUBLE));
	CUDOUBLE *DA = NULL;
	cumalloc(handle, (void**)&DA, MN * K * sizeof(CUDOUBLE));
	CUDOUBLE *S = NULL;
	cumalloc(handle, (void**)&S, MN * sizeof(CUDOUBLE));
	CUDOUBLE *h_Lf = (CUDOUBLE *) MALLOC(K * 1 * sizeof(CUDOUBLE));

	CUINT iterK;
	for (iterK = 0; iterK < K; ++iterK) {
		h_Lf[iterK] = h_AAt[iterK * K + iterK] / numSamples + kappa;
	}

	CUINT MINMN = IMIN(M, N);
	CUINT MAXMN = IMAX(M, N);
	CUDOUBLE *sv = NULL;
	cumalloc(handle, (void**)&sv, MINMN * 1 * sizeof(CUDOUBLE));
	CUDOUBLE *svecsmall = NULL;
	cumalloc(handle, (void**)&svecsmall, MINMN * MINMN * sizeof(CUDOUBLE));
	CUDOUBLE *sveclarge = NULL;
	cumalloc(handle, (void**)&sveclarge, MAXMN * MINMN * sizeof(CUDOUBLE));

	CUINT AXPBYN;
	CUDOUBLE alpha;
	CUDOUBLE beta;
	CUINT incx = 1;
	CUINT incy = 1;
	CUDOUBLE normsum;
	CUDOUBLE normtemp;
	CUINT NRM2N = MN;
	CUDOUBLE *Btemp;
	CUDOUBLE Lftemp;
	CUCHAR side = 'R';
	CUCHAR uplo = 'U';
	CUINT SYMMM = MN;
	CUINT SYMMN = K;
	CUINT SYMMLDA = K;
	CUINT SYMMLDB = MN;
	CUINT SYMMLDC = MN;

	CUDOUBLE tk = 1.0;
	CUDOUBLE tkm1 = 1.0;
	CUDOUBLE muk = mu / delta;
	if (initFlag == 0) {
		cumemset(handle, (void *) B, 0, MN * K * sizeof(CUDOUBLE));
	} else {
		cuh2dcpy(handle, B, h_B, MN * K);
	}
	cuh2dcpy(handle, XAt, h_XAt, MN * K);
	cuh2dcpy(handle, AAt, h_AAt, K * K);
	cumemset(handle, (void *) Bkm1, 0, MN * K * sizeof(CUDOUBLE));

	CUINT iter = 0;
	while (1) {
		cud2dcpy(handle, YB, B, MN * K);
		AXPBYN = MN * K;
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		CUAXPBY(handle, AXPBYN, &alpha, Bkm1, incx, &beta, YB, incy);

		cud2dcpy(handle, DA, XAt, MN * K);

		alpha = 1.0 / (CUDOUBLE) numSamples;
		beta = - 1.0 / (CUDOUBLE) numSamples;
		CUSYMM(handle, side, uplo, SYMMM, SYMMN, &alpha, AAt, SYMMLDA, YB, \
			SYMMLDB, &beta, DA, SYMMLDC);

		cud2dcpy(handle, Bkm1, B, MN * K);
		normsum = 0;
		for (iterK = 0; iterK < K; ++iterK) {
			Btemp = &B[MN * iterK];
			Lftemp = h_Lf[iterK];
			cud2dcpy(handle, Btemp, &YB[MN * iterK], M * N);
			AXPBYN = M * N;
			alpha = - 0.5 / Lftemp;
			beta = 1.0 - 0.5 * kappa / Lftemp;
			CUAXPBY(handle, AXPBYN, &alpha, &DA[iterK * MN], incx, &beta, Btemp, \
					incy);

			cud2dcpy(handle, S, Btemp, M * N);
			nuclear_proximal_cuda(handle, Btemp, NULL, 0.5 * muk / Lftemp, M, N, sv, \
							svecsmall, sveclarge);

			AXPBYN = M * N;
			alpha = - 2.0 * Lftemp;
			beta = 2.0 * Lftemp;
			CUAXPBY(handle, AXPBYN, &alpha, Btemp, incx, &beta, S, incy);
			CUNRM2(handle, NRM2N, S, incx, &normtemp);
			normsum += normtemp;
		}

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
	cud2hcpy(handle, h_B, B, MN * K);

	cufree(handle, B);
	cufree(handle, XAt);
	cufree(handle, AAt);
	cufree(handle, Bkm1);
	cufree(handle, YB);
	cufree(handle, DA);
	cufree(handle, S);
	FREE(h_Lf);
	cufree(handle, sv);
	cufree(handle, svecsmall);
	cufree(handle, sveclarge);
}
#endif

void matrix_dictionary_learning_lowrank_apg_parallel(DOUBLE *B, DOUBLE *XAt, \
				DOUBLE *AAt, DOUBLE mu, DOUBLE kappa, DOUBLE tolerance, \
				DOUBLE delta, INT numIters, DOUBLE eta, INT initFlag, INT M, \
				INT N, INT K, INT numSamples) {

	INT MN = M * N;
	DOUBLE *Bkm1 = (DOUBLE *) CMALLOC(MN * K * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) CMALLOC(MN * K * sizeof(DOUBLE));
	DOUBLE *DA = (DOUBLE *) CMALLOC(MN * K * sizeof(DOUBLE));
	DOUBLE *Lf = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));

	INT iterK;
	for (iterK = 0; iterK < K; ++iterK) {
		Lf[iterK] = AAt[iterK * K + iterK] / numSamples + kappa;
	}

	INT AXPBYN;
	INT AXPBYN1;
	DOUBLE alpha;
	DOUBLE beta;
	DOUBLE alpha1;
	DOUBLE beta1;
	INT incx = 1;
	INT incy = 1;
	DOUBLE normsum;
	CHAR NORM = 'F';
	INT LANGEM = M;
	INT LANGEN = N;
	INT LANGELDA = M;
	DOUBLE *Btemp;
	DOUBLE Lftemp;
	CHAR side = 'R';
	CHAR uplo = 'U';
	INT SYMMM = MN;
	INT SYMMN = K;
	INT SYMMLDA = K;
	INT SYMMLDB = MN;
	INT SYMMLDC = MN;

	DOUBLE tk = 1.0;
	DOUBLE tkm1 = 1.0;
	DOUBLE muk = mu / delta;
	if (initFlag == 0) {
		memset((void *) B, 0, MN * K * sizeof(DOUBLE));
	}
	memset((void *) Bkm1, 0, MN * K * sizeof(DOUBLE));

	INT MINMN = IMIN(M, N);
	INT MAXMN = IMAX(M, N);
	DOUBLE *sv = NULL;
	DOUBLE *svecsmall = NULL;
	DOUBLE *sveclarge = NULL;
	DOUBLE workTemp;
	INT lwork = -1;
	nuclear_proximal(B, NULL, mu, M, N, sv, svecsmall, sveclarge, &workTemp, \
					lwork);
	lwork = (INT) workTemp;
	DOUBLE *work = NULL;

	DOUBLE *S = NULL;
	INT iter = 0;

	#pragma omp parallel private(iterK, Btemp, Lftemp, alpha1, beta1, \
			AXPBYN1, S, sv, svecsmall, sveclarge, work) \
	shared(B, Bkm1, Lf, YB, DA, XAt, AAt, alpha, beta, muk, eta, mu, \
			iter, numIters, K, M, N, MN, kappa, incx, incy, NORM, LANGEM, \
			LANGEN, LANGELDA, numSamples, side, uplo, SYMMM, SYMMN, SYMMLDA, \
			SYMMLDB, SYMMLDC, AXPBYN, tolerance, lwork) \
	reduction(+: normsum)
	{
		S = (DOUBLE *) CMALLOC(MN * sizeof(DOUBLE));
		sv = (DOUBLE *) CMALLOC(MINMN * 1 * sizeof(DOUBLE));
		svecsmall = (DOUBLE *) CMALLOC(MINMN * MINMN * sizeof(DOUBLE));
		sveclarge = (DOUBLE *) CMALLOC(MAXMN * MINMN * sizeof(DOUBLE));
		work = (DOUBLE *) CMALLOC(lwork * 1 * sizeof(DOUBLE));
		while (iter < numIters) {

			#pragma omp master
			{
				alpha = - (tkm1 - 1.0) / tk;
				beta = 1.0 + (tkm1 - 1.0) / tk;
				datacpy(YB, B, MN * K);
				AXPBYN = MN * K;
				AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);

				datacpy(DA, XAt, MN * K);

				alpha = 1.0 / (DOUBLE) numSamples;
				beta = - 1.0 / (DOUBLE) numSamples;
				SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, AAt, &SYMMLDA, YB, \
						&SYMMLDB, &beta, DA, &SYMMLDC);

				datacpy(Bkm1, B, MN * K);
				normsum = 0;
			}
			#pragma omp barrier

			#pragma omp for
			for (iterK = 0; iterK < K; ++iterK) {
				Btemp = &B[MN * iterK];
				Lftemp = Lf[iterK];
				datacpy(Btemp, &YB[MN * iterK], M * N);
				AXPBYN1 = M * N;
				alpha1 = - 0.5 / Lftemp;
				beta1 = 1.0 - 0.5 * kappa / Lftemp;
				AXPBY(&AXPBYN1, &alpha1, &DA[iterK * MN], &incx, &beta1, Btemp, &incy);
				datacpy(S, Btemp, M * N);
				nuclear_proximal(Btemp, NULL, 0.5 * muk / Lftemp, M, N, sv, \
								svecsmall, sveclarge, work, lwork);
				AXPBYN1 = M * N;
				alpha1 = - 2.0 * Lftemp;
				beta1 = 2.0 * Lftemp;
				AXPBY(&AXPBYN1, &alpha1, Btemp, &incx, &beta1, S, &incy);
				normsum += LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);
			}

			#pragma omp master
			{
				tkm1 = tk;
				tk = 0.5 * (1.0 + SQRT(4.0 * SQR(tkm1) + 1.0));
				muk = IMAX(eta * muk, mu);

				++iter;
			}
			#pragma omp barrier

			if (normsum < tolerance) {
				break;
			}
		}

		CFREE(S);
		CFREE(sv);
		CFREE(svecsmall);
		CFREE(sveclarge);
		CFREE(work);
	}

	CFREE(Bkm1);
	CFREE(YB);
	CFREE(DA);
	CFREE(Lf);
}

void matrix_dictionary_learning_lowrank_apg(DOUBLE *B, DOUBLE *XAt, DOUBLE *AAt, \
				DOUBLE mu, DOUBLE kappa, DOUBLE tolerance, DOUBLE delta,\
				INT numIters, DOUBLE eta, INT initFlag, INT M, INT N, INT K, \
				INT numSamples) {

	INT MN = M * N;
	DOUBLE *Bkm1 = (DOUBLE *) MALLOC(MN * K * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) MALLOC(MN * K * sizeof(DOUBLE));
	DOUBLE *DA = (DOUBLE *) MALLOC(MN * K * sizeof(DOUBLE));
	DOUBLE *S = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *Lf = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));

	INT iterK;
	for (iterK = 0; iterK < K; ++iterK) {
		Lf[iterK] = AAt[iterK * K + iterK] / numSamples + kappa;
	}

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

	INT AXPBYN;
	DOUBLE alpha;
	DOUBLE beta;
	INT incx = 1;
	INT incy = 1;
	DOUBLE normsum;
	CHAR NORM = 'F';
	INT LANGEM = M;
	INT LANGEN = N;
	INT LANGELDA = M;
	DOUBLE *Btemp;
	DOUBLE Lftemp;
	CHAR side = 'R';
	CHAR uplo = 'U';
	INT SYMMM = MN;
	INT SYMMN = K;
	INT SYMMLDA = K;
	INT SYMMLDB = MN;
	INT SYMMLDC = MN;

	DOUBLE tk = 1.0;
	DOUBLE tkm1 = 1.0;
	DOUBLE muk = mu / delta;
	if (initFlag == 0) {
		memset((void *) B, 0, MN * K * sizeof(DOUBLE));
	}
	memset((void *) Bkm1, 0, MN * K * sizeof(DOUBLE));

	INT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		datacpy(YB, B, MN * K);
		AXPBYN = MN * K;
		AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);

		datacpy(DA, XAt, MN * K);

		alpha = 1.0 / (DOUBLE) numSamples;
		beta = - 1.0 / (DOUBLE) numSamples;
		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, AAt, &SYMMLDA, YB, \
				&SYMMLDB, &beta, DA, &SYMMLDC);

		datacpy(Bkm1, B, MN * K);
		normsum = 0;
		for (iterK = 0; iterK < K; ++iterK) {
			Btemp = &B[MN * iterK];
			Lftemp = Lf[iterK];
			datacpy(Btemp, &YB[MN * iterK], M * N);
			AXPBYN = M * N;
			alpha = - 0.5 / Lftemp;
			beta = 1.0 - 0.5 * kappa / Lftemp;
			AXPBY(&AXPBYN, &alpha, &DA[iterK * MN], &incx, &beta, Btemp, &incy);
			datacpy(S, Btemp, M * N);
			nuclear_proximal(Btemp, NULL, 0.5 * muk / Lftemp, M, N, sv, \
							svecsmall, sveclarge, work, lwork);
			AXPBYN = M * N;
			alpha = - 2.0 * Lftemp;
			beta = 2.0 * Lftemp;
			AXPBY(&AXPBYN, &alpha, Btemp, &incx, &beta, S, &incy);
			normsum += LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);
		}

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

	FREE(Bkm1);
	FREE(YB);
	FREE(DA);
	FREE(S);
	FREE(Lf);
	FREE(sv);
	FREE(svecsmall);
	FREE(sveclarge);
	FREE(work);
}

void operator_dictionary_learning_lowrank_weighted_apg_parallel(DOUBLE *B, \
				DOUBLE *XWsqYAt, DOUBLE *AAt, DOUBLE *YtWsqY, DOUBLE mu, \
				DOUBLE kappa, DOUBLE tolerance, DOUBLE delta, INT numIters, \
				DOUBLE eta, INT initFlag, INT M, INT N, INT F, INT K, \
				INT numSamples) {

	INT MN = M * N;
	INT MF = M * F;
	DOUBLE *Bkm1 = (DOUBLE *) CMALLOC(MF * K * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) CMALLOC(MF * K * sizeof(DOUBLE));
	DOUBLE *YBYtWsqY = (DOUBLE *) CMALLOC(MF * K * sizeof(DOUBLE));
	DOUBLE *DA = (DOUBLE *) CMALLOC(MF * K * sizeof(DOUBLE));
	DOUBLE *YtWsqYtemp = (DOUBLE *) CMALLOC(F * F * sizeof(DOUBLE));
	DOUBLE *Lf = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));

	DOUBLE LfY;
	datacpy(YtWsqYtemp, YtWsqY, F * F);
	qp_lipschitz(&LfY, NULL, N, F, YtWsqYtemp, 1);
	LfY = LfY / 2;

	INT iterK;
	for (iterK = 0; iterK < K; ++iterK) {
		Lf[iterK] = AAt[iterK * K + iterK] * LfY / numSamples + kappa;
	}

	INT AXPBYN;
	INT AXPBYN1;
	DOUBLE alpha;
	DOUBLE beta;
	DOUBLE alpha1;
	DOUBLE beta1;
	INT incx = 1;
	INT incy = 1;
	CHAR side;
	CHAR uplo;
	INT SYMMM;
	INT SYMMN;
	INT SYMMLDA;
	INT SYMMLDB;
	INT SYMMLDC;
	CHAR NORM = 'F';
	INT LANGEM = M;
	INT LANGEN = N;
	INT LANGELDA = M;
	DOUBLE *Btemp;
	DOUBLE Lftemp;
	DOUBLE normsum;

	DOUBLE tk = 1.0;
	DOUBLE tkm1 = 1.0;
	DOUBLE muk = mu / delta;
	if (initFlag == 0) {
		memset((void *) B, 0, MF * K * sizeof(DOUBLE));
	}
	memset((void *) Bkm1, 0, MF * K * sizeof(DOUBLE));

	INT MINMN = IMIN(M, F);
	INT MAXMN = IMAX(M, F);
	DOUBLE *sv = NULL;
	DOUBLE *svecsmall = NULL;
	DOUBLE *sveclarge = NULL;
	DOUBLE workTemp;
	INT lwork = -1;
	nuclear_proximal(B, NULL, mu, M, F, sv, svecsmall, sveclarge, &workTemp, \
					lwork);
	lwork = (INT) workTemp;
	DOUBLE *work = NULL;

	DOUBLE *S = NULL;
	INT iter = 0;

	#pragma omp parallel private(iterK, Btemp, Lftemp, alpha1, beta1, \
			AXPBYN1, S, sv, svecsmall, sveclarge, work) \
	shared(B, Bkm1, Lf, YB, DA, XWsqYAt, AAt, alpha, beta, muk, eta, mu, \
			iter, numIters, K, M, N, MN, F, MF, kappa, incx, incy, NORM, LANGEM, \
			LANGEN, LANGELDA, numSamples, side, uplo, SYMMM, SYMMN, SYMMLDA, \
			SYMMLDB, SYMMLDC, AXPBYN, tolerance, lwork, YtWsqY, YBYtWsqY) \
	reduction(+: normsum)
	{
		S = (DOUBLE *) CMALLOC(MF * sizeof(DOUBLE));
		sv = (DOUBLE *) CMALLOC(MINMN * 1 * sizeof(DOUBLE));
		svecsmall = (DOUBLE *) CMALLOC(MINMN * MINMN * sizeof(DOUBLE));
		sveclarge = (DOUBLE *) CMALLOC(MAXMN * MINMN * sizeof(DOUBLE));
		work = (DOUBLE *) CMALLOC(lwork * 1 * sizeof(DOUBLE));
		while (iter < numIters) {

			#pragma omp master
			{
				alpha = - (tkm1 - 1.0) / tk;
				beta = 1.0 + (tkm1 - 1.0) / tk;
				datacpy(YB, B, MF * K);
				AXPBYN = MF * K;
				AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);

				side = 'R';
				uplo = 'U';
				SYMMM = M;
				SYMMN = F;
				alpha = 1;
				SYMMLDA = F;
				beta = 0;
				SYMMLDB = M;
				SYMMLDC = M;
			}
			#pragma omp barrier

			#pragma omp for
			for (iterK = 0; iterK < K; ++iterK) {
				SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, YtWsqY, \
					&SYMMLDA, &YB[MF * iterK], &SYMMLDB, &beta, \
					&YBYtWsqY[MF * iterK], &SYMMLDC);
			}

			#pragma omp master
			{
				datacpy(DA, XWsqYAt, MF * K);

				side = 'R';
				uplo = 'U';
				SYMMM = MF;
				SYMMN = K;
				alpha = 1.0 / (DOUBLE) numSamples;
				SYMMLDA = K;
				beta = - 1.0 / (DOUBLE) numSamples;
				SYMMLDB = MF;
				SYMMLDC = MF;
				SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, AAt, &SYMMLDA, YBYtWsqY, \
						&SYMMLDB, &beta, DA, &SYMMLDC);

				datacpy(Bkm1, B, MF * K);
				normsum = 0;
			}
			#pragma omp barrier

			#pragma omp for
			for (iterK = 0; iterK < K; ++iterK) {
				Btemp = &B[MF * iterK];
				Lftemp = Lf[iterK];
				datacpy(Btemp, &YB[MN * iterK], M * F);
				AXPBYN1 = M * F;
				alpha1 = - 0.5 / Lftemp;
				beta1 = 1.0 - 0.5 * kappa / Lftemp;
				AXPBY(&AXPBYN1, &alpha1, &DA[iterK * MF], &incx, &beta1, Btemp, &incy);
				datacpy(S, Btemp, M * F);
				nuclear_proximal(Btemp, NULL, 0.5 * muk / Lftemp, M, F, sv, \
								svecsmall, sveclarge, work, lwork);
				AXPBYN1 = M * F;
				alpha1 = - 2.0 * Lftemp;
				beta1 = 2.0 * Lftemp;
				AXPBY(&AXPBYN1, &alpha1, Btemp, &incx, &beta1, S, &incy);
				normsum += LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);
			}

			#pragma omp master
			{
				tkm1 = tk;
				tk = 0.5 * (1.0 + SQRT(4.0 * SQR(tkm1) + 1.0));
				muk = IMAX(eta * muk, mu);

				++iter;
			}
			#pragma omp barrier

			if (normsum < tolerance) {
				break;
			}
		}
		CFREE(S);
		CFREE(sv);
		CFREE(svecsmall);
		CFREE(sveclarge);
		CFREE(work);
	}

	CFREE(Bkm1);
	CFREE(YB);
	CFREE(YBYtWsqY);
	CFREE(DA);
	CFREE(Lf);
	CFREE(YtWsqYtemp);

}

void operator_dictionary_learning_lowrank_weighted_apg(DOUBLE *B, DOUBLE *XWsqYAt, \
				DOUBLE *AAt, DOUBLE *YtWsqY, DOUBLE mu, DOUBLE kappa, \
				DOUBLE tolerance, DOUBLE delta, INT numIters, DOUBLE eta, \
				INT initFlag, INT M, INT N, INT F, INT K, INT numSamples) {

	INT MN = M * N;
	INT MF = M * F;
	DOUBLE *Bkm1 = (DOUBLE *) MALLOC(MF * K * sizeof(DOUBLE));
	DOUBLE *YB = (DOUBLE *) MALLOC(MF * K * sizeof(DOUBLE));
	DOUBLE *YBYtWsqY = (DOUBLE *) MALLOC(MF * K * sizeof(DOUBLE));
	DOUBLE *DA = (DOUBLE *) MALLOC(MF * K * sizeof(DOUBLE));
	DOUBLE *S = (DOUBLE *) MALLOC(M * F * sizeof(DOUBLE));
	DOUBLE *YtWsqYtemp = (DOUBLE *) MALLOC(F * F * sizeof(DOUBLE));
	DOUBLE *Lf = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));

	DOUBLE LfY;
	datacpy(YtWsqYtemp, YtWsqY, F * F);
	qp_lipschitz(&LfY, NULL, N, F, YtWsqYtemp, 1);
	LfY = LfY / 2;

	INT iterK;
	for (iterK = 0; iterK < K; ++iterK) {
		Lf[iterK] = AAt[iterK * K + iterK] * LfY / numSamples + kappa;
	}

	INT MINMN = IMIN(M, F);
	INT MAXMN = IMAX(M, F);
	DOUBLE *sv = (DOUBLE *) MALLOC(MINMN * 1 * sizeof(DOUBLE));
	DOUBLE *svecsmall = (DOUBLE *) MALLOC(MINMN * MINMN * sizeof(DOUBLE));
	DOUBLE *sveclarge = (DOUBLE *) MALLOC(MAXMN * MINMN * sizeof(DOUBLE));
	DOUBLE workTemp;
	INT lwork = -1;
	nuclear_proximal(B, NULL, mu, M, F, sv, svecsmall, sveclarge, &workTemp, \
					lwork);
	lwork = (INT) workTemp;
	DOUBLE *work = (DOUBLE *) MALLOC(lwork * 1 * sizeof(DOUBLE));

	INT AXPBYN;
	DOUBLE alpha;
	DOUBLE beta;
	INT incx;
	INT incy;
	CHAR side;
	CHAR uplo;
	INT SYMMM;
	INT SYMMN;
	INT SYMMLDA;
	INT SYMMLDB;
	INT SYMMLDC;
	CHAR NORM = 'F';
	INT LANGEM = M;
	INT LANGEN = N;
	INT LANGELDA = M;
	DOUBLE *Btemp;
	DOUBLE Lftemp;
	DOUBLE normsum;

	DOUBLE tk = 1.0;
	DOUBLE tkm1 = 1.0;
	DOUBLE muk = mu / delta;
	if (initFlag == 0) {
		memset((void *) B, 0, MF * K * sizeof(DOUBLE));
	}
	memset((void *) Bkm1, 0, MF * K * sizeof(DOUBLE));

	INT iter = 0;
	while (1) {
		alpha = - (tkm1 - 1.0) / tk;
		beta = 1.0 + (tkm1 - 1.0) / tk;
		datacpy(YB, B, MF * K);
		AXPBYN = MF * K;
		incx = 1;
		incy = 1;
		AXPBY(&AXPBYN, &alpha, Bkm1, &incx, &beta, YB, &incy);

		side = 'R';
		uplo = 'U';
		SYMMM = M;
		SYMMN = F;
		alpha = 1;
		SYMMLDA = F;
		beta = 0;
		SYMMLDB = M;
		SYMMLDC = M;
		for (iterK = 0; iterK < K; ++iterK) {
			SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, YtWsqY, \
				&SYMMLDA, &YB[MF * iterK], &SYMMLDB, &beta, \
				&YBYtWsqY[MF * iterK], &SYMMLDC);
		}

		datacpy(DA, XWsqYAt, MF * K);

		side = 'R';
		uplo = 'U';
		SYMMM = MF;
		SYMMN = K;
		alpha = 1.0 / (DOUBLE) numSamples;
		SYMMLDA = K;
		beta = - 1.0 / (DOUBLE) numSamples;
		SYMMLDB = MF;
		SYMMLDC = MF;
		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, AAt, &SYMMLDA, YBYtWsqY, \
				&SYMMLDB, &beta, DA, &SYMMLDC);

		datacpy(Bkm1, B, MF * K);
		normsum = 0;
		for (iterK = 0; iterK < K; ++iterK) {
			Btemp = &B[MF * iterK];
			Lftemp = Lf[iterK];

			datacpy(Btemp, &YB[MN * iterK], M * F);
			AXPBYN = M * F;
			alpha = - 0.5 / Lftemp;
			beta = 1.0 - 0.5 * kappa / Lftemp;
			AXPBY(&AXPBYN, &alpha, &DA[iterK * MF], &incx, &beta, Btemp, &incy);
			datacpy(S, Btemp, M * F);
			nuclear_proximal(Btemp, NULL, 0.5 * muk / Lftemp, M, F, sv, \
							svecsmall, sveclarge, work, lwork);
			AXPBYN = M * F;
			alpha = - 2.0 * Lftemp;
			incx = 1;
			beta = 2.0 * Lftemp;
			incy = 1;
			AXPBY(&AXPBYN, &alpha, Btemp, &incx, &beta, S, &incy);
			normsum += LANGE(&NORM, &LANGEM, &LANGEN, S, &LANGELDA, NULL);
		}

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

	FREE(Bkm1);
	FREE(YB);
	FREE(YBYtWsqY);
	FREE(DA);
	FREE(S);
	FREE(Lf);
	FREE(sv);
	FREE(svecsmall);
	FREE(sveclarge);
	FREE(work);
	FREE(YtWsqYtemp);

}

void matrix_dictionary_hard_thresholding_parallel(DOUBLE *Dc, DOUBLE *norm, INT rank, INT M, INT N, INT K) {

	INT MINMN = IMIN(M, N);
	INT MAXMN = IMAX(M, N);

	DOUBLE *sv;
	DOUBLE *svecsmall;
	DOUBLE *sveclarge;
	DOUBLE *normvec;
	if (norm != NULL) {
		normvec = (DOUBLE *) MALLOC(K * 1 * sizeof(DOUBLE));
	} else {
		normvec = NULL;
	}
	DOUBLE workTemp;
	INT lwork = -1;
	nuclear_hard_thresholding(Dc, normvec, rank, M, N, NULL, NULL, NULL, &workTemp, lwork);
	lwork = (INT) workTemp;
	DOUBLE *work;

	INT iterK;
	#pragma omp parallel private(iterK, sv, svecsmall, sveclarge, work) \
	shared(Dc, normvec, rank) firstprivate(M, N, K, lwork, MINMN, MAXMN)
	{
		sv = (DOUBLE *) CMALLOC(MINMN * 1 * sizeof(DOUBLE));
		svecsmall = (DOUBLE *) CMALLOC(MINMN * MINMN * sizeof(DOUBLE));
		sveclarge = (DOUBLE *) CMALLOC(MAXMN * MINMN * sizeof(DOUBLE));
		work = (DOUBLE *) CMALLOC(lwork * 1 * sizeof(DOUBLE));
		#pragma omp for
		for (iterK = 0; iterK < K; ++iterK) {
			if (normvec != NULL) {
				nuclear_hard_thresholding(&Dc[M * N * iterK], &normvec[iterK], rank, M, N, sv, \
						svecsmall, sveclarge, work, lwork);
			} else {
				nuclear_hard_thresholding(&Dc[M * N * iterK], normvec, rank, M, N, sv, \
						svecsmall, sveclarge, work, lwork);
			}
		}

		CFREE(sv);
		CFREE(svecsmall);
		CFREE(sveclarge);
		CFREE(work);
	}

	if (norm != NULL) {
		*norm = 0;
		for (iterK = 0; iterK < K; ++iterK) {
			*norm += normvec[iterK];
		}
		FREE(normvec);
	}
}

void matrix_dictionary_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *A, DOUBLE mu, \
		INT M, INT N, INT K, INT numSamples, INT derivFlag, DOUBLE *res) {

	INT MN = M * N;
	INT resFlag = 0;
	if (res == NULL) {
		res = (DOUBLE *) MALLOC(MN * numSamples * sizeof(DOUBLE));
		resFlag = 1;
	}

	datacpy(res, X, MN * numSamples);

	CHAR transa = 'N';
	CHAR transb = 'N';
	INT GEMMM = MN;
	INT GEMMN = numSamples;
	INT GEMMK = K;
	DOUBLE alpha = 1.0;
	INT GEMMLDA = MN;
	INT GEMMLDB = K;
	DOUBLE beta = - 1.0;
	INT GEMMLDC = MN;
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, D, &GEMMLDA, A, &GEMMLDB, &beta, res, &GEMMLDC);

	if (obj != NULL) {
		CHAR norm = 'F';
		INT LANGEM = MN;
		INT LANGEN = numSamples;
		INT LANGELDA = MN;
		DOUBLE normRes = LANGE(&norm, &LANGEM, &LANGEN, res, &LANGELDA, NULL);
		LANGEM = MN;
		LANGEN = K;
		LANGELDA = MN;
		DOUBLE normReg = LANGE(&norm, &LANGEM, &LANGEN, D, &LANGELDA, NULL);
		*obj = SQR(normRes) / numSamples * 0.5 + SQR(normReg) * mu * 0.5;
	}

	if (derivFlag == 1) {
		datacpy(deriv, D, MN * K);

		transa = 'N';
		transb = 'T';
		GEMMM = MN;
		GEMMN = K;
		GEMMK = numSamples;
		alpha = 1.0 / numSamples;
		GEMMLDA = MN;
		GEMMLDB = K;
		beta = mu;
		GEMMLDC = MN;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, res, &GEMMLDA, A, &GEMMLDB, &beta, deriv, &GEMMLDC);
	}

	if (resFlag == 1) {
		FREE(res);
	}
}

void matrix_dictionary_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *A, DOUBLE *Ksq, \
		DOUBLE mu, INT M, INT N, INT K, INT numSamples, INT derivFlag, DOUBLE *res, DOUBLE *Dt, DOUBLE *derivTemp) {

	INT MN = M * N;
	INT resFlag = 0;
	if (res == NULL) {
		res = (DOUBLE *) MALLOC(MN * numSamples * sizeof(DOUBLE));
		resFlag = 1;
	}

	INT DtFlag = 0;
	if (Dt == NULL) {
		Dt = (DOUBLE *) MALLOC(MN * K * sizeof(DOUBLE));
		DtFlag = 1;
	}

	CHAR transa = 'N';
	CHAR transb = 'N';
	INT GEMMM = M;
	INT GEMMN = N;
	INT GEMMK = N;
	DOUBLE alpha = 1.0;
	INT GEMMLDA = M;
	INT GEMMLDB = N;
	DOUBLE beta = 0;
	INT GEMMLDC = M;
	INT iterK;
	for (iterK = 0; iterK < K; ++iterK) {
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, &D[iterK * MN], &GEMMLDA, Ksq, &GEMMLDB, &beta, &Dt[iterK * MN], &GEMMLDC);
	}

	datacpy(res, X, MN * numSamples);

	transa = 'N';
	transb = 'N';
	GEMMM = MN;
	GEMMN = numSamples;
	GEMMK = K;
	alpha = 1.0;
	GEMMLDA = MN;
	GEMMLDB = K;
	beta = - 1.0;
	GEMMLDC = MN;
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Dt, &GEMMLDA, A, &GEMMLDB, &beta, res, &GEMMLDC);

	if (obj != NULL) {
		CHAR norm = 'F';
		INT LANGEM = MN;
		INT LANGEN = numSamples;
		INT LANGELDA = MN;
		DOUBLE normRes = LANGE(&norm, &LANGEM, &LANGEN, res, &LANGELDA, NULL);
		LANGEM = MN;
		LANGEN = K;
		LANGELDA = MN;
		DOUBLE normReg = LANGE(&norm, &LANGEM, &LANGEN, D, &LANGELDA, NULL);
		*obj = SQR(normRes) / numSamples * 0.5 + SQR(normReg) * mu * 0.5;
	}

	if (derivFlag == 1) {

		INT derivTempFlag = 0;
		if (derivTemp == NULL) {
			derivTemp = (DOUBLE *) MALLOC(MN * K * sizeof(DOUBLE));
			derivTempFlag = 1;
		}

		transa = 'N';
		transb = 'T';
		GEMMM = MN;
		GEMMN = K;
		GEMMK = numSamples;
		alpha = 1.0 / numSamples;
		GEMMLDA = MN;
		GEMMLDB = K;
		beta = 0;
//		beta = mu;
		GEMMLDC = MN;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, res, &GEMMLDA, A, &GEMMLDB, &beta, derivTemp, &GEMMLDC);

		datacpy(deriv, D, MN * K);
		transa = 'N';
		transb = 'T';
		GEMMM = M;
		GEMMN = N;
		GEMMK = N;
		alpha = 1.0;
		GEMMLDA = M;
		GEMMLDB = N;
		beta = mu;
		GEMMLDC = N;
		for (iterK = 0; iterK < K; ++iterK) {
			GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, &derivTemp[iterK * MN], &GEMMLDA, Ksq, &GEMMLDB, &beta, &deriv[iterK * MN], &GEMMLDC);
		}

		if (derivTempFlag == 1) {
			FREE(derivTemp);
		}
	}

	if (resFlag == 1) {
		FREE(res);
	}

	if (DtFlag == 1) {
		FREE(Dt);
	}
}
