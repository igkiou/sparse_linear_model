/*
 * ksvd.c
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useblas.h"
#include "useinterfaces.h"
#include "utils.h"
#include "ksvd.h"

/*
 * Atom optimization
 */
void optimize_atom(DOUBLE *D, DOUBLE* X, DOUBLE *Gamma, INT j, INT *unusedSigs, INT *replacedAtoms, INT N, INT K, INT numSamples, \
		INT *numUnusedp, INT *dataIndices, DOUBLE *gammaj, DOUBLE *Dj, DOUBLE *u, DOUBLE *s, DOUBLE *smallGamma, DOUBLE *smallX) {

	INT dataIndicesFlag = 0;
	if (dataIndices == NULL) {
		dataIndices = (INT*) MALLOC(numSamples * sizeof(INT));
		dataIndicesFlag = 1;
	}

	INT gammajFlag = 0;
	if (gammaj == NULL) {
		gammaj = (DOUBLE*) MALLOC(numSamples * sizeof(DOUBLE));
		gammajFlag = 1;
	}

	INT DjFlag = 0;
	if (Dj == NULL) {
		Dj = (DOUBLE*) MALLOC(N * sizeof(DOUBLE));
		DjFlag = 1;
	}

	INT uFlag = 0;
	if (u == NULL) {
		u = (DOUBLE*) MALLOC(N * N * sizeof(DOUBLE));
		uFlag = 1;
	}

	INT sFlag = 0;
	if (s == NULL) {
		s = (DOUBLE*) MALLOC(N * sizeof(DOUBLE));
		sFlag = 1;
	}

	INT smallGammaFlag = 0;
	if (smallGamma == NULL) {
		smallGamma = (DOUBLE*) MALLOC(K * numSamples * sizeof(DOUBLE));
		smallGammaFlag = 1;
	}

	INT smallXFlag = 0;
	if (smallX == NULL) {
		smallX = (DOUBLE*) MALLOC(N * numSamples * sizeof(DOUBLE));
		smallXFlag = 1;
	}

	INT iterX;
	INT numNonZero = 0;
	DOUBLE coeff;
	for (iterX = 0; iterX < numSamples; ++iterX) {
		coeff = Gamma[iterX * K + j];

		if (coeff != 0) {
			dataIndices[numNonZero] = iterX;
			gammaj[numNonZero] = coeff;
			++numNonZero;
		}
	}

	if (numNonZero == 0) {

		INT numUnused = *numUnusedp;
		INT maxSignals = IMIN(5000, numUnused);

		randperm(dataIndices, maxSignals);

		#pragma omp parallel for private(iterX) shared(smallX, smallGamma, X, Gamma, K, N, maxSignals, dataIndices, unusedSigs)
		for (iterX = 0; iterX < maxSignals; ++iterX) {
			datacpy(&smallGamma[iterX * K], &Gamma[unusedSigs[dataIndices[iterX]] * K], K);
			datacpy(&smallX[iterX * N], &X[unusedSigs[dataIndices[iterX]] * N], N);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		DOUBLE alpha = - 1;
		DOUBLE beta = 1;

		GEMM(&transa, &transb, &N, &maxSignals, &K, &alpha, D, &N, smallGamma, &K, &beta, smallX, &N);

		INT iterMax = -1;
		DOUBLE normMax = 0;
		DOUBLE normTemp;
		INT INCX = 1;

		#pragma omp parallel for private(iterX, normTemp) shared(normMax, iterMax, smallX, N, INCX, maxSignals)
		for (iterX = 0; iterX < maxSignals; ++iterX) {
			normTemp = NRM2(&N, &smallX[iterX * N], &INCX);
			#pragma omp critical
			{
				if (normTemp > normMax) {
					normMax = normTemp;
					iterMax = iterX;
				}
			}
		}

		datacpy(&D[j * N], &X[unusedSigs[dataIndices[iterMax]] * N], N);
		alpha = 1 / NRM2(&N, &D[j * N], &INCX);
		SCAL(&N, &alpha, &D[j * N], &INCX);

		replacedAtoms[j] = 1;
		datacpy(&unusedSigs[iterMax], &unusedSigs[iterMax + 1], numUnused - iterMax - 1);
		*numUnusedp = numUnused - 1;

	} else {

		INT INCX = 1;
		INT INCY = 1;

		#pragma omp parallel for private(iterX) shared(smallX, smallGamma, X, Gamma, K, N, numNonZero)
		for (iterX = 0; iterX < numNonZero; ++iterX) {
			datacpy(&smallGamma[iterX * K], &Gamma[dataIndices[iterX] * K], K);
			datacpy(&smallX[iterX * N], &X[dataIndices[iterX] * N], N);
		}

		datacpy(Dj, &D[j * N], N);

		DOUBLE alpha = 1;
		GER(&N, &numNonZero, &alpha, Dj, &INCX, gammaj, &INCY, smallX, &N);

		CHAR transa = 'N';
		CHAR transb = 'N';
		alpha = -1;
		DOUBLE beta = 1;

		GEMM(&transa, &transb, &N, &numNonZero, &K, &alpha, D, &N, smallGamma, &K, &beta, smallX, &N);
		CHAR jobu = 'S';
		CHAR jobvt = 'O';
		DOUBLE work_temp;
		DOUBLE *work;
		INT lwork = -1;
		INT INFO;

		GESVD(&jobu, &jobvt, &N, &numNonZero, smallX, &N, s, u, &N, NULL, &numNonZero, &work_temp, &lwork, &INFO);

		lwork = (INT) work_temp;
		work = (DOUBLE*) MALLOC(lwork * sizeof(DOUBLE));

		GESVD(&jobu, &jobvt, &N, &numNonZero, smallX, &N, s, u, &N, NULL, &numNonZero, work, &lwork, &INFO);

		DOUBLE maxSingVal = s[0];

		#pragma omp parallel for private(iterX) shared(smallX, maxSingVal, X, Gamma, K, j, numNonZero)
		for (iterX = 0; iterX < numNonZero; ++iterX) {
			Gamma[dataIndices[iterX] * K + j] = maxSingVal * smallX[iterX * N];
		}

		datacpy(&D[j * N], u, N);

		FREE(work);
	}

	if (dataIndicesFlag == 1) {
		FREE(dataIndices);
	}

	if (gammajFlag == 1) {
		FREE(gammaj);
	}

	if (DjFlag == 1) {
		FREE(Dj);
	}

	if (uFlag == 1) {
		FREE(u);
	}

	if (sFlag == 1) {
		FREE(s);
	}

	if (smallGammaFlag == 1) {
		FREE(smallGamma);
	}

	if (smallXFlag == 1) {
		FREE(smallX);
	}
}

/*
 * Clear dictionary
 */
void clear_dictionary(DOUBLE *D, DOUBLE *X, DOUBLE *Gamma, DOUBLE muThresh, INT *unusedSigs, INT *replacedAtoms, \
		INT N, INT K, INT numSamples, INT numUnused, DOUBLE *smallGamma, DOUBLE *smallX, DOUBLE *errorVector, \
		INT *sortedUnusedSigs, INT *useCount, DOUBLE *Gj) {

	INT smallGammaFlag = 0;
	if (smallGamma == NULL) {
		smallGamma = (DOUBLE*) MALLOC(K * numUnused * sizeof(DOUBLE));
		smallGammaFlag = 1;
	}

	INT smallXFlag = 0;
	if (smallX == NULL) {
		smallX = (DOUBLE*) MALLOC(N * numUnused * sizeof(DOUBLE));
		smallXFlag = 1;
	}

	INT errorVectorFlag = 0;
	if (errorVector == NULL) {
		errorVector = (DOUBLE *) MALLOC(numUnused * sizeof(DOUBLE));
		errorVectorFlag = 1;
	}

	INT sortedUnusedSigsFlag = 0;
	if (sortedUnusedSigs == NULL) {
		sortedUnusedSigs = (INT *) MALLOC(numUnused * sizeof(INT));
		sortedUnusedSigsFlag = 1;
	}

	INT useCountFlag = 0;
	if (useCount == NULL) {
		useCount = (INT *) MALLOC(K * sizeof(INT));
		useCountFlag = 1;
	}

	INT GjFlag = 0;
	if (Gj == NULL) {
		Gj = (DOUBLE *) MALLOC(K * sizeof(DOUBLE));
		GjFlag = 1;
	}

	INT useThresh = 4;
	INT iterX;
	INT iterK;
	INT iterK2;

	#pragma omp parallel for private(iterX) shared(smallX, smallGamma, X, Gamma, K, N, numUnused, unusedSigs)
	for (iterX = 0; iterX < numUnused; ++iterX) {
		datacpy(&smallGamma[iterX * K], &Gamma[unusedSigs[iterX] * K], K);
		datacpy(&smallX[iterX * N], &X[unusedSigs[iterX] * N], N);
	}

	CHAR transa = 'N';
	CHAR transb = 'N';
	DOUBLE alpha = - 1;
	DOUBLE beta = 1;

	GEMM(&transa, &transb, &N, &numUnused, &K, &alpha, D, &N, smallGamma, &K, &beta, smallX, &N);

	INT INCX = 1;
	INT INCY = 1;

	#pragma omp parallel for private(iterX) shared(errorVector, smallX, N, INCX, numUnused)
	for (iterX = 0; iterX < numUnused; ++iterX) {
		errorVector[iterX] = NRM2(&N, &smallX[iterX * N], &INCX);
		sortedUnusedSigs[iterX] = unusedSigs[iterX];
	}
	quicksort(errorVector, sortedUnusedSigs, numUnused);

	DOUBLE zerothresh = POW(10,-7);

	memset((void *) useCount, 0, K * sizeof(INT));
	for (iterX = 0; iterX < numSamples; ++iterX) {
		for (iterK = 0; iterK < K; ++iterK) {
			if (ABS(Gamma[iterX * K + iterK]) > zerothresh) {
				++useCount[iterK];
			}
		}
	}

	CHAR trans = 'T';
	INT GEMVM = N;
	INT GEMVN = K;
	alpha = 1;
	INT GEMVLDA = N;
	beta = 0;
	INT muThreshFlag;
	DOUBLE muThreshSq = SQR(muThresh);
	DOUBLE alphaScal;

	for (iterK = 0; iterK < K; ++iterK) {

		GEMV(&trans, &GEMVM, &GEMVN, &alpha, D, &GEMVLDA, &D[N * iterK], &INCX, &beta, Gj, &INCY);
		Gj[iterK] = 0;

		muThreshFlag = 0;
		for (iterK2 = 0; iterK2 < K; ++iterK2) {
			if (SQR(Gj[iterK2]) > muThreshSq) {
				muThreshFlag = 1;
				break;
			}
		}

		if(((muThreshFlag == 1) || (useCount[iterK] < useThresh)) && (replacedAtoms[iterK] == 0)) {
			datacpy(&D[iterK * N], &X[sortedUnusedSigs[numUnused - 1] * N], N);
			alphaScal = 1 / NRM2(&N, &D[iterK * N], &INCX);
			SCAL(&N, &alphaScal, &D[iterK * N], &INCX);
			--numUnused;
		}
	}

	if (smallGammaFlag == 1) {
		FREE(smallGamma);
	}

	if (smallXFlag == 1) {
		FREE(smallX);
	}

	if (errorVectorFlag == 1) {
		FREE(errorVector);
	}

	if (sortedUnusedSigsFlag == 1) {
		FREE(sortedUnusedSigs);
	}

	if (useCountFlag == 1) {
		FREE(useCount);
	}

	if (GjFlag == 1) {
		FREE(Gj);
	}
}

/*
 * Atom optimization general (both simple and coupled ksvd)
 */
void optimize_atom_general(DOUBLE *D, DOUBLE *Dorig, DOUBLE* X, DOUBLE *Gamma, DOUBLE *extPhi, DOUBLE *cholPhiLambda, INT j, \
		INT *unusedSigs, INT *replacedAtoms, INT Neq, INT K, INT numSamples, INT numMeasurements, INT *numUnusedp, \
		INT *dataIndices, DOUBLE *gammaj, DOUBLE *Dj, DOUBLE *u, DOUBLE *s, DOUBLE *smallGamma, DOUBLE *smallX, \
		DOUBLE *atom) {

	INT N = Neq;

	INT dataIndicesFlag = 0;
	if (dataIndices == NULL) {
		dataIndices = (INT*) MALLOC(numSamples * sizeof(INT));
		dataIndicesFlag = 1;
	}

	INT gammajFlag = 0;
	if (gammaj == NULL) {
		gammaj = (DOUBLE*) MALLOC(numSamples * sizeof(DOUBLE));
		gammajFlag = 1;
	}

	INT DjFlag = 0;
	if (Dj == NULL) {
		Dj = (DOUBLE*) MALLOC(N * sizeof(DOUBLE));
		DjFlag = 1;
	}

	INT uFlag = 0;
	if (u == NULL) {
		u = (DOUBLE*) MALLOC(N * N * sizeof(DOUBLE));
		uFlag = 1;
	}

	INT sFlag = 0;
	if (s == NULL) {
		s = (DOUBLE*) MALLOC(N * sizeof(DOUBLE));
		sFlag = 1;
	}

	INT smallGammaFlag = 0;
	if (smallGamma == NULL) {
		smallGamma = (DOUBLE*) MALLOC(K * numSamples * sizeof(DOUBLE));
		smallGammaFlag = 1;
	}

	INT smallXFlag = 0;
	if (smallX == NULL) {
		smallX = (DOUBLE*) MALLOC(N * numSamples * sizeof(DOUBLE));
		smallXFlag = 1;
	}

	INT atomFlag = 0;
	if (atom == NULL) {
		atom = (DOUBLE*) MALLOC(N * numSamples * sizeof(DOUBLE));
		atomFlag = 1;
	}

	/*
	 * Get line of codes
	 */
	INT iterX;
	INT numNonZero = 0;
	DOUBLE coeff;
	for (iterX = 0; iterX < numSamples; ++iterX) {
		coeff = Gamma[iterX * K + j];

		if (coeff != 0) {
			dataIndices[numNonZero] = iterX;
			gammaj[numNonZero] = coeff;
			++numNonZero;
		}
	}

	if (numNonZero == 0) {

		INT numUnused = *numUnusedp;
		INT maxSignals = IMIN(5000, numUnused);

		randperm(dataIndices, maxSignals);

		#pragma omp parallel for private(iterX) shared(smallX, smallGamma, X, Gamma, K, N, maxSignals, dataIndices, unusedSigs)
		for (iterX = 0; iterX < maxSignals; ++iterX) {
			datacpy(&smallGamma[iterX * K], &Gamma[unusedSigs[dataIndices[iterX]] * K], K);
			datacpy(&smallX[iterX * N], &X[unusedSigs[dataIndices[iterX]] * N], N);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		DOUBLE alpha = - 1;
		DOUBLE beta = 1;

		GEMM(&transa, &transb, &N, &maxSignals, &K, &alpha, D, &N, smallGamma, &K, &beta, smallX, &N);

		INT iterMax = -1;
		DOUBLE normMax = 0;
		DOUBLE normTemp;
		INT INCX = 1;

		#pragma omp parallel for private(iterX, normTemp) shared(normMax, iterMax, smallX, N, INCX, maxSignals)
		for (iterX = 0; iterX < maxSignals; ++iterX) {
			normTemp = NRM2(&N, &smallX[iterX * N], &INCX);
			#pragma omp critical
			{
				if (normTemp > normMax) {
					normMax = normTemp;
					iterMax = iterX;
				}
			}
		}

		datacpy(atom, &X[unusedSigs[dataIndices[iterMax]] * N], N);
		alpha = 1 / NRM2(&N, &D[j * N], &INCX);
		SCAL(&N, &alpha, atom, &INCX);

		replacedAtoms[j] = 1;
		datacpy(&unusedSigs[iterMax], &unusedSigs[iterMax + 1], numUnused - iterMax - 1);
		*numUnusedp = numUnused - 1;

	} else {

		INT INCX = 1;
		INT INCY = 1;

		#pragma omp parallel for private(iterX) shared(smallX, smallGamma, X, Gamma, K, N, numNonZero)
		for (iterX = 0; iterX < numNonZero; ++iterX) {
			datacpy(&smallGamma[iterX * K], &Gamma[dataIndices[iterX] * K], K);
			datacpy(&smallX[iterX * N], &X[dataIndices[iterX] * N], N);
		}

		datacpy(Dj, &D[j * N], N);

		DOUBLE alpha = 1;
		GER(&N, &numNonZero, &alpha, Dj, &INCX, gammaj, &INCY, smallX, &N);

		CHAR transa = 'N';
		CHAR transb = 'N';
		alpha = -1;
		DOUBLE beta = 1;

		GEMM(&transa, &transb, &N, &numNonZero, &K, &alpha, D, &N, smallGamma, &K, &beta, smallX, &N);

		CHAR jobu = 'S';
		CHAR jobvt = 'O';
		DOUBLE work_temp;
		DOUBLE *work;
		INT lwork = -1;
		INT INFO;

		GESVD(&jobu, &jobvt, &N, &numNonZero, smallX, &N, s, u, &N, NULL, &numNonZero, &work_temp, &lwork, &INFO);

		lwork = (INT) work_temp;
		work = (DOUBLE*) MALLOC(lwork * sizeof(DOUBLE));

		GESVD(&jobu, &jobvt, &N, &numNonZero, smallX, &N, s, u, &N, NULL, &numNonZero, work, &lwork, &INFO);

		DOUBLE maxSingVal = s[0];

		#pragma omp parallel for private(iterX) shared(smallX, maxSingVal, X, Gamma, K, j, numNonZero)
		for (iterX = 0; iterX < numNonZero; ++iterX) {
			Gamma[dataIndices[iterX] * K + j] = maxSingVal * smallX[iterX * N];
		}

		datacpy(atom, u, N);

		FREE(work);
	}

	if (numMeasurements == 0) {
		datacpy(&D[j * N], atom, N);
	} else {

		INT Norig = Neq - numMeasurements;
		INT trans = 'T';
		INT GEMVM = N;
		INT GEMVN = Norig;
		DOUBLE alpha = 1;
		INT GEMVLDA = N;
		DOUBLE beta = 0;
		INT INCX = 1;
		INT INCY = 1;

		GEMV(&trans, &GEMVM, &GEMVN, &alpha, extPhi, &GEMVLDA, atom, &INCX, &beta, &Dorig[j * Norig], &INCY);

		INT uplo = 'U';
		INT POTRSN = Norig;
		INT POTRSNRHS = 1;
		INT POTRSLDA = Norig;
		INT POTRSLDB = Norig;
		INT INFO;
		POTRS(&uplo, &POTRSN, &POTRSNRHS, cholPhiLambda, &POTRSLDA, &Dorig[j * Norig], &POTRSLDB, &INFO);

		alpha = 1 / NRM2(&Norig, &Dorig[j * Norig], &INCX);
		SCAL(&Norig, &alpha, &Dorig[j * Norig], &INCX);
		for (iterX = 0; iterX < numNonZero; ++iterX) {
			Gamma[dataIndices[iterX] * K + j] = Gamma[dataIndices[iterX] * K + j] / alpha;
		}

		trans = 'N';
		GEMVM = N;
		GEMVN = Norig;
		alpha = 1;
		GEMVLDA = N;
		beta = 0;
		INCX = 1;
		INCY = 1;

		GEMV(&trans, &GEMVM, &GEMVN, &alpha, extPhi, &GEMVLDA, &Dorig[j * Norig], &INCX, &beta, &D[j * N], &INCY);
	}

	if (dataIndicesFlag == 1) {
		FREE(dataIndices);
	}

	if (gammajFlag == 1) {
		FREE(gammaj);
	}

	if (DjFlag == 1) {
		FREE(Dj);
	}

	if (uFlag == 1) {
		FREE(u);
	}

	if (sFlag == 1) {
		FREE(s);
	}

	if (smallGammaFlag == 1) {
		FREE(smallGamma);
	}

	if (smallXFlag == 1) {
		FREE(smallX);
	}

	if (atomFlag == 1) {
		FREE(atom);
	}
}

void ksvd_update(DOUBLE *Deq, DOUBLE *D, DOUBLE *Xeq, DOUBLE *X, DOUBLE *Gamma, DOUBLE *extPhi, DOUBLE *cholPhiLambda, INT Norig, INT K, INT numSamples, INT numMeasurements) {

	INT *unusedSigs = (INT *) MALLOC(numSamples * sizeof(INT));
	int iterX;
	for (iterX = 0; iterX < numSamples; ++iterX) {
		unusedSigs[iterX] = iterX;
	}
	INT numUnused = numSamples;

	INT N;
	if (numMeasurements == 0) {
		N = Norig;
	} else {
		N = Norig + numMeasurements;
	}

	INT *replacedAtoms = (INT *) MALLOC(K * sizeof(INT));
	memset((void *) replacedAtoms, 0, K * sizeof(INT));

	INT *dataIndices = (INT *) MALLOC(numSamples * sizeof(INT));
	DOUBLE *gammaj = (DOUBLE*) MALLOC(numSamples * sizeof(DOUBLE));
	DOUBLE *Dj = (DOUBLE*) MALLOC(N * sizeof(DOUBLE));
	DOUBLE *u = (DOUBLE*) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *s = (DOUBLE*) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *smallGamma = (DOUBLE*) MALLOC(K * numSamples * sizeof(DOUBLE));
	DOUBLE *smallX = (DOUBLE*) MALLOC(N * numSamples * sizeof(DOUBLE));
	DOUBLE *atom = (DOUBLE*) MALLOC(N * 1 * sizeof(DOUBLE));

	INT *pDict = (INT *) MALLOC(K * sizeof(INT));
	DOUBLE muThresh = 0.99;

	if (numMeasurements == 0) {
		Deq = D;
		Xeq = X;
	}

	srand(time(NULL));
	randperm(pDict, K);
	INT iterK;
	for (iterK = 0; iterK < K; ++iterK) {
		optimize_atom_general(Deq, D, Gamma, Xeq, extPhi, cholPhiLambda, pDict[iterK], unusedSigs, replacedAtoms, N, K, \
				numSamples, numMeasurements, &numUnused, dataIndices, gammaj, Dj, u, s, smallGamma, smallX, atom);
	}

	clear_dictionary(D, Gamma, X, muThresh, unusedSigs, replacedAtoms, N, K, numSamples, numUnused, \
			smallGamma, smallX, gammaj, dataIndices, pDict, buffer);

	FREE(replacedAtoms);
	FREE(unusedSigs);
	FREE(dataIndices);
	FREE(gammaj);
	FREE(Dj);
	FREE(u);
	FREE(s);
	FREE(smallGamma);
	FREE(smallX);
	FREE(atom);
	FREE(pDict);
}
