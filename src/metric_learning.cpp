/*
 * metric_learning.c
 *
 *  Created on: Jul 14, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useinterfaces.h"
#include "useblas.h"
#include "utils.h"
#include "metric_learning.h"
#include "kernel_learning.h"
#include "matrix_proximal.h"
#include "distance.h"

DIST_LABEL_TYPE convertDistLabelName(CHAR distLabelName) {
	if ((distLabelName == 'B') || (distLabelName == 'b')) {
		return DIST_LABEL_BOUNDS;
	} else if ((distLabelName == 'T') || (distLabelName == 't')) {
		return DIST_LABEL_TARGETS;
	} else if ((distLabelName == 'R') || (distLabelName == 'r')) {
		return DIST_LABEL_RELATIONAL;
	} else if ((distLabelName == 'S') || (distLabelName == 's')) {
		return DIST_LABEL_SQRHINGE;
	} else if ((distLabelName == 'H') || (distLabelName == 'h')) {
		return DIST_LABEL_HUBERHINGE;
	} else {
		ERROR("Unknown distance label name.");
		return DIST_LABEL_INVALID;
	}
}

void convertDistanceLabelMat(INT *labelMat, DOUBLE *valVec, \
							DIST_LABEL_TYPE labelType, DOUBLE *matlabLabelMat, \
							INT numLabels, INT numSamples) {
	INT iterLabel;
	INT currInd1;
	INT currInd2;
	INT currInd3;
	INT currInd4;
	INT currLabelType;
	DOUBLE currVal;
	switch (labelType) {
	case DIST_LABEL_BOUNDS:
		for (iterLabel = 0; iterLabel < numLabels; ++iterLabel) {
			currInd1 = (INT) matlabLabelMat[iterLabel];
			currInd2 = (INT) matlabLabelMat[numLabels + iterLabel];
			currLabelType = \
					(INT) matlabLabelMat[2 * numLabels + iterLabel];
			currVal = matlabLabelMat[3 * numLabels + iterLabel];
			if ((currInd1 > numSamples) || (currInd1 < 1) \
				|| (currInd2 > numSamples) || (currInd2 < 1)) {
				ERROR("Invalid bound constraint: index out of bounds.");
			} else if ((currLabelType != 1) && (currLabelType != -1)) {
				ERROR("Invalid bound constraint: unrecognized type.");
			}
			labelMat[iterLabel] = currInd1 - 1;
			labelMat[numLabels + iterLabel] = currInd2 - 1;
			labelMat[2 * numLabels + iterLabel] = currLabelType;
			valVec[iterLabel] = currVal;
		}
		break;
	case DIST_LABEL_TARGETS:
		for (iterLabel = 0; iterLabel < numLabels; ++iterLabel) {
			currInd1 = (INT) matlabLabelMat[iterLabel];
			currInd2 = (INT) matlabLabelMat[numLabels + iterLabel];
			currVal = matlabLabelMat[2 * numLabels + iterLabel];
			if ((currInd1 > numSamples) || (currInd1 < 1) \
				|| (currInd2 > numSamples) || (currInd2 < 1)) {
				ERROR("Invalid label constraint: index out of bounds.");
			}
			labelMat[iterLabel] = currInd1 - 1;
			labelMat[numLabels + iterLabel] = currInd2 - 1;
			valVec[iterLabel] = currVal;
		}
		break;
	case DIST_LABEL_RELATIONAL:
	case DIST_LABEL_SQRHINGE:
	case DIST_LABEL_HUBERHINGE:
		for (iterLabel = 0; iterLabel < numLabels; ++iterLabel) {
			currInd1 = (INT) matlabLabelMat[iterLabel];
			currInd2 = (INT) matlabLabelMat[numLabels + iterLabel];
			currInd3 = (INT) matlabLabelMat[numLabels * 2 + iterLabel];
			currInd4 = (INT) matlabLabelMat[numLabels * 3 + iterLabel];
			currVal = matlabLabelMat[4 * numLabels + iterLabel];
			if ((currInd1 > numSamples) || (currInd1 < 1) \
				|| (currInd2 > numSamples) || (currInd2 < 1) \
				|| (currInd3 > numSamples) || (currInd3 < 1) \
				|| (currInd4 > numSamples) || (currInd4 < 1)) {
				ERROR("Invalid relational constraint: index out of bounds.");
			}
			labelMat[iterLabel] = currInd1 - 1;
			labelMat[numLabels + iterLabel] = currInd2 - 1;
			labelMat[numLabels * 2 + iterLabel] = currInd3 - 1;
			labelMat[numLabels * 3 + iterLabel] = currInd4 - 1;
			valVec[iterLabel] = currVal;
		}
		break;
	default:
		ERROR("Unrecognized distance label type.");
	}
}

// TODO: relational, target.
void itml(DOUBLE *A, DOUBLE *X, INT *constraintMat, DOUBLE *boundVec, \
		DOUBLE tolerance, DOUBLE gamma, INT maxEpochs, INT randomizationFlag, \
		INT N, INT numConstraints, INT *constraintPerm, DOUBLE *lambda, \
		DOUBLE *lambdaOld, DOUBLE *vec, DOUBLE *Avec) {

	INT constraintPermFlag = 0;
	if (constraintPerm == NULL) {
		constraintPerm = (INT *) MALLOC(numConstraints * 1 * sizeof(INT));
		constraintPermFlag = 1;
	}

	INT lambdaFlag = 0;
	if (lambda == NULL) {
		lambda = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
		lambdaFlag = 1;
	}

	INT lambdaOldFlag = 0;
	if (lambdaOld == NULL) {
		lambdaOld = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
		lambdaOldFlag = 1;
	}

	INT vecFlag = 0;
	if (vec == NULL) {
		vec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vecFlag = 1;
	}

	INT AvecFlag = 0;
	if (Avec == NULL) {
		Avec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		AvecFlag = 1;
	}

	INT currConstraint;
	INT iterConstraint;
	INT currConstraintType;
	INT epochCount;
	INT currInd1;
	INT currInd2;
	DOUBLE *currX1;
	DOUBLE *currX2;
	DOUBLE wtw;
	DOUBLE gammaProj;
	DOUBLE alphaParam;
	DOUBLE betaParam;
	DOUBLE normSum;
	DOUBLE normDiff;
	DOUBLE conv;

	CHAR UPLO = 'U';
	INT INCX = 1;
	INT INCY = 1;
	INT AXPYN = N;
	INT AXPYN2 = numConstraints;
	INT DOTN = N;
	INT SYMVN = N;
	INT SYMVLDA = N;
	INT SYRN = N;
	INT SYRLDA = N;
	INT NRM2N = numConstraints;
	DOUBLE alpha;
	DOUBLE beta;

	memset(lambda, 0, numConstraints * sizeof(DOUBLE));
	memset(lambdaOld, 0, numConstraints * sizeof(DOUBLE));
	if (randomizationFlag == 1) {
		randperm(constraintPerm, numConstraints);
	} else {
		for (iterConstraint = 0; iterConstraint < numConstraints; \
												++iterConstraint) {
			constraintPerm[iterConstraint] = iterConstraint;
		}
	}
	iterConstraint = 0;
	epochCount = 0;
	while(1) {
		currConstraint = constraintPerm[iterConstraint];
		currInd1 = constraintMat[currConstraint];
		currInd2 = constraintMat[numConstraints + currConstraint];
		currX1 = &X[currInd1 * N];
		currX2 = &X[currInd2 * N];

		datacpy(vec, currX1, N);
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, currX2, &INCX, vec, &INCY);

		alpha = 1.0;
		beta = 0;
		SYMV(&UPLO, &SYMVN, &alpha, A, &SYMVLDA, vec, &INCX, &beta, Avec, &INCY);
		wtw = DOT(&DOTN, vec, &INCX, Avec, &INCY);

		gammaProj = gamma / (gamma + 1.0);
		currConstraintType = constraintMat[2 * numConstraints + currConstraint];

		if (currConstraintType == 1) {
			alphaParam = IMIN(lambda[currConstraint], \
					gammaProj * (1.0 / wtw -1.0 / boundVec[currConstraint]));
			lambda[currConstraint] = lambda[currConstraint] - alphaParam;
			betaParam = alphaParam / (1.0 - alphaParam * wtw);
			boundVec[currConstraint] = 1.0 / (1.0 / boundVec[currConstraint] \
											+ alphaParam / gamma);
		} else if (currConstraintType == - 1) {
			alphaParam = IMIN(lambda[currConstraint], gammaProj \
								* (1.0 / boundVec[currConstraint] - 1.0 / wtw));
			lambda[currConstraint] = lambda[currConstraint] - alphaParam;
			betaParam = - 1.0 * alphaParam / (1.0 + alphaParam * wtw);
			boundVec[currConstraint] = 1.0 / (1.0 / boundVec[currConstraint] \
											- alphaParam / gamma);
		} else {
			ERROR("Invalid constraint type.");
		}

		alpha = betaParam;
		SYR(&UPLO, &SYRN, &alpha, Avec, &INCX, A, &SYRLDA);

		++iterConstraint;
		if (iterConstraint % 1000 == 0) {
			PRINTF("Now running constraint %d of %d.\n", iterConstraint, numConstraints);
		}
		if (iterConstraint == numConstraints) {
			++epochCount;
			PRINTF("Finished epoch %d.\n", epochCount);
			if (epochCount == maxEpochs) {
				break;
			}

			normSum = NRM2(&NRM2N, lambda, &INCX) \
					+ NRM2(&NRM2N, lambdaOld, &INCX);
			if (normSum == 0) {
				break;
			}

			alpha = - 1.0;
			AXPY(&AXPYN2, &alpha, lambda, &INCX, lambdaOld, &INCY);
			normDiff = NRM2(&NRM2N, lambdaOld, &INCX);
			conv =  normDiff / normSum;
			if (conv < tolerance) {
				break;
			}
			datacpy(lambdaOld, lambda, numConstraints);
			iterConstraint = 0;
			if (randomizationFlag == 1) {
				randperm(constraintPerm, numConstraints);
			}
		}
	}

	INT iterM;
	INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		for (iterM = iterN + 1; iterM < N; ++iterM) {
			A[iterN * N + iterM] = A[iterM * N + iterN];
		}
	}

	if (constraintPermFlag == 1)  {
		FREE(constraintPerm);
	}
	if (lambdaFlag == 1)  {
		FREE(lambda);
	}
	if (lambdaOldFlag == 1)  {
		FREE(lambdaOld);
	}
	if (vecFlag == 1)  {
		FREE(vec);
	}
	if (AvecFlag == 1)  {
		FREE(Avec);
	}
}

void itkml(DOUBLE *A, DOUBLE *K, INT *constraintMat, DOUBLE *boundVec, \
		DOUBLE *Phi, INT factorizedFlag, DOUBLE tolerance, DOUBLE gamma, \
		INT maxEpochs, INT randomizationFlag, INT numSamples, \
		INT numConstraints, INT *constraintPerm, DOUBLE *lambda, \
		DOUBLE *lambdaOld, DOUBLE *vec) {

	INT PhiFlag = 0;
	if (Phi == NULL) {
		Phi = (DOUBLE *) MALLOC(numSamples * numSamples * sizeof(DOUBLE));
		PhiFlag = 1;
	}

	matcpy(A, K, numSamples, numSamples, 'A');
	itkl(A, constraintMat, boundVec, tolerance, gamma, maxEpochs, \
			randomizationFlag, numSamples, numConstraints, constraintPerm, \
			lambda, lambdaOld, vec);
	/* A = optimal solution Ko */

	INT AXPYN = SQR(numSamples);
	INT INCX = 1;
	double alpha = - 1.0;
	INT INCY = 1;
	AXPY(&AXPYN, &alpha, K, &INCX, A, &INCY);
	/* A = Ko - K */

	CHAR UPLO = 'U';
	if ((factorizedFlag == 0) && (PhiFlag == 1)) {
		INT POTRFN = numSamples;
		INT POTRFLDA = numSamples;
		INT INFO;
		matcpy(Phi, K, numSamples, numSamples, 'U');
		POTRF(&UPLO, &POTRFN, Phi, &POTRFLDA, &INFO);
	}

	CHAR trans = 'T';
	CHAR diag = 'N';
	INT TRSVN = numSamples;
	INT TRSVLDA = numSamples;
	INCX = 1;
	INT iterS;
	for (iterS = 0; iterS < numSamples; ++iterS) {
		TRSV(&UPLO, &trans, &diag, &TRSVN, Phi, &TRSVLDA, &A[iterS * numSamples], &INCX);
	}
	/* A = (Phi') ^ -1 * (Ko - K) */

	transpose_inplace(A, numSamples);
	/* A = ((Phi') ^ - 1 * (Ko - K))' = (Ko - K) * Phi ^ - 1*/

	for (iterS = 0; iterS < numSamples; ++iterS) {
		TRSV(&UPLO, &trans, &diag, &TRSVN, Phi, &TRSVLDA, &A[iterS * numSamples], &INCX);
		A[iterS * numSamples + iterS]  += 1.0;
	}
	/* A = I + (Phi') ^ - 1 * (Ko - K) * Phi ^ - 1 */

	if (PhiFlag == 1) {
		FREE(Phi);
	}
}

void legoUpdate(DOUBLE *A, DOUBLE trueDist, DOUBLE eta, \
			DOUBLE *x1, DOUBLE *x2, INT N, DOUBLE *vec, DOUBLE *Avec) {

	INT tempVec1Flag = 0;
	if (vec == NULL) {
		vec = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		tempVec1Flag = 1;
	}

	INT tempVec2Flag = 0;
	if (Avec == NULL) {
		Avec = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		tempVec2Flag = 1;
	}

	datacpy(vec, x1, N);
	INT INCX = 1;
	INT INCY = 1;
	DOUBLE alpha = - 1.0;

	AXPY(&N, &alpha, x2, &INCX, vec, &INCY);

	CHAR UPLO = 'U';
	INT SYMVN = N;
	alpha = 1.0;
	INT SYMVLDA = N;
	DOUBLE beta = 0.0;

	SYMV(&UPLO, &SYMVN, &alpha, A, &SYMVLDA, vec, &INCX, &beta, Avec, &INCY);

	INT DOTN = N;
	DOUBLE predDist = DOT(&DOTN, Avec, &INCX, vec, &INCY);
	DOUBLE futDist = (eta * trueDist * predDist - 1 \
		+ sqrt(SQR(eta * trueDist * predDist - 1) + 4 * eta * SQR(predDist))) \
		/ (2 * eta * predDist);

	alpha = - (eta * (futDist - trueDist)) \
			/ (1 + eta * (futDist - trueDist) * predDist);

	INT SYRN = N;
	INT SYRLDA = N;
	SYR(&UPLO, &SYRN, &alpha, Avec, &INCX, A, &SYRLDA);

	if (tempVec1Flag == 1)  {
		FREE(vec);
	}
	if (tempVec2Flag == 1)  {
		FREE(Avec);
	}
}

void lego(DOUBLE *A, DOUBLE *X, INT *pairMat, DOUBLE *distVec, \
		DOUBLE tolerance, DOUBLE eta0, INT maxEpochs, INT randomizationFlag, \
		INT continuationFlag, INT N, INT numPairs, DOUBLE *AOld, INT *pairPerm, \
		DOUBLE *vec, DOUBLE *Avec) {

	INT AOldFlag = 0;
	if (AOld == NULL) {
		AOld = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		AOldFlag = 1;
	}

	INT pairPermFlag = 0;
	if (pairPerm == NULL) {
		pairPerm = (INT *) MALLOC(numPairs * 1 * sizeof(INT));
		pairPermFlag = 1;
	}

	INT vecFlag = 0;
	if (vec == NULL) {
		vec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vecFlag = 1;
	}

	INT AvecFlag = 0;
	if (Avec == NULL) {
		Avec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		AvecFlag = 1;
	}

	INT currPair;
	INT iterPair;
	INT epochCount;
	INT currInd1;
	INT currInd2;
	DOUBLE *currX1;
	DOUBLE *currX2;
	DOUBLE eta;
	DOUBLE normDiff;
	DOUBLE normOld;
	DOUBLE conv;

	CHAR UPLO = 'U';
	CHAR norm = 'F';
	INT INCX = 1;
	INT INCY = 1;
	INT AXPYN2 = N * N;
	INT LANSYN = N;
	INT LANSYLDA = N;
	DOUBLE alpha = - 1.0;

	if (randomizationFlag == 1) {
		randperm(pairPerm, numPairs);
	} else {
		for (iterPair = 0; iterPair < numPairs; ++iterPair) {
			pairPerm[iterPair] = iterPair;
		}
	}
	iterPair = 0;
	epochCount = 0;
	matcpy(AOld, A, N, N, 'U');
	while(1) {
		currPair = pairPerm[iterPair];
		currInd1 = pairMat[currPair];
		currInd2 = pairMat[numPairs + currPair];
		currX1 = &X[currInd1 * N];
		currX2 = &X[currInd2 * N];
		if (continuationFlag == 1) {
			eta = eta0 / sqrt((DOUBLE) (epochCount * numPairs + iterPair + 1));
		} else {
			eta = eta0;
		}
		legoUpdate(A, distVec[currPair], eta, currX1, currX2, N, vec, Avec);

		++iterPair;
		if (iterPair == numPairs) {
			++epochCount;
			PRINTF("Finished epoch %d.\n", epochCount);
			if (epochCount > maxEpochs) {
				break;
			}

			normOld = LANSY(&norm, &UPLO, &LANSYN, AOld, &LANSYLDA, NULL);
			AXPY(&AXPYN2, &alpha, A, &INCX, AOld, &INCY);
			normDiff = LANSY(&norm, &UPLO, &LANSYN, AOld, &LANSYLDA, NULL);
			conv = normDiff / normOld;
			if (conv < tolerance) {
				break;
			}

			matcpy(AOld, A, N, N, 'U');
			iterPair = 0;
			if (randomizationFlag == 1) {
				randperm(pairPerm, numPairs);
			}
		}
	}

	INT iterM;
	INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		for (iterM = iterN + 1; iterM < N; ++iterM) {
			A[iterN * N + iterM] = A[iterM * N + iterN];
		}
	}

	if (AOldFlag == 1)  {
		FREE(AOld);
	}
	if (pairPermFlag == 1)  {
		FREE(pairPerm);
	}
	if (vecFlag == 1)  {
		FREE(vec);
	}
	if (AvecFlag == 1)  {
		FREE(Avec);
	}
}

// TODO: Find better initialization than setting everything to zero.
// TODO: Write version for partial eig/svd, and add largeScaleFlag.
// TODO: Write version with lineSearch.
// TODO: Write kernel version, perhaps in conjunction with MDS.
// TODO: Parallelize gradient.
// TODO: Write initialization for tau.
// TODO: Write kernel-metric learning versions.
// TODO: Write code to convert between metric and kernel learning versions.
// TODO: Add weights as in kernel_learning.
void nrml_fpc_continuation(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE tau, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, DOUBLE tauMultiplier, \
			DOUBLE tauRate, INT N, INT numConstraints) {

	DOUBLE *AOld = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *AOldVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *vec1 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *vec2 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * N * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	nuclear_psd_proximal(A, NULL, tau, N, vec1, Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	memset((void *) A, 0, SQR(N) * sizeof(DOUBLE));
	nrml_fp_inner(A, X, problemType, constraintMat, betaVec, tau, delta, \
			numIters, tolerance, tauMultiplier, tauRate, N, numConstraints, \
			AOld, AOldVec, vec1, vec2, Vr, isuppz, work, lwork);

	FREE(AOld);
	FREE(AOldVec);
	FREE(vec1);
	FREE(vec2);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void nrml_fpc(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE* tau, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, INT N, INT numConstraints, \
			INT numRepeats) {

	DOUBLE *AOld = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *AOldVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *vec1 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *vec2 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * N * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	nuclear_psd_proximal(A, NULL, tau[0], N, vec1, Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	memset((void *) A, 0, SQR(N) * sizeof(DOUBLE));
	INT iterRepeat;
//	printf("igkiou\n");
	for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
		printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, numRepeats, tau[iterRepeat]);
		nrml_fp_inner(A, X, problemType, constraintMat, betaVec, \
			tau[iterRepeat], delta, numIters, tolerance, 1, 1, N, \
			numConstraints, AOld, AOldVec, vec1, vec2, Vr, isuppz, work, lwork);
	}

	FREE(AOld);
	FREE(AOldVec);
	FREE(vec1);
	FREE(vec2);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void nrml_fp_inner(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE tauTarget, \
			DOUBLE delta, INT numIters, DOUBLE tolerance, DOUBLE tauMultiplier, \
			DOUBLE tauRate, INT N, INT numConstraints, DOUBLE *AOld, \
			DOUBLE *AOldVec, DOUBLE *vec1, DOUBLE *vec2, DOUBLE *Vr, \
			INT *isuppz, DOUBLE *work, INT lwork) {

	INT AOldFlag = 0;
	if (AOld == NULL) {
		AOld = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
		AOldFlag = 1;
	}

	INT AOldVecFlag = 0;
	if (AOldVec == NULL) {
		AOldVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		AOldVecFlag = 1;
	}

	INT vec1Flag = 0;
	if (vec1 == NULL) {
		vec1 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vec1Flag = 1;
	}

	INT vec2Flag = 0;
	if (vec2 == NULL) {
		vec2 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vec2Flag = 1;
	}

	INT VrFlag = 0;
	if (Vr == NULL) {
		Vr = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		VrFlag = 1;
	}

	INT isuppzFlag = 0;
	if (isuppz == NULL) {
		isuppz = (INT *) MALLOC(2 * N * sizeof(INT));
		isuppzFlag = 1;
	}

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	DOUBLE normA;
	DOUBLE normADiff;
	DOUBLE objtemp;

	CHAR UPLO = 'U';
	CHAR LANSYNORM = 'F';
	INT LANSYN = N;
	INT LANSYLDA = N;
	INT AXPBYN = SQR(N);
	INT AXPYN = SQR(N);
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;

	INT workFlag = 0;
	if (lwork == -1) {
		DOUBLE work_temp;
		nuclear_psd_proximal(A, NULL, tauTarget, N, vec1, \
				Vr, &work_temp, lwork);
		lwork = (INT) work_temp;
		work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));
		workFlag = 1;
	}

	DOUBLE tau = tauMultiplier * tauTarget;
	INT iter = 0;
	while (1) {
		++iter;
		if (iter % 1000 == 0) {
			printf("Now running iter %d\n", iter);
		}

		datacpy(AOld, A, N * N);
//		normAOld = LANSY(&LANSYNORM, &UPLO, &LANSYN, AOld, &LANSYLDA, NULL);

		ml_obj_grad(&objtemp, A, AOld, X, problemType, constraintMat, betaVec, \
								N, numConstraints, AOldVec, vec1, vec2);

		alpha = 1.0;
		beta = - delta;
		AXPBY(&AXPBYN, &alpha, AOld, &INCX, &beta, A, &INCY);
		datacpy(AOld, A, N * N);

		nuclear_psd_proximal(A, NULL, tau * delta, N, vec1, Vr, work, lwork);
		tau = IMAX(tauRate * tau, tauTarget);

		if (iter == numIters) {
			break;
		}

		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, A, &INCX, AOld, &INCY);
		normADiff = LANSY(&LANSYNORM, &UPLO, &LANSYN, AOld, &LANSYLDA, NULL);
		normA = LANSY(&LANSYNORM, &UPLO, &LANSYN, A, &LANSYLDA, NULL);

		if ((normA > eps) && (normADiff / IMAX(1.0, normA) < tolerance)) {
			break;
		}
	}

	if (AOldFlag == 1) {
		FREE(AOld);
	}

	if (AOldVecFlag == 1) {
		FREE(AOldVec);
	}

	if (vec1Flag == 1) {
		FREE(vec1);
	}

	if (vec2Flag == 1) {
		FREE(vec2);
	}

	if (VrFlag == 1) {
		FREE(Vr);
	}

	if (isuppzFlag == 1) {
		FREE(isuppz);
	}

	if (workFlag == 1) {
		FREE(work);
	}
}

void nrml_apg_continuation(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE tau, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, \
			DOUBLE tauMultiplier, DOUBLE tauRate, INT N, INT numConstraints) {

	DOUBLE *AOld = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *AOldVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *L = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *LfGrad = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *LfGD = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *ALfGDDiff = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *vec1 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *vec2 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * N * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	nuclear_psd_proximal(A, NULL, tau, N, vec1, Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	memset((void *) A, 0, SQR(N) * sizeof(DOUBLE));
	nrml_apg_inner(A, X, problemType, constraintMat, betaVec, tau, delta0, \
			numIters, tolerance, lineSearchFlag, eta, tauMultiplier, tauRate, \
			N, numConstraints, AOld, AOldVec, L, LfGrad, LfGD, ALfGDDiff, vec1, \
			vec2, Vr, isuppz, work, lwork);

	FREE(AOld);
	FREE(AOldVec);
	FREE(L);
	FREE(LfGrad);
	FREE(LfGD);
	FREE(ALfGDDiff);
	FREE(vec1);
	FREE(vec2);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void nrml_apg(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE *tau, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, \
			INT N, INT numConstraints, INT numRepeats) {

	DOUBLE *AOld = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *AOldVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *L = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *LfGrad = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *LfGD = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *ALfGDDiff = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *vec1 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *vec2 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * N * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	nuclear_psd_proximal(A, NULL, tau[0], N, vec1, \
				Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	memset((void *) A, 0, SQR(N) * sizeof(DOUBLE));
	DOUBLE delta;
	INT iterRepeat;
	for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
		printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, numRepeats, tau[iterRepeat]);
		delta = delta0;
		nrml_apg_inner(A, X, problemType, constraintMat, betaVec, \
				tau[iterRepeat], delta, numIters, tolerance, lineSearchFlag, \
				eta, 1, 1, N, numConstraints, AOld, AOldVec, L, LfGrad, LfGD, \
				ALfGDDiff, vec1, vec2, Vr, isuppz, work, lwork);
	}

	FREE(AOld);
	FREE(AOldVec);
	FREE(L);
	FREE(LfGrad);
	FREE(LfGD);
	FREE(ALfGDDiff);
	FREE(vec1);
	FREE(vec2);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void nrml_apg_inner(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE tauTarget, \
			DOUBLE delta0, INT numIters, DOUBLE tolerance, INT lineSearchFlag, \
			DOUBLE eta, DOUBLE tauMultiplier, DOUBLE tauRate, INT N, \
			INT numConstraints, DOUBLE *AOld, DOUBLE *AOldVec, DOUBLE *L, \
			DOUBLE *LfGrad, DOUBLE *LfGD, DOUBLE *ALfGDDiff, DOUBLE *vec1, \
			DOUBLE *vec2, DOUBLE *Vr, INT *isuppz, DOUBLE *work, INT lwork) {

	INT AOldFlag = 0;
	if (AOld == NULL) {
		AOld = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
		AOldFlag = 1;
	}

	INT AOldVecFlag = 0;
	if (AOldVec == NULL) {
		AOldVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		AOldVecFlag = 1;
	}

	INT LFlag = 0;
	if (L == NULL) {
		L = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
		LFlag = 1;
	}

	INT LfGradFlag = 0;
	if (LfGrad == NULL) {
		LfGrad = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
		LfGradFlag = 1;
	}

	INT LfGDFlag = 0;
	if (LfGD == NULL) {
		LfGD = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
		LfGDFlag = 1;
	}

	INT ALfGDDiffFlag = 0;
	if (ALfGDDiff == NULL) {
		ALfGDDiff = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
		ALfGDDiffFlag = 1;
	}

	INT vec1Flag = 0;
	if (vec1 == NULL) {
		vec1 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vec1Flag = 1;
	}

	INT vec2Flag = 0;
	if (vec2 == NULL) {
		vec2 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vec2Flag = 1;
	}

	INT VrFlag = 0;
	if (Vr == NULL) {
		Vr = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		VrFlag = 1;
	}

	INT isuppzFlag = 0;
	if (isuppz == NULL) {
		isuppz = (INT *) MALLOC(2 * N * sizeof(INT));
		isuppzFlag = 1;
	}

	INT workFlag = 0;
	if (lwork == -1) {
		DOUBLE work_temp;
		nuclear_psd_proximal(A, NULL, tauTarget, N, vec1, \
				Vr, &work_temp, lwork);
		lwork = (INT) work_temp;
		work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));
		workFlag = 1;
	}

	INT AXPBYN = SQR(N);
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;
	CHAR LANSYNORM = 'F';
	CHAR UPLO = 'U';
	INT LANSYN = N;
	INT LANSYLDA = N;
	INT AXPYN = SQR(N);

	DOUBLE t = 1;
	DOUBLE tOld = 1;
	DOUBLE LfObj;
	DOUBLE delta = delta0;
	DOUBLE normADiff;
//	DOUBLE normAOld;
	DOUBLE normA;

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	INT iter = 0;
	DOUBLE tau = tauMultiplier * tauTarget;
	while (1) {
		++iter;
		if (iter % 1000 == 0) {
			printf("Now running iter %d\n", iter);
		}

		datacpy(L, A, SQR(N));
		alpha = - (tOld - 1.0) / t;
		beta = 1 - alpha;
 		AXPBY(&AXPBYN, &alpha, AOld, &INCX, &beta, L, &INCY);
 		datacpy(AOld, A, SQR(N));
 		ml_obj_grad(&LfObj, LfGrad, L, X, problemType, constraintMat, betaVec, \
 				N, numConstraints, AOldVec, vec1, vec2);

 		nrml_line_search(&delta, A, L, LfObj, LfGrad, tau, lineSearchFlag, \
 				eta, X, problemType, constraintMat, betaVec, N, numConstraints, \
 				LfGD, ALfGDDiff, AOldVec, vec1, vec2, Vr, work, lwork);
		tOld = t;
		t = (1 + SQRT(1 + 4 * SQR(tOld))) * 0.5;
		tau = IMAX(tauRate * tau, tauTarget);

		if (iter == numIters) {
			break;
		}

		datacpy(ALfGDDiff, LfGrad, SQR(N));
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, A, &INCX, ALfGDDiff, &INCY);
		normADiff = LANSY(&LANSYNORM, &UPLO, &LANSYN, ALfGDDiff, &LANSYLDA, NULL);
		normA = LANSY(&LANSYNORM, &UPLO, &LANSYN, A, &LANSYLDA, NULL);

		if ((normA > eps) && (normADiff / IMAX(1.0, normA) < tolerance)) {
			break;
		}

//		datacpy(ALfGDDiff, A, SQR(N));
//		alpha = - 1.0;
//		AXPY(&AXPYN, &alpha, AOld, &INCX, ALfGDDiff, &INCY);
//		normADiff = LANSY(&LANSYNORM, &UPLO, &LANSYN, ALfGDDiff, &LANSYLDA, NULL);
//		normAOld = LANSY(&LANSYNORM, &UPLO, &LANSYN, AOld, &LANSYLDA, NULL);
//		normA = LANSY(&LANSYNORM, &UPLO, &LANSYN, A, &LANSYLDA, NULL);

//		if ((normA > eps) && (normADiff / IMAX(1.0, normAOld) < tolerance)) {
//			break;
//		}
	}

	if (AOldFlag == 1) {
		FREE(AOld);
	}

	if (AOldVecFlag == 1) {
		FREE(AOldVec);
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

	if (vec1Flag == 1) {
		FREE(vec1);
	}

	if (vec2Flag == 1) {
		FREE(vec2);
	}

	if (VrFlag == 1) {
		FREE(Vr);
	}

	if (isuppzFlag == 1) {
		FREE(isuppz);
	}

	if (workFlag == 1) {
		FREE(work);
	}
}

void nrml_line_search(DOUBLE *delta, DOUBLE *LfGDShrink, DOUBLE *L, \
		DOUBLE LfObj, DOUBLE *LfGrad, DOUBLE tau, INT lineSearchFlag, \
		DOUBLE eta, DOUBLE *X, DIST_LABEL_TYPE problemType, INT *constraintMat, \
		DOUBLE *betaVec, INT N, INT numConstraints, DOUBLE *LfGD, \
		DOUBLE *ALfGDDiff, DOUBLE *AOldVec, DOUBLE *vec1, DOUBLE *vec2, \
		DOUBLE *Vr, DOUBLE *work, INT lwork) {

	INT ALfGDDiffFlag = 0;
	if (ALfGDDiff == NULL) {
		ALfGDDiff = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
		ALfGDDiffFlag = 1;
	}

	INT AOldVecFlag = 0;
	if (AOldVec == NULL) {
		AOldVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		AOldVecFlag = 1;
	}

	INT vec1Flag = 0;
	if (vec1 == NULL) {
		vec1 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vec1Flag = 1;
	}

	INT vec2Flag = 0;
	if (vec2 == NULL) {
		vec2 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vec2Flag = 1;
	}

	INT VrFlag = 0;
	if (Vr == NULL) {
		Vr = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		VrFlag = 1;
	}

	INT AXPYN = SQR(N);
	DOUBLE alpha;
	INT INCX = 1;
	INT INCY = 1;

	DOUBLE LfGDShrinkfObj;
	DOUBLE LQmodObj;
	DOUBLE deltatemp = *delta;
	while (1) {
		datacpy(LfGD, L, SQR(N));
		alpha = - 1.0 / deltatemp;
		AXPY(&AXPYN, &alpha, LfGrad, &INCX, LfGD, &INCY);
		nrml_shrinkage(LfGDShrink, NULL, LfGD, tau / deltatemp, N, vec1, \
						Vr, work, lwork);
		if (lineSearchFlag == 0) {
			break;
		}
		ml_obj_grad(&LfGDShrinkfObj, NULL, LfGDShrink, X, problemType, \
				constraintMat, betaVec, N, numConstraints, AOldVec, vec1, vec2);
		nrml_Q_func_mod(&LQmodObj, LfGDShrink, LfGD, LfObj, LfGrad, \
				deltatemp, N, ALfGDDiff);
		if (LfGDShrinkfObj <= LQmodObj) {
			break;
		}
		deltatemp = deltatemp * eta;
	}

	*delta = deltatemp;
	if (ALfGDDiffFlag == 1) {
		FREE(ALfGDDiff);
	}

	if (AOldVecFlag == 1) {
		FREE(AOldVec);
	}

	if (vec1Flag == 1) {
		FREE(vec1);
	}

	if (vec2Flag == 1) {
		FREE(vec2);
	}

	if (VrFlag == 1) {
		FREE(Vr);
	}
}

void nrml_shrinkage(DOUBLE *AShrink, DOUBLE *AShrinkNorm, DOUBLE *A, \
				DOUBLE tau, INT N, DOUBLE *lVec, DOUBLE *Vr, DOUBLE *work, \
				INT lwork) {

	INT lVecFlag = 0;
	if (lVec == NULL) {
		lVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		lVecFlag = 1;
	}

	INT VrFlag = 0;
	if (Vr == NULL) {
		Vr = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		VrFlag = 1;
	}

	datacpy(AShrink, A, SQR(N));
	// TODO: Implement truncation.
	nuclear_psd_proximal(AShrink, AShrinkNorm, tau, N, lVec, \
				Vr, work, lwork);

	if (lVecFlag == 1) {
		FREE(lVec);
	}

	if (VrFlag == 1) {
		FREE(Vr);
	}
}

void nrml_Q_func_mod(DOUBLE *QObj, DOUBLE *A, DOUBLE *LfGD, DOUBLE LfObj, \
			DOUBLE *LfGrad, DOUBLE delta, INT N, DOUBLE *ALfGDDiff) {

	INT ALfGDDiffFlag = 0;
	if (ALfGDDiff == NULL) {
		ALfGDDiff = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
		ALfGDDiffFlag = 1;
	}

	datacpy(ALfGDDiff, A, SQR(N));
	INT AXPYN = SQR(N);
	DOUBLE alpha = - 1.0;
	INT INCX = 1;
	INT INCY = 1;
	AXPY(&AXPYN, &alpha, LfGD, &INCX, ALfGDDiff, &INCY);

	CHAR NORM = 'F';
	CHAR UPLO = 'U';
	INT LANSYN = N;
	INT LANSYLDA = N;
	DOUBLE *work = NULL;
	DOUBLE ALfGDDiffNorm = LANSY(&NORM, &UPLO, &LANSYN, ALfGDDiff, &LANSYLDA, \
								work);
	DOUBLE LfGradNorm = LANSY(&NORM, &UPLO, &LANSYN, LfGrad, &LANSYLDA, \
								work);
	*QObj = delta * 0.5 * SQR(ALfGDDiffNorm) - 0.5 / delta * SQR(LfGradNorm) \
			+ LfObj;

	if (ALfGDDiffFlag == 1) {
		FREE(ALfGDDiff);
	}
}

//void nrml_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, DOUBLE *X, \
//			INT *constraintMat, DOUBLE *betaVec, INT N, INT numConstraints, \
//			DOUBLE *Avec, DOUBLE *vec) {
//
//	INT AvecFlag = 0;
//	if (Avec == NULL) {
//		Avec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
//		AvecFlag = 1;
//	}
//
//	INT vecFlag = 0;
//	if (vec == NULL) {
//		vec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
//		vecFlag = 1;
//	}
//
//	INT coord1;
//	INT coord2;
//	DOUBLE *X1;
//	DOUBLE *X2;
//	DOUBLE bound;
//	DOUBLE wtw;
//
//	CHAR UPLO = 'U';
//	INT SYMVN = N;
//	INT SYMVLDA = N;
//	INT SYRN = N;
//	INT SYRLDA = N;
//	INT DOTN = N;
//	INT AXPYN = SQR(N);
//	DOUBLE alpha;
//	DOUBLE beta;
//	INT INCX = 1;
//	INT INCY = 1;
//
//	DOUBLE objtemp = 0;
//	if (grad != NULL) {
//		memset((void *) grad, 0, SQR(N) * sizeof(DOUBLE));
//	}
//
//	INT iterConstraint;
//	for (iterConstraint = 0; iterConstraint < numConstraints; \
//				++iterConstraint) {
//		bound = betaVec[iterConstraint];
//		/* i */
//		coord1 = constraintMat[iterConstraint];
//		/* j */
//		coord2 = constraintMat[numConstraints + iterConstraint];
//
//		X1 = &X[coord1 * N];
//		X2 = &X[coord2 * N];
//		datacpy(vec, X1, N);
//		alpha = - 1.0;
//		AXPY(&AXPYN, &alpha, X2, &INCX, vec, &INCY);
//
//		alpha = 1.0;
//		beta = 0;
//		SYMV(&UPLO, &SYMVN, &alpha, A, &SYMVLDA, vec, &INCX, &beta, Avec, &INCY);
//		wtw = DOT(&DOTN, vec, &INCX, Avec, &INCY);
//		objtemp += 0.5 * SQR(wtw - bound);
//
//		if (grad != NULL) {
//			alpha = wtw - bound;
//			SYR(&UPLO, &SYRN, &alpha, vec, &INCX, grad, &SYRLDA);
//		}
//	}
//	*obj = objtemp;
//
//	if (AvecFlag == 1) {
//		FREE(Avec);
//	}
//
//	if (vecFlag == 1) {
//		FREE(vec);
//	}
//}

void frml_pgd(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE kappa, \
			DOUBLE delta0, INT numIters, DOUBLE tolerance, INT stepFlag, INT N, \
			INT numConstraints) {

	DOUBLE *AOld = (DOUBLE *) MALLOC(SQR(N) * sizeof(DOUBLE));
	DOUBLE *AOldVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *vec1 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *vec2 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * N * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	matrix_psd_projection(A, N, vec1, Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	DOUBLE normA;
	DOUBLE normGrad;
	DOUBLE objtemp;

	CHAR UPLO = 'U';
	CHAR LANSYNORM = 'F';
	INT LANSYN = N;
	INT LANSYLDA = N;
	INT AXPBYN = SQR(N);
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;

	INT iter = 0;
	DOUBLE delta = delta0;
	while (1) {
		++iter;
		if (iter % 1000 == 0) {
			printf("Now running iter %d\n", iter);
		}

		if (stepFlag == 1) {
			delta = delta0 / SQRT((DOUBLE) iter);
		}

		datacpy(AOld, A, N * N);

		frml_obj_grad(&objtemp, A, AOld, X, problemType, constraintMat, betaVec, \
						kappa, N, numConstraints, AOldVec, vec1, vec2);

		alpha = 1.0;
		beta = - delta;
		AXPBY(&AXPBYN, &alpha, AOld, &INCX, &beta, A, &INCY);
		normGrad = LANSY(&LANSYNORM, &UPLO, &LANSYN, A, &LANSYLDA, NULL);

		matrix_psd_projection(A, N, vec1, Vr, work, lwork);

		if (iter == numIters) {
			break;
		}

		normA = LANSY(&LANSYNORM, &UPLO, &LANSYN, A, &LANSYLDA, NULL);

		if ((normA > eps) && (normGrad / IMAX(1.0, normA) < tolerance)) {
			break;
		}
	}

	FREE(AOld);
	FREE(AOldVec);
	FREE(vec1);
	FREE(vec2);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void frml_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, DOUBLE *X, \
		DIST_LABEL_TYPE problemType, INT *constraintMat, DOUBLE *betaVec, \
		DOUBLE kappa, INT N, INT numConstraints, DOUBLE *Avec, DOUBLE *vec1, \
		DOUBLE *vec2) {

	ml_obj_grad(obj, grad, A, X, problemType, constraintMat, betaVec, N, \
				numConstraints, Avec, vec1, vec2);

	CHAR LANSYNORM = 'F';
	CHAR UPLO = 'U';
	INT LANSYN = N;
	INT LANSYLDA = N;
	DOUBLE normA = LANSY(&LANSYNORM, &UPLO, &LANSYN, A, &LANSYLDA, NULL);
	*obj += SQR(normA) * kappa / 2.0;
	if (grad != NULL) {
		INT AXPYN = SQR(N);
		DOUBLE alpha = kappa;
		INT incx = 1;
		INT incy = 1;
		AXPY(&AXPYN, &alpha, A, &incx, grad, &incy);
	}
}

// TODO: Add control for NULL obj or grad here.
// TODO: Add 1 / numConstraints factor to m-file versions.
void ml_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, DOUBLE *X, \
			DIST_LABEL_TYPE problemType, INT *constraintMat, DOUBLE *betaVec, \
			INT N, INT numConstraints, DOUBLE *Avec, DOUBLE *vec1, \
			DOUBLE *vec2) {

	if (problemType == DIST_LABEL_TARGETS) {
		ml_target_obj_grad(obj, grad, A, X, constraintMat, betaVec, N, \
						numConstraints, Avec, vec1);
	} else if (problemType == DIST_LABEL_RELATIONAL) {
		ml_relational_obj_grad(obj, grad, A, X, constraintMat, betaVec, N, \
						numConstraints, Avec, vec1, vec2);
	} else if (problemType == DIST_LABEL_BOUNDS) {
		ml_bound_obj_grad(obj, grad, A, X, constraintMat, betaVec, N, \
						numConstraints, Avec, vec1);
	} else if (problemType == DIST_LABEL_SQRHINGE) {
		ml_sqrhinge_obj_grad(obj, grad, A, X, constraintMat, betaVec, N, \
						numConstraints, Avec, vec1, vec2);
	} else if (problemType == DIST_LABEL_HUBERHINGE) {
		ERROR("ml_obj_grad function not implemented for DIST_LABEL_HUBERHINGE case yet.");
	}
}

void ml_target_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, DOUBLE *X, \
			INT *constraintMat, DOUBLE *targetVec, INT N, INT numConstraints, \
			DOUBLE *Avec, DOUBLE *vec) {

	INT AvecFlag = 0;
	if (Avec == NULL) {
		Avec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		AvecFlag = 1;
	}

	INT vecFlag = 0;
	if (vec == NULL) {
		vec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vecFlag = 1;
	}

	INT coordi;
	INT coordj;
	DOUBLE *Xi;
	DOUBLE *Xj;
	DOUBLE target;
	DOUBLE dij;

	CHAR UPLO = 'U';
	INT SYMVN = N;
	INT SYMVLDA = N;
	INT SYRN = N;
	INT SYRLDA = N;
	INT DOTN = N;
	INT AXPYN = N;
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;

	DOUBLE objtemp = 0;
	if (grad != NULL) {
		memset((void *) grad, 0, SQR(N) * sizeof(DOUBLE));
	}

	INT iterConstraint;
	for (iterConstraint = 0; iterConstraint < numConstraints; \
				++iterConstraint) {
		/* target */
		target = targetVec[iterConstraint];
		/* i */
		coordi = constraintMat[iterConstraint];
		/* j */
		coordj = constraintMat[numConstraints + iterConstraint];

		Xi = &X[coordi * N];
		Xj = &X[coordj * N];
		datacpy(vec, Xi, N);
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, Xj, &INCX, vec, &INCY);

		alpha = 1.0;
		beta = 0;
		SYMV(&UPLO, &SYMVN, &alpha, A, &SYMVLDA, vec, &INCX, &beta, Avec, \
			&INCY);
		dij = DOT(&DOTN, vec, &INCX, Avec, &INCY);
		objtemp += 0.5 * SQR(dij - target);

		if (grad != NULL) {
			alpha = dij - target;
			SYR(&UPLO, &SYRN, &alpha, vec, &INCX, grad, &SYRLDA);
		}
	}
	*obj = objtemp / (DOUBLE) numConstraints;
	if (grad != NULL) {
		INT SCALN = SQR(N);
		alpha = 1.0 / (DOUBLE) numConstraints;
		INT incx = 1;
		SCAL(&SCALN, &alpha, grad, &incx);
	}

	if (AvecFlag == 1) {
		FREE(Avec);
	}

	if (vecFlag == 1) {
		FREE(vec);
	}
}

void ml_relational_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, \
			DOUBLE *X, INT *constraintMat, DOUBLE *marginVec, INT N, \
			INT numConstraints, DOUBLE *Avec, DOUBLE *vecij, DOUBLE *veckl) {

	INT AvecFlag = 0;
	if (Avec == NULL) {
		Avec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		AvecFlag = 1;
	}

	INT vecijFlag = 0;
	if (vecij == NULL) {
		vecij = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vecijFlag = 1;
	}

	INT vecklFlag = 0;
	if (veckl == NULL) {
		veckl = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vecklFlag = 1;
	}

	INT coordi;
	INT coordj;
	INT coordk;
	INT coordl;
	DOUBLE *Xi;
	DOUBLE *Xj;
	DOUBLE *Xk;
	DOUBLE *Xl;
	DOUBLE margin;
	DOUBLE dij;
	DOUBLE dkl;
	DOUBLE viol;

	CHAR UPLO = 'U';
	INT SYMVN = N;
	INT SYMVLDA = N;
	INT SYRN = N;
	INT SYRLDA = N;
	INT DOTN = N;
	INT AXPYN = N;
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;

	DOUBLE objtemp = 0;
	if (grad != NULL) {
		memset((void *) grad, 0, SQR(N) * sizeof(DOUBLE));
	}

	INT iterConstraint;
	for (iterConstraint = 0; iterConstraint < numConstraints; \
				++iterConstraint) {
		/* margin */
		margin = marginVec[iterConstraint];
		/* i */
		coordi = constraintMat[iterConstraint];
		/* j */
		coordj = constraintMat[numConstraints + iterConstraint];
		/* k */
		coordk = constraintMat[numConstraints * 2 + iterConstraint];
		/* l */
		coordl = constraintMat[numConstraints * 3 + iterConstraint];

		/* Get X vectors */
		Xi = &X[coordi * N];
		Xj = &X[coordj * N];
		Xk = &X[coordk * N];
		Xl = &X[coordl * N];

		/* Find dij */
		datacpy(vecij, Xi, N);
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, Xj, &INCX, vecij, &INCY);

		alpha = 1.0;
		beta = 0;
		SYMV(&UPLO, &SYMVN, &alpha, A, &SYMVLDA, vecij, &INCX, &beta, Avec, &INCY);
		dij = DOT(&DOTN, vecij, &INCX, Avec, &INCY);

		/* Find dkl */
		datacpy(veckl, Xk, N);
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, Xl, &INCX, veckl, &INCY);

		alpha = 1.0;
		beta = 0;
		SYMV(&UPLO, &SYMVN, &alpha, A, &SYMVLDA, veckl, &INCX, &beta, Avec, &INCY);
		dkl = DOT(&DOTN, veckl, &INCX, Avec, &INCY);

		viol = dij - dkl - margin;
		if (viol > 0) {
			objtemp += viol;
			if (grad != NULL) {
				alpha = 1.0;
				SYR(&UPLO, &SYRN, &alpha, vecij, &INCX, grad, &SYRLDA);

				alpha = - 1.0;
				SYR(&UPLO, &SYRN, &alpha, veckl, &INCX, grad, &SYRLDA);
			}
		}
	}
	*obj = objtemp / (DOUBLE) numConstraints;
	if (grad != NULL) {
		INT SCALN = SQR(N);
		alpha = 1.0 / (DOUBLE) numConstraints;
		INT incx = 1;
		SCAL(&SCALN, &alpha, grad, &incx);
	}

	if (AvecFlag == 1) {
		FREE(Avec);
	}

	if (vecijFlag == 1) {
		FREE(vecij);
	}

	if (vecklFlag == 1) {
		FREE(veckl);
	}
}

void ml_sqrhinge_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, \
			DOUBLE *X, INT *constraintMat, DOUBLE *marginVec, INT N, \
			INT numConstraints, DOUBLE *Avec, DOUBLE *vecij, DOUBLE *veckl) {

	INT AvecFlag = 0;
	if (Avec == NULL) {
		Avec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		AvecFlag = 1;
	}

	INT vecijFlag = 0;
	if (vecij == NULL) {
		vecij = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vecijFlag = 1;
	}

	INT vecklFlag = 0;
	if (veckl == NULL) {
		veckl = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vecklFlag = 1;
	}

	INT coordi;
	INT coordj;
	INT coordk;
	INT coordl;
	DOUBLE *Xi;
	DOUBLE *Xj;
	DOUBLE *Xk;
	DOUBLE *Xl;
	DOUBLE margin;
	DOUBLE dij;
	DOUBLE dkl;
	DOUBLE viol;

	CHAR UPLO = 'U';
	INT SYMVN = N;
	INT SYMVLDA = N;
	INT SYRN = N;
	INT SYRLDA = N;
	INT DOTN = N;
	INT AXPYN = N;
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;

	DOUBLE objtemp = 0;
	if (grad != NULL) {
		memset((void *) grad, 0, SQR(N) * sizeof(DOUBLE));
	}

	INT iterConstraint;
	for (iterConstraint = 0; iterConstraint < numConstraints; \
				++iterConstraint) {
		/* margin */
		margin = marginVec[iterConstraint];
		/* i */
		coordi = constraintMat[iterConstraint];
		/* j */
		coordj = constraintMat[numConstraints + iterConstraint];
		/* k */
		coordk = constraintMat[numConstraints * 2 + iterConstraint];
		/* l */
		coordl = constraintMat[numConstraints * 3 + iterConstraint];

		/* Get X vectors */
		Xi = &X[coordi * N];
		Xj = &X[coordj * N];
		Xk = &X[coordk * N];
		Xl = &X[coordl * N];

		/* Find dij */
		datacpy(vecij, Xi, N);
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, Xj, &INCX, vecij, &INCY);

		alpha = 1.0;
		beta = 0;
		SYMV(&UPLO, &SYMVN, &alpha, A, &SYMVLDA, vecij, &INCX, &beta, Avec, &INCY);
		dij = DOT(&DOTN, vecij, &INCX, Avec, &INCY);

		/* Find dkl */
		datacpy(veckl, Xk, N);
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, Xl, &INCX, veckl, &INCY);

		alpha = 1.0;
		beta = 0;
		SYMV(&UPLO, &SYMVN, &alpha, A, &SYMVLDA, veckl, &INCX, &beta, Avec, &INCY);
		dkl = DOT(&DOTN, veckl, &INCX, Avec, &INCY);

		viol = dij - dkl - margin;
		if (viol > 0) {
			objtemp += SQR(viol);
			if (grad != NULL) {
				alpha = 2.0 * viol;
				SYR(&UPLO, &SYRN, &alpha, vecij, &INCX, grad, &SYRLDA);

				alpha = - 2.0 * viol;
				SYR(&UPLO, &SYRN, &alpha, veckl, &INCX, grad, &SYRLDA);
			}
		}
	}
	*obj = objtemp / (DOUBLE) numConstraints;
	if (grad != NULL) {
		INT SCALN = SQR(N);
		alpha = 1.0 / (DOUBLE) numConstraints;
		INT incx = 1;
		SCAL(&SCALN, &alpha, grad, &incx);
	}

	if (AvecFlag == 1) {
		FREE(Avec);
	}

	if (vecijFlag == 1) {
		FREE(vecij);
	}

	if (vecklFlag == 1) {
		FREE(veckl);
	}
}

void ml_bound_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, \
			DOUBLE *X, INT *constraintMat, DOUBLE *boundVec, INT N, \
			INT numConstraints, DOUBLE *Avec, DOUBLE *vec) {

	INT AvecFlag = 0;
	if (Avec == NULL) {
		Avec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		AvecFlag = 1;
	}

	INT vecFlag = 0;
	if (vec == NULL) {
		vec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		vecFlag = 1;
	}

	INT coordi;
	INT coordj;
	INT label;
	DOUBLE *Xi;
	DOUBLE *Xj;
	DOUBLE bound;
	DOUBLE dij;

	CHAR UPLO = 'U';
	INT SYMVN = N;
	INT SYMVLDA = N;
	INT SYRN = N;
	INT SYRLDA = N;
	INT DOTN = N;
	INT AXPYN = N;
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;

	DOUBLE objtemp = 0;
	if (grad != NULL) {
		memset((void *) grad, 0, SQR(N) * sizeof(DOUBLE));
	}

	INT iterConstraint;
	for (iterConstraint = 0; iterConstraint < numConstraints; \
				++iterConstraint) {
		/* bound */
		bound = boundVec[iterConstraint];
		/* i */
		coordi = constraintMat[iterConstraint];
		/* j */
		coordj = constraintMat[numConstraints + iterConstraint];
		/* label */
		label = constraintMat[numConstraints + 2 * iterConstraint];

		/* Get X vectors */
		Xi = &X[coordi * N];
		Xj = &X[coordj * N];

		/* Find dij */
		datacpy(vec, Xi, N);
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, Xj, &INCX, vec, &INCY);

		alpha = 1.0;
		beta = 0;
		SYMV(&UPLO, &SYMVN, &alpha, A, &SYMVLDA, vec, &INCX, &beta, Avec, &INCY);
		dij = DOT(&DOTN, vec, &INCX, Avec, &INCY);

		if (label * (dij - bound) > 0) {
			objtemp += label * (dij - bound);
			if (grad != NULL) {
				alpha = (DOUBLE) label;
				SYR(&UPLO, &SYRN, &alpha, vec, &INCX, grad, &SYRLDA);
			}
		}
	}
	*obj = objtemp / (DOUBLE) numConstraints;
	if (grad != NULL) {
		INT SCALN = SQR(N);
		alpha = 1.0 / (DOUBLE) numConstraints;
		INT incx = 1;
		SCAL(&SCALN, &alpha, grad, &incx);
	}

	if (AvecFlag == 1) {
		FREE(Avec);
	}

	if (vecFlag == 1) {
		FREE(vec);
	}
}
