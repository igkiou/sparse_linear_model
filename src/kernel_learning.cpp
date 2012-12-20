/*
 * kernel_learning.c
 *
 *  Created on: Apr 27, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useinterfaces.h"
#include "useblas.h"
#include "utils.h"
#include "matrix_proximal.h"
#include "kernel_learning.h"
#include "metric_learning.h"
//#include "svm_optimization.h"
//#include "learn_mahalanobis.h"

void itkl(DOUBLE *K, INT *constraintMat, DOUBLE *boundVec, \
		DOUBLE tolerance, DOUBLE gamma, INT maxEpochs, INT randomizationFlag, \
		INT numSamples, INT numConstraints, INT *constraintPerm, DOUBLE *lambda, \
		DOUBLE *lambdaOld, DOUBLE *vec) {

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
		vec = (DOUBLE *) MALLOC(numSamples * 1 * sizeof(DOUBLE));
		vecFlag = 1;
	}

	INT currConstraint;
	INT iterConstraint;
	INT currConstraintType;
	INT epochCount;
	INT currInd1;
	INT currInd2;
	DOUBLE *currK1;
	DOUBLE *currK2;
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
	INT AXPYN = numSamples;
	INT AXPYN2 = numConstraints;
	INT SYRN = numSamples;
	INT SYRLDA = numSamples;
	INT NRM2N = numConstraints;
	DOUBLE alpha;

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
		currK1 = &K[currInd1 * numSamples];
		currK2 = &K[currInd2 * numSamples];

		datacpy(vec, currK1, numSamples);
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, currK2, &INCX, vec, &INCY);

		wtw = K[currInd1 * numSamples + currInd1] \
			+ K[currInd2 * numSamples + currInd2] \
			- K[currInd1 * numSamples + currInd2];

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
		SYR(&UPLO, &SYRN, &alpha, vec, &INCX, K, &SYRLDA);

		++iterConstraint;
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
	for (iterN = 0; iterN < numSamples; ++iterN) {
		for (iterM = iterN + 1; iterM < numSamples; ++iterM) {
			K[iterN * numSamples + iterM] = K[iterM * numSamples + iterN];
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
}

void nrkl_svt(DOUBLE *K, INT *constraintMat, DOUBLE *betaVec, DOUBLE tau, \
			DOUBLE delta, INT numIters, DOUBLE tolerance, INT numPoints, \
			INT numConstraints) {

	DOUBLE *zetaVec = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
	DOUBLE *lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * numPoints * sizeof(INT));

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	INT iterConstraint;
	INT iter;
	DOUBLE maxDivergence;
	DOUBLE normDivergence;
	DOUBLE zeta;
	DOUBLE beta;
	DOUBLE normK;
	INT coord1, coord2, coord3, coord4;
	DOUBLE dij, dkl;

	INT lwork = -1;
	DOUBLE *work;
	DOUBLE work_temp;

	nuclear_psd_proximal(K, NULL, tau, numPoints, lVec, \
			Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

//	memset(zetaVec, 0, numConstraints * sizeof(DOUBLE));

	/* kick by a few iterations */
	/* dij - dkl < - 1, beta = - 1, NIPS10 */
	DOUBLE minNegBeta = 0;
	for (iterConstraint = 0; iterConstraint < numConstraints; \
		++iterConstraint) {
		if (betaVec[iterConstraint] < minNegBeta) {
			minNegBeta = betaVec[iterConstraint];
		}
	}
	DOUBLE kappaInit = floor(tau / numPoints / ABS(minNegBeta) / delta);
	for (iterConstraint = 0; iterConstraint < numConstraints; \
		++iterConstraint) {
		zetaVec[iterConstraint] = kappaInit * delta * IMAX(-betaVec[iterConstraint], 0.0);
	}
//	/* dkl - dij > 1, beta = 1, Uzawa08 */
//	DOUBLE maxPosBeta = 0;
//	for (iterConstraint = 0; iterConstraint < numConstraints; \
//		++iterConstraint) {
//		if (betaVec[iterConstraint] > maxPosBeta) {
//			maxPosBeta = betaVec[iterConstraint];
//		}
//	}
//	DOUBLE kappaInit = floor(tau / numPoints / ABS(maxPosBetaBeta) / delta);
//	for (iterConstraint = 0; iterConstraint < numConstraints; \
//		++iterConstraint) {
//		zetaVec[iterConstraint] = kappaInit * delta * IMAX(betaVec[iterConstraint], 0);
//	}

	iter = 0;
	while (1) {
		++iter;
		if (iter % 1000 == 0) {
			printf("Now running iter %d\n", iter);
		}
		memset((void *) K, 0, numPoints * numPoints * sizeof(DOUBLE));
		for (iterConstraint = 0; iterConstraint < numConstraints; \
			++iterConstraint) {
			zeta = zetaVec[iterConstraint];
			if (zeta != 0) {
				/* i */
				coord1 = constraintMat[iterConstraint];
				/* j */
				coord2 = constraintMat[numConstraints + iterConstraint];
				/* k */
				coord3 = constraintMat[numConstraints * 2 + iterConstraint];
				/* l */
				coord4 = constraintMat[numConstraints * 3 + iterConstraint];
				/* dij - dkl */
				K[numPoints * coord1 + coord1] += zeta;
				K[numPoints * coord2 + coord2] += zeta;
				K[numPoints * coord3 + coord3] -= zeta;
				K[numPoints * coord4 + coord4] -= zeta;
				K[numPoints * coord1 + coord2] -= zeta;
				K[numPoints * coord2 + coord1] -= zeta;
				K[numPoints * coord3 + coord4] += zeta;
				K[numPoints * coord4 + coord3] += zeta;
//				/* dkl - dij */
//				K[numPoints * coord1 + coord1] -= zeta;
//				K[numPoints * coord2 + coord2] -= zeta;
//				K[numPoints * coord3 + coord3] += zeta;
//				K[numPoints * coord4 + coord4] += zeta;
//				K[numPoints * coord1 + coord2] += zeta;
//				K[numPoints * coord2 + coord1] += zeta;
//				K[numPoints * coord3 + coord4] -= zeta;
//				K[numPoints * coord4 + coord3] -= zeta;
			}
		}

		nuclear_psd_proximal(K, &normK, tau, numPoints, lVec, \
					Vr, work, lwork);

		maxDivergence = 0;
		normDivergence = 0;
		for (iterConstraint = 0; iterConstraint < numConstraints; \
			++iterConstraint) {
			beta = betaVec[iterConstraint];
			/* i */
			coord1 = constraintMat[iterConstraint];
			/* j */
			coord2 = constraintMat[numConstraints + iterConstraint];
			/* k */
			coord3 = constraintMat[numConstraints * 2 + iterConstraint];
			/* l */
			coord4 = constraintMat[numConstraints * 3 + iterConstraint];
			dij = K[coord1 * numPoints + coord1] \
				+ K[coord2 * numPoints + coord2] \
				- 2 * K[coord1 * numPoints + coord2];
			dkl = K[coord3 * numPoints + coord3] \
				+ K[coord4 * numPoints + coord4] \
				- 2 * K[coord3 * numPoints + coord4];
			/* dij - dkl < - 1, beta = - 1, NIPS10 */
			zetaVec[iterConstraint] -= delta * IMAX(dij - dkl - beta, 0.0);
			if (dij - dkl - beta > 0) {
				maxDivergence = dij - dkl - beta;
				normDivergence += SQR(dij - dkl - beta);
			}
//			/* dkl - dij > 1, beta = 1, Uzawa08 */
//	 		zetaVec[iterConstraint] = IMAX(zetaVec[iterConstraint] \
//	 									+ delta * (beta + dij - dkl), 0);
//	 		if (dkl - dij - beta < 0) {
//	 			maxDivergence = beta - dkl + dij;
//	 			normDivergence += SQR(beta + dij - dkl);
//	 		}
		}

		if ((normDivergence < tolerance) || (maxDivergence < eps)) {
			break;
		}
		if (iter == numIters) {
			break;
		}
	}

	FREE(work);
	FREE(zetaVec);
	FREE(lVec);
	FREE(Vr);
	FREE(isuppz);
}

// TODO: Find better initialization than setting everything to zero.
// TODO: Write version with lineSearch.
// TODO: Check convergence criteria.
// TODO: Change mfiles to also have continuation.
void nrkl_fpc_continuation(DOUBLE *K, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE *weights, \
			DOUBLE tau, DOUBLE delta, INT numIters, DOUBLE tolerance, \
			DOUBLE tauMultiplier, DOUBLE tauRate, INT numPoints, \
			INT numConstraints) {

	DOUBLE *KOld = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * numPoints * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	nuclear_psd_proximal(K, NULL, tau, numPoints, lVec, Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	memset((void *) K, 0, SQR(numPoints) * sizeof(DOUBLE));
	nrkl_fp_inner(K, problemType, constraintMat, betaVec, weights, tau, delta, \
			numIters, tolerance, tauMultiplier, tauRate, numPoints, \
			numConstraints, KOld, lVec, Vr, isuppz, work, lwork);

	FREE(KOld);
	FREE(lVec);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void nrkl_fpc(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE* tau, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, INT numPoints, INT numConstraints, \
			INT numRepeats) {

	DOUBLE *KOld = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * numPoints * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	nuclear_psd_proximal(K, NULL, tau[0], numPoints, lVec, \
				Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	memset((void *) K, 0, SQR(numPoints) * sizeof(DOUBLE));
	INT iterRepeat;
	for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
		printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, \
				numRepeats, tau[iterRepeat]);
		nrkl_fp_inner(K, problemType, constraintMat, betaVec, weights, \
			tau[iterRepeat], delta, numIters, tolerance, 1, 1, numPoints, \
			numConstraints, KOld, lVec, Vr, isuppz, work, lwork);
	}

	FREE(KOld);
	FREE(lVec);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void nrkl_fp_inner(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE tauTarget, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, DOUBLE tauMultiplier, \
			DOUBLE tauRate, INT numPoints, INT numConstraints, DOUBLE *KOld, \
			DOUBLE *lVec, DOUBLE *Vr, INT *isuppz, DOUBLE *work, INT lwork) {

	INT KOldFlag = 0;
	if (KOld == NULL) {
		KOld = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
		KOldFlag = 1;
	}

	INT lVecFlag = 0;
	if (lVec == NULL) {
		lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
		lVecFlag = 1;
	}

	INT VrFlag = 0;
	if (Vr == NULL) {
		Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
		VrFlag = 1;
	}

	INT isuppzFlag = 0;
	if (isuppz == NULL) {
		isuppz = (INT *) MALLOC(2 * numPoints * sizeof(INT));
		isuppzFlag = 1;
	}

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	DOUBLE normK;
//	DOUBLE normKOld;
	DOUBLE normKDiff;
	DOUBLE objtemp;

	CHAR LANSYNORM = 'F';
	CHAR UPLO = 'U';
	INT LANSYN = numPoints;
	INT LANSYLDA = numPoints;
	INT AXPBYN = SQR(numPoints);
	INT AXPYN = SQR(numPoints);
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;

	INT workFlag = 0;
	if (lwork == -1) {
		DOUBLE work_temp;
		nuclear_psd_proximal(K, NULL, tauTarget, numPoints, lVec, \
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

		datacpy(KOld, K, numPoints * numPoints);
//		normKOld = LANSY(&LANSYNORM, &UPLO, &LANSYN, KOld, &LANSYLDA, NULL);

		kl_obj_grad(&objtemp, K, KOld, problemType, constraintMat, betaVec, \
					weights, numPoints, numConstraints);

		alpha = 1.0;
		beta = - delta;
		AXPBY(&AXPBYN, &alpha, KOld, &INCX, &beta, K, &INCY);
		datacpy(KOld, K, numPoints * numPoints);

		nuclear_psd_proximal(K, NULL, tau * delta, numPoints, lVec, \
					Vr, work, lwork);

		tau = IMAX(tauRate * tau, tauTarget);

		if (iter == numIters) {
			break;
		}

		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, K, &INCX, KOld, &INCY);
		normKDiff = LANSY(&LANSYNORM, &UPLO, &LANSYN, KOld, &LANSYLDA, NULL);
		normK = LANSY(&LANSYNORM, &UPLO, &LANSYN, K, &LANSYLDA, NULL);

		if ((normK > eps) && (normKDiff / IMAX(1.0, normK) < tolerance)) {
			break;
		}
	}

	if (KOldFlag == 1) {
		FREE(KOld);
	}

	if (lVecFlag == 1) {
		FREE(lVec);
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

void nrkl_svp(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, INT rank, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, INT numPoints, INT numConstraints) {

	DOUBLE *KOld = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * numPoints * sizeof(INT));

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	INT iter;
	DOUBLE normK;
	DOUBLE normKOld;
	DOUBLE normKDiff;
	DOUBLE objtemp;

	CHAR LANSYNORM = 'F';
	CHAR UPLO = 'U';
	INT LANSYN = numPoints;
	INT LANSYLDA = numPoints;
	INT AXPBYN = SQR(numPoints);
	INT AXPYN = SQR(numPoints);
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;

	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	nuclear_psd_hard_thresholding(K, NULL, rank, numPoints, lVec, Vr, \
			&work_temp, lwork); //, &iwork_temp, liwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	memset((void *) K, 0, SQR(numPoints) * sizeof(DOUBLE));
	iter = 0;
	while (1) {
		++iter;
		if (iter % 1000 == 0) {
			printf("Now running iter %d\n", iter);
		}

		datacpy(KOld, K, numPoints * numPoints);
		normKOld = LANSY(&LANSYNORM, &UPLO, &LANSYN, KOld, &LANSYLDA, NULL);

		kl_obj_grad(&objtemp, K, KOld, problemType, constraintMat, betaVec, \
					weights, numPoints, numConstraints);

		alpha = 1.0;
		beta = - delta;
		AXPBY(&AXPBYN, &alpha, KOld, &INCX, &beta, K, &INCY);

		nuclear_psd_hard_thresholding(K, NULL, rank, numPoints, lVec, Vr, \
				work, lwork); //, iwork, liwork);

		if (iter == numIters) {
			break;
		}
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, K, &INCX, KOld, &INCY);
		normKDiff = LANSY(&LANSYNORM, &UPLO, &LANSYN, KOld, &LANSYLDA, NULL);
		normK = LANSY(&LANSYNORM, &UPLO, &LANSYN, K, &LANSYLDA, NULL);

		if ((normK > eps) && (normKDiff / IMAX(1.0, normKOld) < tolerance)) {
			break;
		}
	}

	FREE(work);
//	FREE(iwork);
	FREE(KOld);
	FREE(lVec);
	FREE(Vr);
	FREE(isuppz);
}

void nrkl_apg_continuation(DOUBLE *K, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE *weights, \
			DOUBLE tau, DOUBLE delta0, 	INT numIters, DOUBLE tolerance, \
			INT lineSearchFlag, DOUBLE eta, DOUBLE tauMultiplier, \
			DOUBLE tauRate, INT numPoints, INT numConstraints) {

	DOUBLE *KOld = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *L = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *LfGrad = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *LfGD = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *KLfGDDiff = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * numPoints * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	nuclear_psd_proximal(K, NULL, tau, numPoints, lVec, Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	memset((void *) K, 0, SQR(numPoints) * sizeof(DOUBLE));
	nrkl_apg_inner(K, problemType, constraintMat, betaVec, weights, \
			tau, delta0, numIters, tolerance, lineSearchFlag, eta, \
			tauMultiplier, tauRate, numPoints, numConstraints, KOld, L, LfGrad, \
			LfGD, KLfGDDiff, lVec, Vr, isuppz, work, lwork);

	FREE(KOld);
	FREE(L);
	FREE(LfGrad);
	FREE(LfGD);
	FREE(KLfGDDiff);
	FREE(lVec);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void nrkl_apg(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE *tau, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, \
			INT numPoints, INT numConstraints, INT numRepeats) {

	DOUBLE *KOld = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *L = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *LfGrad = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *LfGD = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *KLfGDDiff = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * numPoints * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	nuclear_psd_proximal(K, NULL, tau[0], numPoints, lVec, \
				Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	memset((void *) K, 0, SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE delta;
	INT iterRepeat;
	for (iterRepeat = 0; iterRepeat < numRepeats; ++iterRepeat) {
		printf("Now running repeat %d out of %d, tau %lf.\n", iterRepeat + 1, numRepeats, tau[iterRepeat]);
		delta = delta0;
		nrkl_apg_inner(K, problemType, constraintMat, betaVec, weights, \
				tau[iterRepeat], delta, numIters, tolerance, lineSearchFlag, \
				eta, 1, 1, numPoints, numConstraints, KOld, L, LfGrad, LfGD, \
				KLfGDDiff, lVec, Vr, isuppz, work, lwork);
	}

	FREE(KOld);
	FREE(L);
	FREE(LfGrad);
	FREE(LfGD);
	FREE(KLfGDDiff);
	FREE(lVec);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void nrkl_apg_inner(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE tauTarget, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, \
			DOUBLE tauMultiplier, DOUBLE tauRate, INT numPoints, \
			INT numConstraints, DOUBLE *KOld, DOUBLE *L, DOUBLE *LfGrad, \
			DOUBLE *LfGD, DOUBLE *KLfGDDiff, DOUBLE *lVec, DOUBLE *Vr, \
			INT *isuppz, DOUBLE *work, INT lwork) {

	INT KOldFlag = 0;
	if (KOld == NULL) {
		KOld = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
		KOldFlag = 1;
	}

	INT LFlag = 0;
	if (L == NULL) {
		L = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
		LFlag = 1;
	}

	INT LfGradFlag = 0;
	if (LfGrad == NULL) {
		LfGrad = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
		LfGradFlag = 1;
	}

	INT LfGDFlag = 0;
	if (LfGD == NULL) {
		LfGD = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
		LfGDFlag = 1;
	}

	INT KLfGDDiffFlag = 0;
	if (KLfGDDiff == NULL) {
		KLfGDDiff = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
		KLfGDDiffFlag = 1;
	}

	INT lVecFlag = 0;
	if (lVec == NULL) {
		lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
		lVecFlag = 1;
	}

	INT VrFlag = 0;
	if (Vr == NULL) {
		Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
		VrFlag = 1;
	}

	INT isuppzFlag = 0;
	if (isuppz == NULL) {
		isuppz = (INT *) MALLOC(2 * numPoints * sizeof(INT));
		isuppzFlag = 1;
	}

	INT workFlag = 0;
	if (lwork == -1) {
		DOUBLE work_temp;
		nuclear_psd_proximal(K, NULL, tauTarget, numPoints, lVec, \
				Vr, &work_temp, lwork);
		lwork = (INT) work_temp;
		work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));
		workFlag = 1;
	}

	INT AXPBYN = SQR(numPoints);
	DOUBLE alpha;
	DOUBLE beta;
	INT INCX = 1;
	INT INCY = 1;
	CHAR LANSYNORM = 'F';
	CHAR UPLO = 'U';
	INT LANSYN = numPoints;
	INT LANSYLDA = numPoints;
	INT AXPYN = SQR(numPoints);

	DOUBLE t = 1;
	DOUBLE tOld = 1;
	DOUBLE LfObj;
	DOUBLE delta = delta0;
	DOUBLE normKDiff;
//	DOUBLE normKOld;
	DOUBLE normK;

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	INT iter = 0;
	DOUBLE tau = tauMultiplier * tauTarget;
	while (1) {
		++iter;
		if (iter % 1000 == 0) {
			printf("Now running iter %d\n", iter);
		}

		datacpy(L, K, SQR(numPoints));
		alpha = - (tOld - 1.0) / t;
		beta = 1 - alpha;
 		AXPBY(&AXPBYN, &alpha, KOld, &INCX, &beta, L, &INCY);
 		datacpy(KOld, K, SQR(numPoints));
 		kl_obj_grad(&LfObj, LfGrad, L, problemType, constraintMat, betaVec, \
 				weights, numPoints, numConstraints);

 		nrkl_line_search(&delta, K, L, LfObj, LfGrad, tau, lineSearchFlag, eta, \
 				problemType, constraintMat, betaVec, weights, numPoints, \
 				numConstraints, LfGD, KLfGDDiff, lVec, Vr, work, lwork);
		tOld = t;
		t = (1 + SQRT(1 + 4 * SQR(tOld))) * 0.5;
		tau = IMAX(tauRate * tau, tauTarget);

		if (iter == numIters) {
			break;
		}

		datacpy(KLfGDDiff, LfGrad, SQR(numPoints));
		alpha = - 1.0;
		AXPY(&AXPYN, &alpha, K, &INCX, KLfGDDiff, &INCY);
//		alpha = - 1.0;
//		AXPY(&AXPYN, &alpha, K, &INCX, KOld, &INCY);
		normKDiff = LANSY(&LANSYNORM, &UPLO, &LANSYN, KLfGDDiff, &LANSYLDA, NULL);
		normK = LANSY(&LANSYNORM, &UPLO, &LANSYN, K, &LANSYLDA, NULL);

		if ((normK > eps) && (normKDiff / IMAX(1.0, normK) < tolerance)) {
			break;
		}

//		datacpy(KLfGDDiff, K, SQR(numPoints));
//		alpha = - 1.0;
//		AXPY(&AXPYN, &alpha, KOld, &INCX, KLfGDDiff, &INCY);
//		normKDiff = LANSY(&LANSYNORM, &UPLO, &LANSYN, KLfGDDiff, &LANSYLDA, NULL);
//		normKOld = LANSY(&LANSYNORM, &UPLO, &LANSYN, KOld, &LANSYLDA, NULL);
//		normK = LANSY(&LANSYNORM, &UPLO, &LANSYN, K, &LANSYLDA, NULL);
//
//		if ((normK > eps) && (normKDiff / IMAX(1.0, normKOld) < tolerance)) {
//			break;
//		}
	}

	if (KOldFlag == 1) {
		FREE(KOld);
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

	if (KLfGDDiffFlag == 1) {
		FREE(KLfGDDiff);
	}

	if (lVecFlag == 1) {
		FREE(lVec);
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

void nrkl_line_search(DOUBLE *delta, DOUBLE *LfGDShrink, DOUBLE *L, \
		DOUBLE LfObj, DOUBLE *LfGrad, DOUBLE tau, INT lineSearchFlag, \
		DOUBLE eta, DIST_LABEL_TYPE problemType, INT *constraintMat, \
		DOUBLE *betaVec, DOUBLE *weights, INT numPoints, INT numConstraints, \
		DOUBLE *LfGD, DOUBLE *KLfGDDiff, DOUBLE *lVec, DOUBLE *Vr, DOUBLE *work, \
		INT lwork) {

	INT KLfGDDiffFlag = 0;
	if (KLfGDDiff == NULL) {
		KLfGDDiff = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
		KLfGDDiffFlag = 1;
	}

	INT lVecFlag = 0;
	if (lVec == NULL) {
		lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
		lVecFlag = 1;
	}

	INT VrFlag = 0;
	if (Vr == NULL) {
		Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
		VrFlag = 1;
	}

	INT AXPYN = SQR(numPoints);
	DOUBLE alpha;
	INT INCX = 1;
	INT INCY = 1;

	DOUBLE LfGDShrinkfObj;
	DOUBLE LQmodObj;
	DOUBLE deltatemp = *delta;
	while (1) {
		datacpy(LfGD, L, SQR(numPoints));
		alpha = - 1.0 / deltatemp;
		AXPY(&AXPYN, &alpha, LfGrad, &INCX, LfGD, &INCY);
		nrkl_shrinkage(LfGDShrink, NULL, LfGD, tau / deltatemp, numPoints, lVec, \
						Vr, work, lwork);
		if (lineSearchFlag == 0) {
			break;
		}
		kl_obj_grad(&LfGDShrinkfObj, NULL, LfGDShrink, problemType, \
				constraintMat, betaVec, weights, numPoints, numConstraints);
		nrkl_Q_func_mod(&LQmodObj, LfGDShrink, LfGD, LfObj, LfGrad, \
				deltatemp, numPoints, KLfGDDiff);
		if (LfGDShrinkfObj <= LQmodObj) {
			break;
		}
		deltatemp = deltatemp * eta;
	}

	*delta = deltatemp;
	if (KLfGDDiffFlag == 1) {
		FREE(KLfGDDiff);
	}

	if (lVecFlag == 1) {
		FREE(lVec);
	}

	if (VrFlag == 1) {
		FREE(Vr);
	}
}

void nrkl_shrinkage(DOUBLE *KShrink, DOUBLE *KShrinkNorm, DOUBLE *K, \
				DOUBLE tau, INT numPoints, DOUBLE *lVec, DOUBLE *Vr, \
				DOUBLE *work, INT lwork) {

	INT lVecFlag = 0;
	if (lVec == NULL) {
		lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
		lVecFlag = 1;
	}

	INT VrFlag = 0;
	if (Vr == NULL) {
		Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
		VrFlag = 1;
	}

	datacpy(KShrink, K, SQR(numPoints));
	// TODO: Implement truncation.
	nuclear_psd_proximal(KShrink, KShrinkNorm, tau, numPoints, lVec, \
				Vr, work, lwork);

	if (lVecFlag == 1) {
		FREE(lVec);
	}

	if (VrFlag == 1) {
		FREE(Vr);
	}
}

void nrkl_Q_func_mod(DOUBLE *QObj, DOUBLE *K, DOUBLE *LfGD, DOUBLE LfObj, \
			DOUBLE *LfGrad, DOUBLE delta, INT numPoints, DOUBLE *KLfGDDiff) {

	INT KLfGDDiffFlag = 0;
	if (KLfGDDiff == NULL) {
		KLfGDDiff = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
		KLfGDDiffFlag = 1;
	}

	datacpy(KLfGDDiff, K, SQR(numPoints));
	INT AXPYN = SQR(numPoints);
	DOUBLE alpha = - 1.0;
	INT INCX = 1;
	INT INCY = 1;
	AXPY(&AXPYN, &alpha, LfGD, &INCX, KLfGDDiff, &INCY);

	CHAR NORM = 'F';
	CHAR UPLO = 'U';
	INT LANSYN = numPoints;
	INT LANSYLDA = numPoints;
	DOUBLE *work = NULL;
	DOUBLE KLfGDDiffNorm = LANSY(&NORM, &UPLO, &LANSYN, KLfGDDiff, &LANSYLDA, \
								work);
	DOUBLE LfGradNorm = LANSY(&NORM, &UPLO, &LANSYN, LfGrad, &LANSYLDA, \
								work);
	*QObj = delta * 0.5 * SQR(KLfGDDiffNorm) - 0.5 / delta * SQR(LfGradNorm) \
			+ LfObj;

	if (KLfGDDiffFlag == 1) {
		FREE(KLfGDDiff);
	}
}

//void nrkl_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, INT *constraintMat, \
//			DOUBLE *betaVec, INT numPoints, INT numConstraints) {
//
//	INT coord1;
//	INT coord2;
//	INT coord3;
//	INT coord4;
//	DOUBLE beta;
//	DOUBLE dij;
//	DOUBLE dkl;
//
//	DOUBLE objtemp = 0;
//	if (grad != NULL) {
//		memset((void *) grad, 0, SQR(numPoints) * sizeof(DOUBLE));
//	}
//
//	INT iterConstraint;
//	for (iterConstraint = 0; iterConstraint < numConstraints; \
//				++iterConstraint) {
//		beta = betaVec[iterConstraint];
//		/* i */
//		coord1 = constraintMat[iterConstraint];
//		/* j */
//		coord2 = constraintMat[numConstraints + iterConstraint];
//		/* k */
//		coord3 = constraintMat[numConstraints * 2 + iterConstraint];
//		/* l */
//		coord4 = constraintMat[numConstraints * 3 + iterConstraint];
//		dij = K[coord1 * numPoints + coord1] \
//			+ K[coord2 * numPoints + coord2] \
//			- 2 * K[coord1 * numPoints + coord2];
//		dkl = K[coord3 * numPoints + coord3] \
//			+ K[coord4 * numPoints + coord4] \
//			- 2 * K[coord3 * numPoints + coord4];
//		/* dij - dkl < - 1, beta = - 1, NIPS10 */
//		if (dij - dkl - beta > 0) {
//			objtemp += dij - dkl - beta;
//			if (grad != NULL) {
//				grad[numPoints * coord1 + coord1]++;
//				grad[numPoints * coord2 + coord2]++;
//				grad[numPoints * coord3 + coord3]--;
//				grad[numPoints * coord4 + coord4]--;
//				grad[numPoints * coord1 + coord2]--;
//				grad[numPoints * coord2 + coord1]--;
//				grad[numPoints * coord3 + coord4]++;
//				grad[numPoints * coord4 + coord3]++;
//			}
//		}
//	}
//	*obj = objtemp;
//}

void frkl_pgd(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE kappa, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT stepFlag, INT numPoints, \
			INT numConstraints) {

	DOUBLE *KOld = (DOUBLE *) MALLOC(SQR(numPoints) * sizeof(DOUBLE));
	DOUBLE *lVec = (DOUBLE *) MALLOC(numPoints * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(numPoints * numPoints * sizeof(DOUBLE));
	INT *isuppz = (INT *) MALLOC(2 * numPoints * sizeof(INT));
	DOUBLE work_temp;
	DOUBLE *work;
	INT lwork = -1;
	matrix_psd_projection(K, numPoints, lVec, Vr, &work_temp, lwork);

	lwork = (INT) work_temp;
	work = (DOUBLE *) MALLOC(lwork * sizeof(DOUBLE));

	CHAR lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);

	DOUBLE normGrad;
	DOUBLE normK;
	DOUBLE objtemp;

	CHAR LANSYNORM = 'F';
	CHAR UPLO = 'U';
	INT LANSYN = numPoints;
	INT LANSYLDA = numPoints;
	INT AXPBYN = SQR(numPoints);
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
		datacpy(KOld, K, numPoints * numPoints);

		frkl_obj_grad(&objtemp, K, KOld, problemType, constraintMat, betaVec, \
					weights, kappa, numPoints, numConstraints);

		alpha = 1.0;
		beta = - delta;
		AXPBY(&AXPBYN, &alpha, KOld, &INCX, &beta, K, &INCY);
		normGrad = LANSY(&LANSYNORM, &UPLO, &LANSYN, K, &LANSYLDA, NULL);

		matrix_psd_projection(K, numPoints, lVec, Vr, work, lwork);

		if (iter == numIters) {
			break;
		}

		normK = LANSY(&LANSYNORM, &UPLO, &LANSYN, K, &LANSYLDA, NULL);

		if ((normK > eps) && (normGrad / IMAX(1.0, normK) < tolerance)) {
			break;
		}
	}

	FREE(KOld);
	FREE(lVec);
	FREE(Vr);
	FREE(isuppz);
	FREE(work);
}

void frkl_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
			DIST_LABEL_TYPE problemType, INT *constraintMat, DOUBLE *betaVec, \
			DOUBLE *weights, DOUBLE kappa, INT numPoints, INT numConstraints) {

	kl_obj_grad(obj, grad, K, problemType, constraintMat, betaVec, weights, \
			numPoints, numConstraints);

	CHAR LANSYNORM = 'F';
	CHAR UPLO = 'U';
	INT LANSYN = numPoints;
	INT LANSYLDA = numPoints;
	DOUBLE normK = LANSY(&LANSYNORM, &UPLO, &LANSYN, K, &LANSYLDA, NULL);
	*obj += SQR(normK) * kappa / 2.0;
	if (grad != NULL) {
		INT AXPYN = SQR(numPoints);
		DOUBLE alpha = kappa;
		INT incx = 1;
		INT incy = 1;
		AXPY(&AXPYN, &alpha, K, &incx, grad, &incy);
	}
}

// TODO: Add control for NULL obj or grad here.
// TODO: Add 1 / numConstraints factor to m-file versions.
// TODO: Add weights.
void kl_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
			DIST_LABEL_TYPE problemType, INT *constraintMat, DOUBLE *betaVec, \
			DOUBLE *weights, INT numPoints, INT numConstraints) {

	if (problemType == DIST_LABEL_TARGETS) {
		kl_target_obj_grad(obj, grad, K, constraintMat, betaVec, weights, \
						numPoints, numConstraints);
	} else if (problemType == DIST_LABEL_RELATIONAL) {
		kl_relational_obj_grad(obj, grad, K, constraintMat, betaVec, weights, \
						numPoints, numConstraints);
	} else if (problemType == DIST_LABEL_BOUNDS) {
		kl_bound_obj_grad(obj, grad, K, constraintMat, betaVec, weights, \
						numPoints, numConstraints);
	} else if (problemType == DIST_LABEL_SQRHINGE) {
		kl_sqrhinge_obj_grad(obj, grad, K, constraintMat, betaVec, weights, \
						numPoints, numConstraints);
	} else if (problemType == DIST_LABEL_HUBERHINGE) {
		ERROR("kl_obj_grad function not implemented for DIST_LABEL_HUBERHINGE case yet.");
	}

}

void kl_target_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
			INT *constraintMat, DOUBLE *targetVec, DOUBLE *weights, \
			INT numPoints, INT numConstraints) {

	INT coordi;
	INT coordj;
	DOUBLE target;
	DOUBLE dij;
	DOUBLE alpha;
	DOUBLE weight;

	DOUBLE objtemp = 0;
	if (grad != NULL) {
		memset((void *) grad, 0, SQR(numPoints) * sizeof(DOUBLE));
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
		/* w */
		weight = (weights == NULL)?(1):(weights[iterConstraint]);

		dij = K[coordi * numPoints + coordi] \
			+ K[coordj * numPoints + coordj] \
			- 2 * K[coordi * numPoints + coordj];
		objtemp += 0.5 * SQR(dij - target) * weight;

		if (grad != NULL) {
			alpha = dij - target;
			grad[numPoints * coordi + coordi] += alpha * weight;
			grad[numPoints * coordj + coordj] += alpha * weight;
			grad[numPoints * coordi + coordj] -= alpha * weight;
			grad[numPoints * coordj + coordi] -= alpha * weight;
		}
	}
	*obj = objtemp / (DOUBLE) numConstraints;
	if (grad != NULL) {
		INT SCALN = SQR(numPoints);
		alpha = 1.0 / (DOUBLE) numConstraints;
		INT incx = 1;
		SCAL(&SCALN, &alpha, grad, &incx);
	}
}

/* Maybe order dataMatrix in MATLAB to avoid cache misses. */
void kl_relational_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
			INT *constraintMat, DOUBLE *marginVec, DOUBLE *weights, \
			INT numPoints, INT numConstraints) {

	INT coordi;
	INT coordj;
	INT coordk;
	INT coordl;
	DOUBLE margin;
	DOUBLE dij;
	DOUBLE dkl;
	DOUBLE viol;
	DOUBLE weight;

	DOUBLE objtemp = 0;
	if (grad != NULL) {
		memset((void *) grad, 0, SQR(numPoints) * sizeof(DOUBLE));
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
		/* w */
		weight = (weights == NULL)?(1):(weights[iterConstraint]);

		/* Find dij */
		dij = K[coordi * numPoints + coordi] \
			+ K[coordj * numPoints + coordj] \
			- 2 * K[coordi * numPoints + coordj];

		/* Find dkl */
		dkl = K[coordk * numPoints + coordk] \
			+ K[coordl * numPoints + coordl] \
			- 2 * K[coordk * numPoints + coordl];

		viol = dij - dkl - margin;
		if (viol > 0) {
			objtemp += viol * weight;
			if (grad != NULL) {
				grad[numPoints * coordi + coordi]+= weight;
				grad[numPoints * coordj + coordj]+= weight;
				grad[numPoints * coordi + coordj]-= weight;
				grad[numPoints * coordj + coordi]-= weight;

				grad[numPoints * coordk + coordk]-= weight;
				grad[numPoints * coordl + coordl]-= weight;
				grad[numPoints * coordk + coordl]+= weight;
				grad[numPoints * coordl + coordk]+= weight;
			}
		}
	}
	*obj = objtemp / (DOUBLE) numConstraints;
	if (grad != NULL) {
		INT SCALN = SQR(numPoints);
		DOUBLE alpha = 1.0 / (DOUBLE) numConstraints;
		INT incx = 1;
		SCAL(&SCALN, &alpha, grad, &incx);
	}
}

void kl_sqrhinge_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
		INT *constraintMat, DOUBLE *marginVec, DOUBLE *weights, INT numPoints, \
		INT numConstraints) {

	INT coordi;
	INT coordj;
	INT coordk;
	INT coordl;
	DOUBLE margin;
	DOUBLE dij;
	DOUBLE dkl;
	DOUBLE viol;
	DOUBLE weight;

	DOUBLE objtemp = 0;
	if (grad != NULL) {
		memset((void *) grad, 0, SQR(numPoints) * sizeof(DOUBLE));
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
		/* w */
		weight = (weights == NULL)?(1):(weights[iterConstraint]);

		/* Find dij */
		dij = K[coordi * numPoints + coordi] \
			+ K[coordj * numPoints + coordj] \
			- 2 * K[coordi * numPoints + coordj];

		/* Find dkl */
		dkl = K[coordk * numPoints + coordk] \
			+ K[coordl * numPoints + coordl] \
			- 2 * K[coordk * numPoints + coordl];

		viol = dij - dkl - margin;
		if (viol > 0) {
			objtemp += SQR(viol) * weight;
			if (grad != NULL) {
				grad[numPoints * coordi + coordi]+= 2 * viol * weight;
				grad[numPoints * coordj + coordj]+= 2 * viol * weight;
				grad[numPoints * coordi + coordj]-= 2 * viol * weight;
				grad[numPoints * coordj + coordi]-= 2 * viol * weight;

				grad[numPoints * coordk + coordk]-= 2 * viol * weight;
				grad[numPoints * coordl + coordl]-= 2 * viol * weight;
				grad[numPoints * coordk + coordl]+= 2 * viol * weight;
				grad[numPoints * coordl + coordk]+= 2 * viol * weight;
			}
		}
	}
	*obj = objtemp / (DOUBLE) numConstraints;
	if (grad != NULL) {
		INT SCALN = SQR(numPoints);
		DOUBLE alpha = 1.0 / (DOUBLE) numConstraints;
		INT incx = 1;
		SCAL(&SCALN, &alpha, grad, &incx);
	}
}

void kl_bound_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, INT *constraintMat, \
			DOUBLE *boundVec, DOUBLE *weights, INT numPoints, \
			INT numConstraints) {

	INT coordi;
	INT coordj;
	INT label;
	DOUBLE bound;
	DOUBLE dij;
	DOUBLE alpha;
	DOUBLE weight;

	DOUBLE objtemp = 0;
	if (grad != NULL) {
		memset((void *) grad, 0, SQR(numPoints) * sizeof(DOUBLE));
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
		/* w */
		weight = (weights == NULL)?(1):(weights[iterConstraint]);

		/* Find dij */
		dij = K[coordi * numPoints + coordi] \
			+ K[coordj * numPoints + coordj] \
			- 2 * K[coordi * numPoints + coordj];

		if (label * (dij - bound) > 0) {
			objtemp += label * (dij - bound) * weight;
			if (grad != NULL) {
				alpha = (DOUBLE) label;
				grad[numPoints * coordi + coordi] += alpha * weight;
				grad[numPoints * coordj + coordj] += alpha * weight;
				grad[numPoints * coordi + coordj] -= alpha * weight;
				grad[numPoints * coordj + coordi] -= alpha * weight;
			}
		}
	}
	*obj = objtemp / (DOUBLE) numConstraints;
	if (grad != NULL) {
		INT SCALN = SQR(numPoints);
		alpha = 1.0 / (DOUBLE) numConstraints;
		INT incx = 1;
		SCAL(&SCALN, &alpha, grad, &incx);
	}
}

//void semisupervised_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *W, \
//		DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, DOUBLE *classLabels, DOUBLE *D, DOUBLE *DDt, \
//		DOUBLE *mu1p, DOUBLE *mu2p, INT N, INT K, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag, \
//		DOUBLE *MD, DOUBLE *ObjMat, DOUBLE *MDDt, DOUBLE *derivTemp) {
//
//	INT MDFlag = 0;
//	if (MD == NULL) {
//		MD = (DOUBLE *) MALLOC(N * K * sizeof(DOUBLE));
//		MDFlag = 1;
//	}
//	INT ObjMatFlag = 0;
//	if (ObjMat == NULL) {
//		ObjMat = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
//		ObjMatFlag = 1;
//	}
//	INT MDDtFlag = 0;
//	if ((MDDt == NULL) && (derivFlag == 1)){
//		MDDt = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
//		MDDtFlag = 1;
//	}
//	INT derivTempFlag = 0;
//	if ((derivTemp == NULL) && (derivFlag == 1)){
//		derivTemp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
//		derivTempFlag = 1;
//	}
//
//	DOUBLE obj_super;
//	DOUBLE obj_unsuper;
//	mahalanobis_unweighted_obj_grad(&obj_unsuper, derivTemp, kernelMatrix, D, DDt, N, K, derivFlag, MD, ObjMat, MDDt);
//	datacpy(deriv, derivTemp, N * N);
//	multihuberhinge_kernel_obj_grad(&obj_super, derivTemp, kernelMatrix, W, X, Y, lambdap, classLabels, \
//			N, numSamples, numTasks, derivFlag, regularizationFlag);
//	INT AXPBYN = SQR(N);
//	DOUBLE alpha = *mu1p;
//	DOUBLE beta = *mu2p;
//	INT incx = 1;
//	INT incy = 1;
//	AXPBY(&AXPBYN, &alpha, derivTemp, &incx, &beta, deriv, &incy);
//	*obj = alpha * obj_super + beta * obj_unsuper;
//
//	if (MDFlag == 1) {
//		FREE(MD);
//	}
//
//	if (ObjMatFlag == 1) {
//		FREE(ObjMat);
//	}
//
//	if (MDDtFlag == 1) {
//		FREE(MDDt);
//	}
//
//	if (derivTempFlag == 1) {
//		FREE(derivTemp);
//	}
//}
