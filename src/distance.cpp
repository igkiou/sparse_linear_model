#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useinterfaces.h"
#include "useblas.h"
#include "distance.h"
#include "utils.h"

DISTANCE_TYPE convertDistanceName(CHAR distanceName) {
	if ((distanceName == 'E') || (distanceName == 'e')) {
		return DISTANCE_L2;
	} else if ((distanceName == 'M') || (distanceName == 'm')) {
		return DISTANCE_MAHALANOBIS;
	} else if ((distanceName == 'K') || (distanceName == 'k')) {
		return DISTANCE_KERNEL;
	} else if ((distanceName == 'L') || (distanceName == 'l')) {
		return DISTANCE_L1;
	} else {
		ERROR("Unknown distance name.");
		return DISTANCE_INVALID;
	}
}

/*
 * Some utils: weighted inner products, norms, and quadratic forms.
 */
DOUBLE dotw(INT *n, DOUBLE *x, INT *incx, DOUBLE *y, INT *incy, DOUBLE *A, \
			DOUBLE *Ax) {

	INT AxFlag = 0;
	if (Ax == NULL) {
		Ax = (DOUBLE *) CMALLOC(*n * 1 * sizeof(DOUBLE));
		AxFlag = 1;
	}

	CHAR uplo = 'U';
	INT SYMVN = *n;
	DOUBLE alpha = 1;
	INT SYMVLDA = *n;
	DOUBLE beta = 0;
	INT incy1 = 1;
	SYMV(&uplo, &SYMVN, &alpha, A, &SYMVLDA, x, incx, &beta, Ax, &incy1);

	INT DOTN = *n;
	DOUBLE result = DOT(&DOTN, Ax, &incy1, y, incy);
	if (AxFlag == 1) {
		CFREE(Ax);
	}
	return result;
}

DOUBLE nrm2w(INT *n, DOUBLE *x, INT *incx, DOUBLE *A, DOUBLE *Ax) {

	INT AxFlag = 0;
	if (Ax == NULL) {
		Ax = (DOUBLE *) CMALLOC(*n * 1 * sizeof(DOUBLE));
		AxFlag = 1;
	}

	CHAR uplo = 'U';
	INT SYMVN = *n;
	DOUBLE alpha = 1;
	INT SYMVLDA = *n;
	DOUBLE beta = 0;
	INT incy1 = 1;
	SYMV(&uplo, &SYMVN, &alpha, A, &SYMVLDA, x, incx, &beta, Ax, &incy1);

	INT DOTN = *n;
	DOUBLE result = DOT(&DOTN, Ax, &incy1, x, incx);
	if (AxFlag == 1) {
		CFREE(Ax);
	}
	return sqrt(result);
}

void quadform(DOUBLE *XtAX, DOUBLE *X, DOUBLE *A, INT M, INT N, DOUBLE alpha, \
		DOUBLE beta, INT transposeFlag, DOUBLE *AX) {

	if (transposeFlag == 0) { // A' * S * A
		INT SAFlag = 0;
		if (AX == NULL) {
			AX = (DOUBLE *) CMALLOC(M * N * sizeof(DOUBLE));
			SAFlag = 1;
		}

		CHAR side = 'L';
		CHAR uplo = 'U';
		INT SYMMM = M;
		INT SYMMN = N;
		DOUBLE alpha2 = 1;
		INT SYMMLDA = M;
		INT SYMMLDB = M;
		DOUBLE beta2 = 0;
		INT SYMMLDC = M;
		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha2, A, &SYMMLDA, X, &SYMMLDB, &beta2, AX, &SYMMLDC);

		CHAR transa = 'T';
		CHAR transb = 'N';
		INT GEMMM = N;
		INT GEMMN = N;
		INT GEMMK = M;
		INT GEMMLDA = M;
		INT GEMMLDB = M;
		INT GEMMLDC = N;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X, &GEMMLDA, AX, &GEMMLDB, &beta, XtAX, &GEMMLDC);

		if (SAFlag == 1) {
			CFREE(AX);
		}
	} else if (transposeFlag == 1) { // A * S * A'
		INT SAFlag = 0;
		if (AX == NULL) {
			AX = (DOUBLE *) CMALLOC(M * N * sizeof(DOUBLE));
			SAFlag = 1;
		}

		CHAR side = 'R';
		CHAR uplo = 'U';
		INT SYMMM = M;
		INT SYMMN = N;
		DOUBLE alpha2 = 1;
		INT SYMMLDA = N;
		INT SYMMLDB = M;
		DOUBLE beta2 = 0;
		INT SYMMLDC = M;
		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha2, A, &SYMMLDA, X, &SYMMLDB, &beta2, AX, &SYMMLDC);

		CHAR transa = 'N';
		CHAR transb = 'T';
		INT GEMMM = M;
		INT GEMMN = M;
		INT GEMMK = N;
		INT GEMMLDA = M;
		INT GEMMLDB = M;
		INT GEMMLDC = M;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, AX, &GEMMLDA, X, &GEMMLDB, &beta, XtAX, &GEMMLDC);

		if (SAFlag == 1) {
			CFREE(AX);
		}
	}
}

// TODO: Write aggregate distance function and vec variant, like in kernel.

/*
 * L2-norm distance.
 */
void l2_distance(DOUBLE *distanceMat, DOUBLE *X1, DOUBLE *X2, INT N, INT numSamples1, INT numSamples2, INT sqrtFlag, \
					DOUBLE *normMat1, DOUBLE *oneVec) {
	
	INT oneArgFlag = 0;
	if (X2 == NULL) {
		oneArgFlag = 1;
		X2 = X1;
		numSamples2 = numSamples1;
	}
	
	INT normMat1Flag = 0;
	if (normMat1 == NULL) {
		normMat1 = (DOUBLE *) MALLOC(numSamples1 * 1 * sizeof(DOUBLE));
		normMat1Flag = 1;
	}
	
	INT oneVecFlag = 0;
	if (oneVec == NULL) {
		oneVec = (DOUBLE *) MALLOC(numSamples1 * 1 * sizeof(DOUBLE));
		oneVecFlag = 1;
	}
	
	DOUBLE alpha;
	DOUBLE beta;
	if (oneArgFlag == 1) {
		CHAR uplo = 'U';
		CHAR trans = 'T';
		INT SYRKN = numSamples1;
		INT SYRKK = N;
		alpha = -2.0;
		INT SYRKLDA = N;
		beta = 0;
		INT SYRKLDC = numSamples1;
		
		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, X1, &SYRKLDA, &beta, distanceMat, &SYRKLDC);
	} else {
		CHAR transa = 'T';
		CHAR transb = 'N';
		INT GEMMM = numSamples1;
		INT GEMMN = numSamples2;
		INT GEMMK = N;
		alpha = -2.0;
		INT GEMMLDA = N;
		INT GEMMLDB = N;
		beta = 0;
		INT GEMMLDC = numSamples1;
		
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X1, &GEMMLDA, X2, &GEMMLDB, &beta, distanceMat, &GEMMLDC);
	}
	
	INT iterX1;
	INT iterX2;
	DOUBLE normTemp;
	INT NRM2N = N;
	INT AXPYN = numSamples1;
	INT incx = 1;
	INT incy = 1;
	/*
	 * TODO: Replace "filling in" of triangular matrix with something more efficient.
	 * TODO: Add flag for when "filling in" is not necessary to skip it.
	 */
	#pragma omp parallel private(iterX1, iterX2, alpha, normTemp) shared(distanceMat, normMat1, oneVec) \
		firstprivate(numSamples1, numSamples2, oneArgFlag, sqrtFlag, incx, incy, AXPYN, NRM2N)
	{
		if (oneArgFlag == 1) {
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				normMat1[iterX1] = - distanceMat[iterX1 * numSamples1 + iterX1] / (DOUBLE) 2.0;
				oneVec[iterX1] = 1;
				for (iterX2 = iterX1 + 1; iterX2 < numSamples1; ++iterX2) {
					distanceMat[iterX1 * numSamples1 + iterX2] = distanceMat[iterX2 * numSamples1 + iterX1];
				}
			}

			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples1; ++iterX2) {
				alpha = 1;
				AXPY(&AXPYN, &alpha, normMat1, &incx, &distanceMat[iterX2 * numSamples1], &incy);
				alpha = normMat1[iterX2];
				AXPY(&AXPYN, &alpha, oneVec, &incx, &distanceMat[iterX2 * numSamples1], &incy);
			}

			if (sqrtFlag == 1) {
				#pragma omp for
				for (iterX2 = 0; iterX2 < numSamples1; ++iterX2) {
					for (iterX1 = 0; iterX1 < iterX2; ++iterX1) {
						distanceMat[iterX2 * numSamples1 + iterX1] = SQRT(distanceMat[iterX2 * numSamples1 + iterX1]);
					}
				}

				#pragma omp for
				for (iterX2 = 0; iterX2 < numSamples1; ++iterX2) {
					for (iterX1 = iterX2 + 1; iterX1 < numSamples1; ++iterX1) {
						distanceMat[iterX2 * numSamples1 + iterX1] = distanceMat[iterX1 * numSamples1 + iterX2];
					}
				}
			}
		} else {
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				normTemp = NRM2(&NRM2N, &X1[N * iterX1], &incx);
				normMat1[iterX1] = SQR(normTemp);
				oneVec[iterX1] = 1;
			}

			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples2; ++iterX2) {
				alpha = 1;
				AXPY(&AXPYN, &alpha, normMat1, &incx, &distanceMat[iterX2 * numSamples1], &incy);
				normTemp = NRM2(&NRM2N, &X2[N * iterX2], &incx);
				alpha = SQR(normTemp);
				AXPY(&AXPYN, &alpha, oneVec, &incx, &distanceMat[iterX2 * numSamples1], &incy);
			}

			if (sqrtFlag == 1) {
				#pragma omp for
				for (iterX2 = 0; iterX2 < numSamples2; ++iterX2) {
					for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
						distanceMat[iterX2 * numSamples1 + iterX1] = SQRT(distanceMat[iterX2 * numSamples1 + iterX1]);
					}
				}
			}
		}
	}

	if (normMat1Flag == 1) {
		FREE(normMat1);
	}
	
	if (oneVecFlag == 1) {
		FREE(oneVec);
	}
}

DOUBLE l2_distance_vec(DOUBLE *x1, DOUBLE *x2, INT N, INT sqrtFlag, \
						DOUBLE *tempVec) {

	INT tempVecFlag = 0;
	if (tempVec == NULL) {
		tempVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		tempVecFlag = 1;
	}
	datacpy(tempVec, x1, N);
	DOUBLE alpha = - 1.0;
	INT incx = 1;
	INT incy = 1;
	AXPY(&N, &alpha, x2, &incx, tempVec, &incy);

	INT NRM2N = N;
	DOUBLE distance = NRM2(&NRM2N, tempVec, &incx);

	if (tempVecFlag == 1) {
		FREE(tempVec);
	}
	if (sqrtFlag == 0) {
		return SQR(distance);
	} else {
		return distance;
	}
}

/*
 * Mahalanobis distance.
 */
void mahalanobis_distance(DOUBLE *distanceMat, DOUBLE *X1, DOUBLE *X2, DOUBLE *A, \
						INT N, INT numSamples1, INT numSamples2, INT sqrtFlag, \
						DOUBLE *tempX1, DOUBLE *tempX2, DOUBLE *normMat1, \
						DOUBLE *oneVec) {

	INT oneArgFlag = 0;
	if (X2 == NULL) {
		oneArgFlag = 1;
		X2 = X1;
		numSamples2 = numSamples1;
	}

	INT normMat1Flag = 0;
	if (normMat1 == NULL) {
		normMat1 = (DOUBLE *) MALLOC(numSamples1 * 1 * sizeof(DOUBLE));
		normMat1Flag = 1;
	}

	INT oneVecFlag = 0;
	if (oneVec == NULL) {
		oneVec = (DOUBLE *) MALLOC(numSamples1 * 1 * sizeof(DOUBLE));
		oneVecFlag = 1;
	}

	INT tempX1Flag = 0;
	if (tempX1 == NULL) {
		tempX1 = (DOUBLE *) MALLOC(N * numSamples1 * sizeof(DOUBLE));
		tempX1Flag = 1;
	}

	INT tempX2Flag = 0;
	if ((tempX2 == NULL) && (oneArgFlag == 0)) {
		tempX2 = (DOUBLE *) MALLOC(N * numSamples2 * sizeof(DOUBLE));
		tempX2Flag = 1;
	}

	DOUBLE alpha;
	DOUBLE beta;
	INT iterX1;
	INT iterX2;
	INT NRM2N = N;
	INT AXPYN = numSamples1;
	INT incx = 1;
	INT incy = 1;
	/*
	 * TODO: Replace "filling in" of triangular matrix with something more efficient.
	 * TODO: Add flag for when "filling in" is not necessary to skip it.
	 */
	if (oneArgFlag == 1) {
		alpha = - 2.0;
		beta = 0.0;
		quadform(distanceMat, X1, A, N, numSamples1, alpha, beta, 0, tempX1);
		#pragma omp parallel private(iterX1, iterX2, alpha) \
				shared(distanceMat, normMat1, oneVec) \
				firstprivate(numSamples1, numSamples2, sqrtFlag, incx, incy, AXPYN, NRM2N)
		{
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				normMat1[iterX1] = - distanceMat[iterX1 * numSamples1 + iterX1] \
								/ (DOUBLE) 2.0;
				oneVec[iterX1] = 1;
				for (iterX2 = iterX1 + 1; iterX2 < numSamples1; ++iterX2) {
					distanceMat[iterX1 * numSamples1 + iterX2] = \
									distanceMat[iterX2 * numSamples1 + iterX1];
				}
			}

			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples1; ++iterX2) {
				alpha = 1;
				AXPY(&AXPYN, &alpha, normMat1, &incx, \
									&distanceMat[iterX2 * numSamples1], &incy);
				alpha = normMat1[iterX2];
				AXPY(&AXPYN, &alpha, oneVec, &incx, \
									&distanceMat[iterX2 * numSamples1], &incy);
			}

			if (sqrtFlag == 1) {
				#pragma omp for
				for (iterX2 = 0; iterX2 < numSamples1; ++iterX2) {
					for (iterX1 = 0; iterX1 < iterX2; ++iterX1) {
						distanceMat[iterX2 * numSamples1 + iterX1] = \
								SQRT(distanceMat[iterX2 * numSamples1 + iterX1]);
					}
				}

				#pragma omp for
				for (iterX2 = 0; iterX2 < numSamples1; ++iterX2) {
					for (iterX1 = iterX2 + 1; iterX1 < numSamples1; ++iterX1) {
						distanceMat[iterX2 * numSamples1 + iterX1] = \
								distanceMat[iterX1 * numSamples1 + iterX2];
					}
				}
			}
		}
	} else {
		CHAR side = 'L';
		CHAR uplo = 'U';
		INT SYMMM = N;
		INT SYMMN = numSamples1;
		alpha = 1.0;
		INT SYMMLDA = N;
		INT SYMMLDB = N;
		beta = 0.0;
		INT SYMMLDC = N;
		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, A, &SYMMLDA, X1, &SYMMLDB, \
				&beta, tempX1, &SYMMLDC);

		SYMMN = numSamples2;
		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, A, &SYMMLDA, X2, &SYMMLDB, \
				&beta, tempX2, &SYMMLDC);

		CHAR transa = 'T';
		CHAR transb = 'N';
		INT GEMMM = numSamples1;
		INT GEMMN = numSamples2;
		INT GEMMK = N;
		alpha = -2.0;
		INT GEMMLDA = N;
		INT GEMMLDB = N;
		beta = 0;
		INT GEMMLDC = numSamples1;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, tempX1, &GEMMLDA, \
				X2, &GEMMLDB, &beta, distanceMat, &GEMMLDC);

		#pragma omp parallel private(iterX1, iterX2, alpha) \
				shared(distanceMat, normMat1, oneVec, tempX1, tempX2) \
				firstprivate(numSamples1, numSamples2, oneArgFlag, sqrtFlag, incx, incy, AXPYN, NRM2N)
		{
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				normMat1[iterX1] = DOT(&NRM2N, &X1[N * iterX1], &incx, \
						&tempX1[N * iterX1], &incy);
				oneVec[iterX1] = 1;
			}

			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples2; ++iterX2) {
				alpha = 1;
				AXPY(&AXPYN, &alpha, normMat1, &incx, \
									&distanceMat[iterX2 * numSamples1], &incy);
				alpha = DOT(&NRM2N, &X2[N * iterX2], &incx, \
										&tempX2[N * iterX2], &incy);
				AXPY(&AXPYN, &alpha, oneVec, &incx, \
									&distanceMat[iterX2 * numSamples1], &incy);
			}

			if (sqrtFlag == 1) {
				#pragma omp for
				for (iterX2 = 0; iterX2 < numSamples2; ++iterX2) {
					for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
						distanceMat[iterX2 * numSamples1 + iterX1] = \
								SQRT(distanceMat[iterX2 * numSamples1 + iterX1]);
					}
				}
			}
		}
	}

	if (normMat1Flag == 1) {
		FREE(normMat1);
	}

	if (oneVecFlag == 1) {
		FREE(oneVec);
	}

	if (tempX1Flag == 1) {
		FREE(tempX1);
	}

	if (tempX2Flag == 1) {
		FREE(tempX2);
	}
}

DOUBLE mahalanobis_distance_vec(DOUBLE *x1, DOUBLE *x2, DOUBLE *A, INT N, \
							INT sqrtFlag, DOUBLE *tempVec, DOUBLE *tempVec2) {

	INT tempVecFlag = 0;
	if (tempVec == NULL) {
		tempVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		tempVecFlag = 1;
	}
	INT tempVec2Flag = 0;
	if (tempVec2 == NULL) {
		tempVec2 = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		tempVec2Flag = 1;
	}
	datacpy(tempVec, x1, N);
	DOUBLE alpha = - 1.0;
	INT incx = 1;
	INT incy = 1;
	AXPY(&N, &alpha, x2, &incx, tempVec, &incy);

	INT NRM2N = N;
	DOUBLE distance = nrm2w(&NRM2N, tempVec, &incx, A, tempVec2);

	if (tempVecFlag == 1) {
		FREE(tempVec);
	}
	if (tempVec2Flag == 1) {
		FREE(tempVec2);
	}
	if (sqrtFlag == 0) {
		return SQR(distance);
	} else {
		return distance;
	}
}

void mahalanobis_distance_factored(DOUBLE *distanceMat, DOUBLE *X1, DOUBLE *X2, \
						DOUBLE *U, INT N, INT numSamples1, INT numSamples2, INT M, \
						INT sqrtFlag, DOUBLE *tempX1, DOUBLE *tempX2, \
						DOUBLE *normMat1, DOUBLE *oneVec) {
	INT tempX1Flag = 0;
	if (tempX1 == NULL) {
		tempX1 = (DOUBLE *) MALLOC(M * numSamples1 * sizeof(DOUBLE));
		tempX1Flag = 1;
	}
	INT tempX2Flag = 0;
	if ((X2 != NULL) && (tempX2 == NULL)) {
		tempX2 = (DOUBLE *) MALLOC(M * numSamples2 * sizeof(DOUBLE));
		tempX2Flag = 1;
	}

	CHAR transa = 'N';
	CHAR transb = 'N';
	INT GEMMM = M;
	INT GEMMN = numSamples1;
	INT GEMMK = N;
	DOUBLE alpha = 1;
	INT GEMMLDA = M;
	INT GEMMLDB = N;
	DOUBLE beta = 0;
	INT GEMMLDC = M;
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, U, &GEMMLDA, X1, \
											&GEMMLDB, &beta, tempX1, &GEMMLDC);

	if (X2 != NULL) {
		GEMMN = numSamples2;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, U, &GEMMLDA, X2, \
											&GEMMLDB, &beta, tempX2, &GEMMLDC);
		l2_distance(distanceMat, tempX1, tempX2, M, numSamples1, numSamples2, \
				sqrtFlag, normMat1, oneVec);
	} else {
		l2_distance(distanceMat, tempX1, NULL, M, numSamples1, 0, \
						sqrtFlag, normMat1, oneVec);
	}

	if (tempX1Flag == 1) {
		FREE(tempX1);
	}
	if (tempX2Flag == 1) {
		FREE(tempX2);
	}
}

void kernel_distance(DOUBLE *distanceMat, DOUBLE *K, INT numSamples, INT sqrtFlag, \
					DOUBLE *normMat, DOUBLE *oneVec) {

	INT normMatFlag = 0;
	if (normMat == NULL) {
		normMat = (DOUBLE *) MALLOC(numSamples * 1 * sizeof(DOUBLE));
		normMatFlag = 1;
	}

	INT oneVecFlag = 0;
	if (oneVec == NULL) {
		oneVec = (DOUBLE *) MALLOC(numSamples * 1 * sizeof(DOUBLE));
		oneVecFlag = 1;
	}

	INT iterX1;
	INT iterX2;
	DOUBLE alpha;
	DOUBLE beta;
	INT AXPYN = numSamples;
	INT AXBPYN = numSamples;
	INT incx = 1;
	INT incy = 1;

	datacpy(distanceMat, K, SQR(numSamples));
	/*
	 * TODO: Replace "filling in" of triangular matrix with something more efficient.
	 * TODO: Add flag for when "filling in" is not necessary to skip it.
	 */
	#pragma omp parallel private(iterX1, iterX2, alpha) shared(distanceMat, normMat, oneVec) \
		firstprivate(numSamples, sqrtFlag, incx, incy, AXPYN)
	{
		#pragma omp for
		for (iterX1 = 0; iterX1 < numSamples; ++iterX1) {
			normMat[iterX1] = distanceMat[iterX1 * numSamples + iterX1];
			oneVec[iterX1] = 1;
		}

		#pragma omp for
		for (iterX2 = 0; iterX2 < numSamples; ++iterX2) {
			alpha = 1.0;
			beta = - 2.0;
			AXPBY(&AXBPYN, &alpha, normMat, &incx, &beta, &distanceMat[iterX2 * numSamples], &incy);
			alpha = normMat[iterX2];
			AXPY(&AXPYN, &alpha, oneVec, &incx, &distanceMat[iterX2 * numSamples], &incy);
		}

		if (sqrtFlag == 1) {
			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples; ++iterX2) {
				for (iterX1 = 0; iterX1 < iterX2; ++iterX1) {
					distanceMat[iterX2 * numSamples + iterX1] = SQRT(distanceMat[iterX2 * numSamples + iterX1]);
				}
			}

			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples; ++iterX2) {
				for (iterX1 = iterX2 + 1; iterX1 < numSamples; ++iterX1) {
					distanceMat[iterX2 * numSamples + iterX1] = distanceMat[iterX1 * numSamples + iterX2];
				}
			}
		}
	}

	if (normMatFlag == 1) {
		FREE(normMat);
	}

	if (oneVecFlag == 1) {
		FREE(oneVec);
	}
}

/*
 * L2-norm distance.
 */
void l1_distance(DOUBLE *distanceMat, DOUBLE *X1, DOUBLE *X2, INT N, INT numSamples1, INT numSamples2) { //, \
//					DOUBLE *tempVec) {

	INT oneArgFlag = 0;
	if (X2 == NULL) {
		oneArgFlag = 1;
		X2 = X1;
		numSamples2 = numSamples1;
	}

	/*INT tempVecFlag = 0;
	if (tempVec == NULL) {
		tempVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		tempVecFlag = 1;
	}*/
	DOUBLE *tempVec;
	DOUBLE alpha = - 1.0;
	INT iterX1;
	INT iterX2;
	INT ASUMN = N;
	INT AXPYN = N;
	INT incx = 1;
	INT incy = 1;
	/*
	 * TODO: Replace "filling in" of triangular matrix with something more efficient.
	 * TODO: Add flag for when "filling in" is not necessary to skip it.
	 */
	#pragma omp parallel private(iterX1, iterX2, tempVec) shared(distanceMat) \
		firstprivate(numSamples1, numSamples2, oneArgFlag, incx, incy, AXPYN, ASUMN)
	{
		tempVec = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		if (oneArgFlag == 1) {
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				distanceMat[iterX1 * numSamples1 + iterX1] = 0;
				for (iterX2 = iterX1 + 1; iterX2 < numSamples1; ++iterX2) {
					datacpy(tempVec, &X1[iterX1 * N], N);
					AXPY(&AXPYN, &alpha, &X1[iterX2 * N], &incx, tempVec, &incy);
					distanceMat[iterX1 * numSamples1 + iterX2] = ASUM(&ASUMN, tempVec, &incx);
				}
			}

			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples1; ++iterX2) {
				for (iterX1 = iterX2 + 1; iterX1 < numSamples1; ++iterX1) {
					distanceMat[iterX1 * numSamples1 + iterX2] = distanceMat[iterX2 * numSamples1 + iterX1];
				}
			}

		} else {
			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples2; ++iterX2) {
				for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
					datacpy(tempVec, &X1[iterX1 * N], N);
					AXPY(&AXPYN, &alpha, &X2[iterX2 * N], &incx, tempVec, &incy);
					distanceMat[iterX2 * numSamples1 + iterX1] = ASUM(&ASUMN, tempVec, &incx);
				}
			}
		}
		CFREE(tempVec);
	}

	/*if (tempVecFlag == 1) {
		FREE(tempVec);
	}*/
}

DOUBLE l1_distance_vec(DOUBLE *x1, DOUBLE *x2, INT N, DOUBLE *tempVec) {

	INT tempVecFlag = 0;
	if (tempVec == NULL) {
		tempVec = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		tempVecFlag = 1;
	}
	datacpy(tempVec, x1, N);
	DOUBLE alpha = - 1.0;
	INT incx = 1;
	INT incy = 1;
	AXPY(&N, &alpha, x2, &incx, tempVec, &incy);

	INT ASUMN = N;
	DOUBLE distance = ASUM(&ASUMN, tempVec, &incx);

	if (tempVecFlag == 1) {
		FREE(tempVec);
	}

	return distance;
}
