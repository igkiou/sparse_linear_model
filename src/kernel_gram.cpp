#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useinterfaces.h"
#include "useblas.h"
#include "kernel_gram.h"
#include "distance.h"

KERNEL_TYPE convertKernelName(CHAR kernelName) {
	if ((kernelName == 'L') || (kernelName == 'l')) {
		return KERNEL_LINEAR;
	} else if ((kernelName == 'G') || (kernelName == 'g')) {
		return KERNEL_GAUSSIAN;
	} else if ((kernelName == 'P') || (kernelName == 'p')) {
		return KERNEL_POLYNOMIAL;
	} else if ((kernelName == 'H') || (kernelName == 'h')) {
		return KERNEL_SOBOLEV;
	} else {
		ERROR("Unknown kernel name.");
		return KERNEL_INVALID;
	}
}

/* TODO: Parallelize these using CUDA kernels. */
void kernel_gram(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
			INT numSamples1, INT numSamples2, KERNEL_TYPE kernelType, \
			DOUBLE *pparam1, DOUBLE *pparam2, DOUBLE *normMat1, \
			DOUBLE *oneVec) {

	if (kernelType == KERNEL_LINEAR) {
		kernel_gram_linear(kernelMat, X1, X2, N, numSamples1, numSamples2);
	} else if (kernelType == KERNEL_GAUSSIAN) {
		kernel_gram_gaussian(kernelMat, X1, X2, N, numSamples1, \
							numSamples2, pparam1, normMat1, oneVec);
	} else if (kernelType == KERNEL_POLYNOMIAL) {
		kernel_gram_polynomial(kernelMat, X1, X2, N, numSamples1, \
							numSamples2, pparam1, pparam2);
	} else if (kernelType == KERNEL_SOBOLEV) {
		kernel_gram_sobolev(kernelMat, X1, X2, N, numSamples1, \
							numSamples2, pparam1);
	} else {
		ERROR("Unknown kernel function.");
	}
}

void kernel_gram_linear(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
					INT numSamples1, INT numSamples2) {

	INT oneArgFlag = 0;
	if (X2 == NULL) {
		oneArgFlag = 1;
	}

	INT iterX1;
	INT iterX2;
	DOUBLE alpha;
	DOUBLE beta;
	if (oneArgFlag == 1) {
		CHAR uplo = 'U';
		CHAR trans = 'T';
		INT SYRKN = numSamples1;
		INT SYRKK = N;
		alpha = 1.0;
		INT SYRKLDA = N;
		beta = 0;
		INT SYRKLDC = numSamples1;

		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, X1, &SYRKLDA, &beta, \
				kernelMat, &SYRKLDC);
		#pragma omp parallel for private(iterX1, iterX2) shared(kernelMat) \
			firstprivate(numSamples1)
		for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
			for (iterX2 = iterX1 + 1; iterX2 < numSamples1; ++iterX2) {
				kernelMat[iterX1 * numSamples1 + iterX2] = \
									kernelMat[iterX2 * numSamples1 + iterX1];
			}
		}
	} else {
		CHAR transa = 'T';
		CHAR transb = 'N';
		INT GEMMM = numSamples1;
		INT GEMMN = numSamples2;
		INT GEMMK = N;
		alpha = 1.0;
		INT GEMMLDA = N;
		INT GEMMLDB = N;
		beta = 0;
		INT GEMMLDC = numSamples1;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X1, &GEMMLDA, \
				X2, &GEMMLDB, &beta, kernelMat, &GEMMLDC);
	}
}

void kernel_gram_gaussian(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
			INT numSamples1, INT numSamples2, DOUBLE *pparam1, \
			DOUBLE *normMat1, DOUBLE *oneVec) {

	INT oneArgFlag = 0;
	if (X2 == NULL) {
		oneArgFlag = 1;
	}

	INT iterX1;
	INT iterX2;
	DOUBLE param1;
	if (pparam1 == NULL) {
		param1 = 1.0;
	} else {
		param1 = *pparam1;
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
	l2_distance(kernelMat, X1, X2, N, numSamples1, numSamples2, 0, normMat1, \
				oneVec);
	if (normMat1Flag == 1) {
		FREE(normMat1);
	}
	if (oneVecFlag == 1) {
		FREE(oneVec);
	}

	DOUBLE denom = 2.0 * SQR(param1);
	#pragma omp parallel private(iterX1, iterX2) shared(kernelMat, denom) \
		firstprivate(numSamples1, numSamples2, oneArgFlag)
	{
		if (oneArgFlag == 1) {
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				for (iterX2 = 0; iterX2 < iterX1; ++iterX2) {
					kernelMat[iterX1 * numSamples1 + iterX2] = \
						EXP(- kernelMat[iterX1 * numSamples1 + iterX2] / denom);
				}
				kernelMat[iterX1 * numSamples1 + iterX1] = 1;
			}
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				for (iterX2 = iterX1 + 1; iterX2 < numSamples1; ++iterX2) {
					kernelMat[iterX1 * numSamples1 + iterX2] = \
							kernelMat[iterX2 * numSamples1 + iterX1];
				}
			}
		} else {
			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples2; ++iterX2) {
				for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
					kernelMat[iterX2 * numSamples1 + iterX1] = \
						EXP(- kernelMat[iterX2 * numSamples1 + iterX1] / denom);
				}
			}
		}
	}
}

void kernel_gram_polynomial(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
			INT numSamples1, INT numSamples2, DOUBLE *pparam1, \
			DOUBLE *pparam2) {

	INT oneArgFlag = 0;
	if (X2 == NULL) {
		oneArgFlag = 1;
	}

	INT iterX1;
	INT iterX2;
	DOUBLE param1;
	DOUBLE param2;
	DOUBLE alpha;
	DOUBLE beta;

	if (pparam1 == NULL) {
		param1 = 0;
	} else {
		param1 = *pparam1;
	}
	if (pparam2 == NULL) {
		param2 = 3.0;
	} else {
		param2 = *pparam2;
	}

	if (oneArgFlag == 1) {
		CHAR uplo = 'U';
		CHAR trans = 'T';
		INT SYRKN = numSamples1;
		INT SYRKK = N;
		alpha = 1.0;
		INT SYRKLDA = N;
		beta = 0;
		INT SYRKLDC = numSamples1;

		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, X1, &SYRKLDA, &beta, \
				kernelMat, &SYRKLDC);
		#pragma omp parallel private(iterX1, iterX2) shared(kernelMat, param1, param2) \
			firstprivate(numSamples1)
		{
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				for (iterX2 = 0; iterX2 < iterX1 + 1; ++iterX2) {
					kernelMat[iterX1 * numSamples1 + iterX2] = \
						POW(kernelMat[iterX1 * numSamples1 + iterX2] + param1, \
						param2);
				}
			}

			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				for (iterX2 = iterX1 + 1; iterX2 < numSamples1; ++iterX2) {
					kernelMat[iterX1 * numSamples1 + iterX2] = \
						kernelMat[iterX2 * numSamples1 + iterX1];
				}
			}
		}
	} else {
		CHAR transa = 'T';
		CHAR transb = 'N';
		INT GEMMM = numSamples1;
		INT GEMMN = numSamples2;
		INT GEMMK = N;
		alpha = 1.0;
		INT GEMMLDA = N;
		INT GEMMLDB = N;
		beta = 0;
		INT GEMMLDC = numSamples1;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X1, &GEMMLDA, \
				X2, &GEMMLDB, &beta, kernelMat, &GEMMLDC);
		#pragma omp parallel for private(iterX1, iterX2) shared(kernelMat, param1, param2) \
			firstprivate(numSamples1, numSamples2)
		for (iterX2 = 0; iterX2 < numSamples2; ++iterX2) {
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				kernelMat[iterX2 * numSamples1 + iterX1] = \
						POW(kernelMat[iterX2 * numSamples1 + iterX1] + param1, \
						param2);
			}
		}
	}
}

void kernel_gram_sobolev(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
			INT numSamples1, INT numSamples2, DOUBLE *pparam1) {

	INT oneArgFlag = 0;
	if (X2 == NULL) {
		oneArgFlag = 1;
	}

	INT iterX1;
	INT iterX2;
	DOUBLE param1;
	if (pparam1 == NULL) {
		param1 = 1.0;
	} else {
		param1 = *pparam1;
	}

	l1_distance(kernelMat, X1, X2, N, numSamples1, numSamples2);

	DOUBLE denom = param1;
	#pragma omp parallel private(iterX1, iterX2) shared(kernelMat, denom) \
		firstprivate(numSamples1, numSamples2, oneArgFlag)
	{
		if (oneArgFlag == 1) {
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				for (iterX2 = 0; iterX2 < iterX1; ++iterX2) {
					kernelMat[iterX1 * numSamples1 + iterX2] = \
						EXP(- kernelMat[iterX1 * numSamples1 + iterX2] / denom);
				}
				kernelMat[iterX1 * numSamples1 + iterX1] = 1;
			}
			#pragma omp for
			for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
				for (iterX2 = iterX1 + 1; iterX2 < numSamples1; ++iterX2) {
					kernelMat[iterX1 * numSamples1 + iterX2] = \
						kernelMat[iterX2 * numSamples1 + iterX1];
				}
			}
		} else {
			#pragma omp for
			for (iterX2 = 0; iterX2 < numSamples2; ++iterX2) {
				for (iterX1 = 0; iterX1 < numSamples1; ++iterX1) {
					kernelMat[iterX2 * numSamples1 + iterX1] = \
						EXP(- kernelMat[iterX2 * numSamples1 + iterX1] / denom);
				}
			}
		}
	}
}
