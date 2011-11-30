#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "useblas.h"
#include "useinterfaces.h"
#include "svm_optimization.h"
#include "matrix_proximal.h"
#include "utils.h"

void squaredhinge_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *wb, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
			DOUBLE *kernelMatrix, INT M, INT numSamples, INT biasFlag, INT derivFlag, INT regularizationFlag, \
			DOUBLE *Ypred, DOUBLE *KX) {
	
	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}

	INT KXFlag = 0;
	if ((KX == NULL) && (kernelMatrix != NULL)) {
		KX = (DOUBLE *) MALLOC(M * numSamples * sizeof(DOUBLE));
		KXFlag = 1;
	}

	DOUBLE *Xp;
	if (kernelMatrix != NULL) {
		CHAR side = 'L';
		CHAR uplo = 'U';
		INT SYMMM = M;
		INT SYMMN = numSamples;
		DOUBLE alpha = 1;
		INT SYMMLDA = M;
		INT SYMMLDB = M;
		DOUBLE beta = 0;
		INT SYMMLDC = M;

		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, kernelMatrix, &SYMMLDA, X, &SYMMLDB, &beta, KX, &SYMMLDC);
		Xp = KX;
	} else {
		Xp = X;
	}

	DOUBLE *weights = wb;
	DOUBLE biasterm = wb[M];
	DOUBLE lambda = * lambdap;
	
	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = numSamples;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Xp, &GEMVLDA, weights, &incx, &beta, Ypred, &incy);

	INT iterX;
	DOUBLE YYpred;
	DOUBLE biasderiv = 0;
	DOUBLE objtemp = 0;
	
	INT AXPYN = M;
	
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * sizeof(DOUBLE));
	}
	
	for (iterX = 0; iterX < numSamples; ++iterX) {
		
		YYpred = Y[iterX] * (Ypred[iterX] + biasterm);
		
		if (YYpred < 1) {
			
			objtemp += SQR(1 - YYpred);
			
			if (derivFlag == 1) {
				alpha = 2 * (Ypred[iterX] + biasterm - Y[iterX]);
				AXPY(&AXPYN, &alpha, &Xp[iterX * M], &incx, deriv, &incy);

				if (biasFlag == 1) {
					biasderiv += alpha;
				}
			}
		}
	}
	*obj = objtemp;
	if (derivFlag == 1) {
		deriv[M] = biasderiv;
	}
	
	if (regularizationFlag == 1) {
		if (kernelMatrix != NULL) {
			CHAR uplo = 'U';
			INT SYMVN = M;
			alpha = 1;
			INT SYMVLDA = M;
			beta = 0;
			incx = 1;
			incy = 1;
			SYMV(&uplo, &SYMVN, &alpha, kernelMatrix, &SYMVLDA, weights, &incx, &beta, KX, &incy);

			INT DOTN = M;
			*obj = objtemp + lambda * DOT(&DOTN, weights, &incx, KX, &incy);

			if (derivFlag == 1) {
				alpha = 2 * lambda;
				AXPY(&AXPYN, &alpha, KX, &incx, deriv, &incy);
			}
		} else {
			DOUBLE normtemp;
			INT NRM2N = M;
			normtemp = NRM2(&NRM2N, weights, &incx);
			*obj = objtemp + lambda * SQR(normtemp);

			if (derivFlag == 1) {
				alpha = 2 * lambda;
				AXPY(&AXPYN, &alpha, weights, &incx, deriv, &incy);
			}
		}
	}

	if (YpredFlag == 1) {
		FREE(Ypred);
	}

	if (KXFlag == 1) {
		FREE(KX);
	}
}

void huberhinge_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *wb, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
			DOUBLE *kernelMatrix, INT M, INT numSamples, INT biasFlag, INT derivFlag, INT regularizationFlag, \
			DOUBLE *Ypred, DOUBLE *KX) {
	
	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}

	INT KXFlag = 0;
	if ((KX == NULL) && (kernelMatrix != NULL)) {
		KX = (DOUBLE *) MALLOC(M * numSamples * sizeof(DOUBLE));
		KXFlag = 1;
	}

	DOUBLE *Xp;
	if (kernelMatrix != NULL) {
		CHAR side = 'L';
		CHAR uplo = 'U';
		INT SYMMM = M;
		INT SYMMN = numSamples;
		DOUBLE alpha = 1;
		INT SYMMLDA = M;
		INT SYMMLDB = M;
		DOUBLE beta = 0;
		INT SYMMLDC = M;

		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, kernelMatrix, &SYMMLDA, X, &SYMMLDB, &beta, KX, &SYMMLDC);
		Xp = KX;
	} else {
		Xp = X;
	}

	DOUBLE *weights = wb;
	DOUBLE biasterm = wb[M];
	DOUBLE lambda = * lambdap;
	
	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = numSamples;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Xp, &GEMVLDA, weights, &incx, &beta, Ypred, &incy);
	INT iterX;
	DOUBLE YYpred;
	DOUBLE biasderiv = 0;
	DOUBLE objtemp = 0;
	
	INT AXPYN = M;
	
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * sizeof(DOUBLE));
	}
	
	/*
	 * NOTE: Faster without omp, probably due to the critical parts. Maybe change
	 * this to a large multiplication? Many zeros (no special function for multiplication
	 * of dense matrix with sparse vector.)
	 */
//	#pragma omp parallel for private(iterX, alpha, YYpred) shared(Xp, Y, Ypred, deriv, biasterm) \
		reduction(+: objtemp, biasderiv) firstprivate(numSamples, AXPYN, incx, incy, derivFlag)
	for (iterX = 0; iterX < numSamples; ++iterX) {
		
		YYpred = Y[iterX] * (Ypred[iterX] + biasterm);
		if (YYpred < -1) {
			
			objtemp += - 4.0 * YYpred;
	
			if (derivFlag == 1) {
				alpha = - 4.0 * Y[iterX];
//				#pragma omp critical
				AXPY(&AXPYN, &alpha, &Xp[iterX * M], &incx, deriv, &incy);
			
				if (biasFlag == 1) {
					biasderiv += alpha;
				}
			}
		} else if ((YYpred < 1.0) && (YYpred >= - 1.0)) {
			objtemp += SQR(1.0 - YYpred);
			
			if (derivFlag == 1) {
				alpha = 2.0 * (Ypred[iterX] + biasterm - Y[iterX]);
//				#pragma omp critical
				AXPY(&AXPYN, &alpha, &Xp[iterX * M], &incx, deriv, &incy);

				if (biasFlag == 1) {
					biasderiv += alpha;
				}
			}
		}
	}
	*obj = objtemp;
	if (derivFlag == 1) {
		deriv[M] = biasderiv;
	}

	if (regularizationFlag == 1) {
		if (kernelMatrix != NULL) {
			CHAR uplo = 'U';
			INT SYMVN = M;
			alpha = 1;
			INT SYMVLDA = M;
			beta = 0;
			incx = 1;
			incy = 1;
			SYMV(&uplo, &SYMVN, &alpha, kernelMatrix, &SYMVLDA, weights, &incx, &beta, KX, &incy);

			INT DOTN = M;
			*obj = objtemp + lambda * DOT(&DOTN, weights, &incx, KX, &incy);

			if (derivFlag == 1) {
				alpha = 2 * lambda;
				AXPY(&AXPYN, &alpha, KX, &incx, deriv, &incy);
			}
		} else {
			DOUBLE normtemp;
			INT NRM2N = M;
			normtemp = NRM2(&NRM2N, weights, &incx);
			*obj = objtemp + lambda * SQR(normtemp);

			if (derivFlag == 1) {
				alpha = 2 * lambda;
				AXPY(&AXPYN, &alpha, weights, &incx, deriv, &incy);
			}
		}
	}

	if (YpredFlag == 1) {
		FREE(Ypred);
	}

	if (KXFlag == 1) {
		FREE(KX);
	}
}

// TODO: Merge alt with regular versions and edit mex files appropriately.
void huberhinge_obj_grad_alt(DOUBLE *obj, DOUBLE *deriv, DOUBLE *w, DOUBLE *X, DOUBLE *Y, INT taskLabel, \
		DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, INT regularizationFlag, DOUBLE *Ypred) {

	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}

	DOUBLE *weights = w;
	DOUBLE lambda = *lambdap;

	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = numSamples;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;

	GEMV(&trans, &GEMVM, &GEMVN, &alpha, X, &GEMVLDA, weights, &incx, &beta, Ypred, &incy);
	INT iterX;
	DOUBLE YYpred;
	DOUBLE Ytask;
	DOUBLE objtemp = 0;

	INT AXPYN = M;
	for (iterX = 0; iterX < numSamples; ++iterX) {

		Ytask = ((INT) Y[iterX] == taskLabel) ? (1) : (- 1);
		YYpred = Ytask * Ypred[iterX];
		if (YYpred < -1) {
			objtemp += - 4.0 * YYpred;
			if (derivFlag == 1) {
				alpha = - 4.0 * Ytask;
				AXPY(&AXPYN, &alpha, &X[iterX * M], &incx, deriv, &incy);
			}
		} else if ((YYpred < 1.0) && (YYpred >= - 1.0)) {
			objtemp += SQR(1.0 - YYpred);
			if (derivFlag == 1) {
				alpha = 2.0 * (Ypred[iterX] - Ytask);
				AXPY(&AXPYN, &alpha, &X[iterX * M], &incx, deriv, &incy);
			}
		}
	}
	*obj = objtemp;

	if (regularizationFlag == 1) {
		DOUBLE normtemp;
		INT NRM2N = M;
		normtemp = NRM2(&NRM2N, weights, &incx);
		*obj += lambda * SQR(normtemp);

		if (derivFlag == 1) {
			alpha = 2 * lambda;
			AXPY(&AXPYN, &alpha, weights, &incx, deriv, &incy);
		}
	}

	if (YpredFlag == 1) {
		FREE(Ypred);
	}
}

void multihuberhinge_obj_grad_memory(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
				DOUBLE *classLabels, INT M, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag) {

	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * numTasks * sizeof(DOUBLE));
	}

	INT iterT;
	DOUBLE *Ypred;
	DOUBLE lambda = *lambdap;
	INT taskLabel;

	#pragma omp parallel private(Ypred, taskLabel, iterT) shared(obj, deriv, W, X, Y, classLabels) \
		firstprivate(numTasks, M, numSamples, derivFlag, regularizationFlag, lambda)
	{
		Ypred = (DOUBLE *) CMALLOC(1 * numSamples * sizeof(DOUBLE));
		#pragma omp for
		for (iterT = 0; iterT < numTasks; ++iterT) {
			taskLabel = (INT) classLabels[iterT];
			huberhinge_obj_grad_alt(&obj[iterT], &deriv[iterT * M], &W[iterT * M], X, \
					Y, taskLabel, &lambda, M, numSamples, derivFlag, regularizationFlag, Ypred);
		}
		CFREE(Ypred);
	}
}

// TODO: Implement with matrix instead of many AXPY. Perhaps sparse matrix as well.
void multihuberhinge_obj_grad_speed(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
		 	 DOUBLE *classLabels, INT M, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag, \
		 	 DOUBLE *Ypred) {

	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(numSamples * numTasks * sizeof(DOUBLE));
		YpredFlag = 1;
	}

	DOUBLE lambda = * lambdap;

	CHAR transa = 'T';
	CHAR transb = 'N';
	INT GEMMM = numSamples;
	INT GEMMN = numTasks;
	INT GEMMK = M;
	DOUBLE alpha = 1;
	INT GEMMLDA = M;
	INT GEMMLDB = M;
	DOUBLE beta = 0;
	INT GEMMLDC = numSamples;
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X, &GEMMLDA, W, &GEMMLDB, &beta, Ypred, &GEMMLDC);

	memset((void *) obj, 0, numTasks * sizeof(DOUBLE));
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * numTasks * sizeof(DOUBLE));
	}

	INT iterT;
	INT iterX;
	INT taskLabel;
	DOUBLE YYpred;
	INT AXPYN = M;
	DOUBLE normtemp;
	DOUBLE Ytask;
	INT NRM2N = M;
	INT incx = 1;
	INT incy = 1;

	#pragma omp parallel for \
		private(iterT, iterX, YYpred, alpha, normtemp, Ytask, taskLabel) \
		shared(obj, W, deriv, X, Y, Ypred, classLabels) \
		firstprivate(numSamples, numTasks, AXPYN, NRM2N, incx, incy, regularizationFlag, lambda)
	for (iterT = 0; iterT < numTasks; ++iterT) {
		taskLabel = (INT) classLabels[iterT];
		for (iterX = 0; iterX < numSamples; ++iterX) {
			Ytask = ((INT) Y[iterX] == taskLabel) ? (1) : (- 1);
			YYpred = Ytask * Ypred[iterT * numSamples + iterX];
			if (YYpred < -1) {
				obj[iterT] += - 4.0 * YYpred;
				if (derivFlag == 1) {
					alpha = - 4.0 * Ytask;
					AXPY(&AXPYN, &alpha, &X[iterX * M], &incx, &deriv[M * iterT], &incy);
				}
			} else if ((YYpred < 1.0) && (YYpred >= - 1.0)) {
				obj[iterT] += SQR(1.0 - YYpred);
				if (derivFlag == 1) {
					alpha = 2.0 * (Ypred[iterT * numSamples + iterX] - Ytask);
					AXPY(&AXPYN, &alpha, &X[iterX * M], &incx, &deriv[M * iterT], &incy);
				}
			}
		}
		if (regularizationFlag == 1) {
			normtemp = NRM2(&NRM2N, &W[M * iterT], &incx);
			obj[iterT] += lambda * SQR(normtemp);
			if (derivFlag == 1) {
				alpha = 2 * lambda;
				AXPY(&AXPYN, &alpha, &W[M * iterT], &incx, &deriv[M * iterT], &incy);
			}
		}
	}

	if (YpredFlag == 1) {
		FREE(Ypred);
	}
}

void cramersinger_approx_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
					INT N, INT numSamples, INT numTasks, INT derivFlag, DOUBLE *oneVec, DOUBLE *expMat, DOUBLE *logArg) {

	INT oneVecFlag = 0;
	if (oneVec == NULL) {
		oneVec = (DOUBLE *) MALLOC(1 * numTasks * sizeof(DOUBLE));
		oneVecFlag = 1;
	}

	INT expMatFlag = 0;
	if (expMat == NULL) {
		expMat = (DOUBLE *) MALLOC(numTasks * numSamples * sizeof(DOUBLE));
		expMatFlag = 1;
	}

	INT logArgFlag = 0;
	if (logArg == NULL) {
		logArg = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		logArgFlag = 1;
	}

	DOUBLE lambda = * lambdap;
	DOUBLE oneMinusExpLambda = 1 - EXP(lambda);

	CHAR transa = 'T';
	CHAR transb = 'N';
	INT GEMMM = numTasks;
	INT GEMMN = numSamples;
	INT GEMMK = N;
	DOUBLE alpha = 1;
	INT GEMMLDA = N;
	INT GEMMLDB = N;
	DOUBLE beta = 0;
	INT GEMMLDC = numTasks;
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, W, &GEMMLDA, X, &GEMMLDB, &beta, expMat, &GEMMLDC);

	INT iterT;
	for (iterT = 0; iterT < numTasks; ++iterT) {
		oneVec[iterT] = 1;
	}

	DOUBLE objTemp = 0;
	INT AXPYN = numTasks;
	INT SCALN = numTasks;
	INT incx = 1;
	INT incy = 1;
	INT iterX;

	#pragma omp parallel for private(iterX, iterT, alpha) shared(logArg, expMat, X, Y, oneVec) \
		reduction(+: objTemp) firstprivate(N, numSamples, numTasks, AXPYN, SCALN, incx, incy, lambda, oneMinusExpLambda, derivFlag)
	for (iterX = 0; iterX < numSamples; ++iterX) {
		alpha = - expMat[numTasks * iterX + (INT) Y[iterX] - 1];
		AXPY(&AXPYN, &alpha, oneVec, &incx, &expMat[numTasks * iterX], &incy);
		logArg[iterX] = 0;
		for (iterT = 0; iterT < numTasks; ++iterT) {
			expMat[numTasks * iterX + iterT] = EXP(lambda * (expMat[numTasks * iterX + iterT] + 1));
			logArg[iterX] += expMat[numTasks * iterX + iterT];
		}
		logArg[iterX] = logArg[iterX] + oneMinusExpLambda;
		objTemp += LOG(logArg[iterX]);
		if (derivFlag == 1) {
			alpha = 1 / logArg[iterX];
			SCAL(&SCALN, &alpha, &expMat[numTasks * iterX], &incx);
			expMat[numTasks * iterX + (INT) Y[iterX] - 1] = - alpha * (logArg[iterX] - 1);
		}
	}
	objTemp = objTemp / lambda;
	*obj = objTemp / (DOUBLE) numSamples;

	if (derivFlag == 1) {
		transa = 'N';
		transb = 'T';
		GEMMM = N;
		GEMMN = numTasks;
		GEMMK = numSamples;
		alpha = 1 / (DOUBLE) numSamples;
		GEMMLDA = N;
		GEMMLDB = numTasks;
		beta = 0;
		GEMMLDC = N;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X, &GEMMLDA, expMat, &GEMMLDB, &beta, deriv, &GEMMLDC);
	}

	if (oneVecFlag == 1) {
		FREE(oneVec);
	}

	if (expMatFlag == 1) {
		FREE(expMat);
	}

	if (logArgFlag == 1) {
		FREE(logArg);
	}
}

void cramersinger_nuclear_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *gammap, \
					DOUBLE *rhop, DOUBLE *lambdap, INT N, INT numSamples, INT numTasks, INT derivFlag, \
					DOUBLE *oneVecSvdVec, DOUBLE *expMatVtMat, DOUBLE *logArgDerivVec, DOUBLE *dataBuffer, \
					DOUBLE *work, INT lwork) {

	if (lwork == -1) {
		CHAR jobu;
		CHAR jobvt;
		if (derivFlag == 1) {
			jobu = 'O';
			jobvt = 'S';
		} else {
			jobu = 'N';
			jobvt = 'N';
		}
		INT MN = IMIN(N, numTasks);
		INT GESVDM = N;
		INT GESVDN = numTasks;
		INT GESVDLDA = N;
		INT GESVDLDU = N;
		INT GESVDLDVT = MN;
		INT INFO;

		GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, NULL, &GESVDLDA, NULL, NULL, &GESVDLDU, NULL, &GESVDLDVT, work, &lwork, &INFO);
		return;
	}

	INT oneVecSvdVecFlag = 0;
	if (oneVecSvdVec == NULL) {
		oneVecSvdVec = (DOUBLE *) MALLOC(1 * numTasks * sizeof(DOUBLE));
		oneVecSvdVecFlag = 1;
	}

	INT minNNumTasks = IMIN(N, numTasks);
	INT maxNumSamplesMinNNumTasks = IMAX(numSamples, minNNumTasks);
	INT logArgDerivVecFlag = 0;
	if ((logArgDerivVec == NULL) && (derivFlag == 1)) {
		logArgDerivVec = (DOUBLE *) MALLOC(1 * maxNumSamplesMinNNumTasks * sizeof(DOUBLE));
		logArgDerivVecFlag = 1;
	}

	INT expMatVtMatFlag = 0;
	if ((expMatVtMat == NULL) && (derivFlag == 1)) {
		expMatVtMat = (DOUBLE *) MALLOC(maxNumSamplesMinNNumTasks * numTasks * sizeof(DOUBLE));
		expMatVtMatFlag = 1;
	}

	INT dataBufferFlag = 0;
	if (dataBuffer == NULL) {
		dataBuffer = (DOUBLE *) MALLOC(N * numTasks * sizeof(DOUBLE));
		dataBufferFlag = 1;
	}

	INT workFlag = 0;
	if (work == NULL) {
		workFlag = 1;
	}

	DOUBLE obj1;
	DOUBLE obj2;
	nuclear_approx_obj_grad(&obj1, deriv, W, rhop, N, numTasks, derivFlag, oneVecSvdVec, expMatVtMat, dataBuffer, logArgDerivVec, work, lwork);
	datacpy(dataBuffer, deriv, N * numTasks);
	cramersinger_approx_obj_grad(&obj2, deriv, W, X, Y, gammap, N, numSamples, numTasks, derivFlag, oneVecSvdVec, expMatVtMat, logArgDerivVec);
	*obj = obj2 + *lambdap * obj1;

	if (derivFlag == 1) {
		INT AXPYN = N * numTasks;
		DOUBLE alpha = *lambdap;
		INT incx = 1;
		INT incy = 1;
		AXPY(&AXPYN, &alpha, dataBuffer, &incx, deriv, &incy);
	}

	if (oneVecSvdVecFlag == 1) {
		FREE(oneVecSvdVec);
	}

	if (logArgDerivVecFlag == 1) {
		FREE(logArgDerivVec);
	}

	if (expMatVtMatFlag == 1) {
		FREE(expMatVtMat);
	}

	if (dataBufferFlag == 1) {
		FREE(dataBuffer);
	}

	if (workFlag == 1) {
		FREE(work);
	}
}

void cramersinger_frobenius_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *gammap, \
					DOUBLE *lambdap, INT N, INT numSamples, INT numTasks, INT derivFlag, \
					DOUBLE *oneVec, DOUBLE *expMat, DOUBLE *logArg) {

	INT oneVecFlag = 0;
	if (oneVec == NULL) {
		oneVec = (DOUBLE *) MALLOC(1 * numTasks * sizeof(DOUBLE));
		oneVecFlag = 1;
	}

	INT expMatFlag = 0;
	if (expMat == NULL) {
		expMat = (DOUBLE *) MALLOC(numTasks * numSamples * sizeof(DOUBLE));
		expMatFlag = 1;
	}

	INT logArgFlag = 0;
	if (logArg == NULL) {
		logArg = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		logArgFlag = 1;
	}

	DOUBLE obj2;
	cramersinger_approx_obj_grad(&obj2, deriv, W, X, Y, gammap, N, numSamples, numTasks, derivFlag, oneVec, expMat, logArg);
	CHAR norm = 'F';
	INT LANGEM = N;
	INT LANGEN = numTasks;
	INT LANGELDA = N;
	DOUBLE obj1 = LANGE(&norm, &LANGEM, &LANGEN, W, &LANGELDA, NULL);
	obj1 = SQR(obj1);
	*obj = obj2 + *lambdap * obj1;

	if (derivFlag == 1) {
		INT AXPYN = N * numTasks;
		DOUBLE alpha = *lambdap * 2.0;
		INT incx = 1;
		INT incy = 1;
		AXPY(&AXPYN, &alpha, W, &incx, deriv, &incy);
	}

	if (oneVecFlag == 1) {
		FREE(oneVec);
	}

	if (expMatFlag == 1) {
		FREE(expMat);
	}

	if (logArgFlag == 1) {
		FREE(logArg);
	}
}

void multihuberhinge_nuclear_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, \
					DOUBLE *rhop, DOUBLE *lambdap, DOUBLE *classLabels, INT N, INT numSamples, INT numTasks, \
					INT derivFlag, DOUBLE *YpredVtMat, DOUBLE *svdVec, DOUBLE *derivVec, DOUBLE *dataBuffer, \
					DOUBLE *work, INT lwork) {

	if (lwork == -1) {
		CHAR jobu;
		CHAR jobvt;
		if (derivFlag == 1) {
			jobu = 'O';
			jobvt = 'S';
		} else {
			jobu = 'N';
			jobvt = 'N';
		}
		INT MN = IMIN(N, numTasks);
		INT GESVDM = N;
		INT GESVDN = numTasks;
		INT GESVDLDA = N;
		INT GESVDLDU = N;
		INT GESVDLDVT = MN;
		INT INFO;

		GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, NULL, &GESVDLDA, NULL, NULL, &GESVDLDU, NULL, &GESVDLDVT, work, &lwork, &INFO);
		return;
	}

	INT maxNNumSamples = IMAX(N, numSamples);
	INT YpredVtMatFlag = 0;
	if (YpredVtMat == NULL) {
		YpredVtMat = (DOUBLE *) MALLOC(maxNNumSamples * numTasks * sizeof(DOUBLE));
		YpredVtMatFlag = 1;
	}

	INT minNNumTasks = IMIN(N, numTasks);
	INT svdVecFlag = 0;
	if (svdVec == NULL) {
		svdVec = (DOUBLE *) MALLOC(1 * minNNumTasks * sizeof(DOUBLE));
		svdVecFlag = 1;
	}

	INT derivVecFlag = 0;
	if ((derivVec == NULL) && (derivFlag == 1)) {
		derivVec = (DOUBLE *) MALLOC(1 * minNNumTasks * sizeof(DOUBLE));
		derivVecFlag = 1;
	}

	INT dataBufferFlag = 0;
	if (dataBuffer == NULL) {
		dataBuffer = (DOUBLE *) MALLOC(N * numTasks * sizeof(DOUBLE));
		dataBufferFlag = 1;
	}

	INT workFlag = 0;
	if (work == NULL) {
		workFlag = 1;
	}

	DOUBLE obj1;
	DOUBLE obj2;
	nuclear_approx_obj_grad(&obj1, deriv, W, rhop, N, numTasks, derivFlag, svdVec, YpredVtMat, \
			dataBuffer, derivVec, work, lwork);
	datacpy(dataBuffer, deriv, N * numTasks);
	multihuberhinge_obj_grad_speed(&obj2, deriv, W, X, Y, lambdap, classLabels, N, numSamples, numTasks, \
			derivFlag, 0, YpredVtMat);
	*obj = obj2 + *lambdap * obj1;

	if (derivFlag == 1) {
		INT AXPYN = N * numTasks;
		DOUBLE alpha = *lambdap;
		INT incx = 1;
		INT incy = 1;
		AXPY(&AXPYN, &alpha, dataBuffer, &incx, deriv, &incy);
	}

	if (YpredVtMatFlag == 1) {
		FREE(YpredVtMat);
	}

	if (svdVecFlag == 1) {
		FREE(svdVec);
	}

	if (derivVecFlag == 1) {
		FREE(derivVec);
	}

	if (dataBufferFlag == 1) {
		FREE(dataBuffer);
	}

	if (workFlag == 1) {
		FREE(work);
	}
}

// TODO: Fix issues with PEGASOS: very slow random generator, very slow when
// compared to vlFeat due to (our) use of MKL (!). Maybe use Intel's random
// generator.
void pegasos_svm_vl(DOUBLE *model, DOUBLE *X, DOUBLE *Y, INT N, INT numSamples,
				DOUBLE *lambdap, INT numIters) {

	/*
	 The model is stored as scale*model[]. When a sample does not violate
	 the margin, only scale needs to be updated.
	*/
	DOUBLE acc;
	INT iter;
	INT i;
	INT randIndex;
	DOUBLE *x;
	DOUBLE y;
	DOUBLE scale = 1;
	DOUBLE eta;
	DOUBLE lambda = *lambdap;
	DOUBLE sqrtLambda = sqrt(lambda) ;

	srand(time(NULL));
	for (iter = 0; iter < numIters; ++iter) {
		/* pick a sample  */
//		randchoose(&randIndex, numSamples, 1);
//		randIndex = iter % numSamples;
		randIndex = (INT) floor((rand() / (DOUBLE) RAND_MAX) * numSamples);
		x = X + N * randIndex ;
		y = Y[randIndex] ;

		/* project on the weight vector */
		acc = 0;
		/*
		 * NOTE: Custom for loop works (significantly) faster than MKL DOT.
		 */
		for (i = 0; i < N; ++i) {
			acc += model[i] * x[i];
		}
//		acc = DOT(&N, model, &incx, x, &incy);
		acc *= scale ;

		/* learning rate */
		eta = 1.0 / ((iter + 2) * lambda) ;

		if (y * acc < (DOUBLE) 1.0) {
			DOUBLE a = scale * (1 - eta * lambda)  ;
			DOUBLE b = y * eta ;

			acc = 0;
			for (i = 0 ; i < N; ++i) {
				model[i] = a * model[i] + b * x[i] ;
				acc += model[i] * model[i] ;
			}
			DOUBLE c = (DOUBLE) 1.0 / (sqrtLambda * sqrt(acc));
			scale = IMIN(c, (DOUBLE)1.0) ;
		} else {
			/* margin not violated */
			scale *= 1.0 - eta * lambda ;
		}
	}

	/* denormalize representation */
	for (i = 0 ; i < N; ++i) {
		model[i] *= scale ;
	}
}

void pegasos_svm_sub(DOUBLE *weights, DOUBLE *bias, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
		INT *taskLabel, INT N, INT numSamples, INT biasFlag, INT numIters, INT batchSize, INT returnAverageFlag, \
		INT *batch, DOUBLE *deriv, DOUBLE *weightsAverage) {

	INT iter;
	DOUBLE alpha;
	DOUBLE beta;
	DOUBLE normw;
	DOUBLE eta;
	INT iterB;
	DOUBLE Ypred;
	DOUBLE Ytask;
	INT DOTN = N;
	INT AXPYN = N;
	INT AXPBYN = N;
	INT NRM2N = N;
	INT SCALN = N;
	DOUBLE lambda = *lambdap;
	DOUBLE invSqrtLambda = 1 / sqrt(lambda);
	INT incx = 1;
	INT incy = 1;

	INT batchFlag = 0;
	if (batch == NULL) {
		batch = (INT *) CMALLOC(batchSize * sizeof(INT));
		batchFlag = 1;
	}
	INT derivFlag = 0;
	if (deriv == NULL) {
		deriv = (DOUBLE *) CMALLOC(N * sizeof(INT));
		derivFlag = 1;
	}

	DOUBLE biasDeriv;
	DOUBLE biasAverage;

	memset((void *) weights, 0, N * sizeof(DOUBLE));
	*bias = 0;
	INT weightsAverageFlag = 0;
	if (returnAverageFlag == 1) {
		if (weightsAverage == NULL) {
			weightsAverage = (DOUBLE *) CMALLOC(N * sizeof(DOUBLE));
			weightsAverageFlag = 1;
		}
		memset((void *) weightsAverage, 0, N * sizeof(DOUBLE));
		biasAverage = 0;
	}

	for (iter = 0; iter < numIters; ++iter) {
		memset((void *) deriv, 0, N * sizeof(DOUBLE));
		biasDeriv = 0;
		randchoose(batch, numSamples, batchSize);
//		*batch = iter % numSamples;

		/*
		* NOTE: Parallelization here does not work well, because of the overhead
		* of creating the threads at every iteration. Not 100% sure if a parallel
		* region can be created outside the outer for, with everything other than
		* the inner loop being called with pragma omp single. Non-threaded version
		* already much faster than MATLAB though, in both use scenaria (large patches
		*  vs many iterations).
		*/
		/*
		 * NOTE: Alternative parallelization scheme with large parallel region and
		 * many singles doesn't appear to work either. The issue is (probably)
		 * the critical part. It may eventually become faster if custom
		 * reduction is written for AXPY, although the issue is that deriv is
		 * allocated memory. See this version commented out below, for reference.
		 */
		for (iterB = 0; iterB < batchSize; ++iterB) {
			Ypred = DOT(&DOTN, weights, &incx, &X[batch[iterB] * N], &incy) + *bias;
			if (taskLabel == NULL) {
				Ytask = Y[batch[iterB]];
			} else {
				Ytask = ((INT) Y[batch[iterB]] == *taskLabel) ? (1) : (- 1);
			}
			if (Ytask * Ypred < 1) {
				alpha = Ytask;
				AXPY(&AXPYN, &alpha, &X[batch[iterB] * N], &incx, deriv, &incy);
				biasDeriv += (biasFlag) ? (alpha) : 0;
			}
		}

		eta = 1 / (lambda * ((DOUBLE) iter + 2.0));
		alpha = eta / (DOUBLE) batchSize;
		beta = 1.0 - eta * lambda;
		AXPBY(&AXPBYN, &alpha, deriv, &incx, &beta, weights, &incy);
		normw = NRM2(&NRM2N, weights, &incx);
		alpha = invSqrtLambda / normw;
		if (alpha < 1) {
			SCAL(&SCALN, &alpha, weights, &incx);
		}
		*bias += eta / (DOUBLE) batchSize * biasDeriv;

		if (returnAverageFlag) {
			alpha = 1 / (DOUBLE) numIters;
			AXPY(&AXPYN, &alpha, weights, &incx, weightsAverage, &incy);
			biasAverage += alpha * (*bias);
		}
	}

	if (returnAverageFlag) {
		datacpy(weights, weightsAverage, N);
		*bias = biasAverage;
	}

	if (batchFlag == 1) {
		CFREE(batch);
	}
	if (derivFlag == 1) {
		CFREE(deriv);
	}
	if (weightsAverageFlag == 1) {
		CFREE(weightsAverage);
	}
}

//void pegasos_binary_svm(DOUBLE *weights, DOUBLE *bias, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
//		INT N, INT numSamples, INT biasFlag, INT numIters, INT batchSize, INT returnAverageFlag) {
//
//	INT iter;
//	DOUBLE alpha;
//	DOUBLE beta;
//	DOUBLE normw;
//	DOUBLE eta;
//
//	INT *batch = (INT *) MALLOC(batchSize * sizeof(INT));
//	DOUBLE *deriv = (DOUBLE *) MALLOC(N * sizeof(DOUBLE));
//	DOUBLE biasDeriv;
//	DOUBLE *weightsAverage;
//	DOUBLE biasAverage;
//
//	memset((void *) weights, 0, N * sizeof(DOUBLE));
//	*bias = 0;
//	if (returnAverageFlag) {
//		weightsAverage = (DOUBLE *) MALLOC(N * sizeof(DOUBLE));
//		memset((void *) weightsAverage, 0, N * sizeof(DOUBLE));
//		biasAverage = 0;
//	}
//
//	srand(time(NULL));
//	#pragma omp parallel default(shared)
//	{
//		INT iterB;
//		DOUBLE Ypred;
//		DOUBLE alpha2;
//		INT DOTN = N;
//		INT AXPYN = N;
//		INT AXPBYN = N;
//		INT NRM2N = N;
//		INT SCALN = N;
//		DOUBLE lambda = *lambdap;
//		DOUBLE invSqrtLambda = 1 / sqrt(lambda);
//		INT incx = 1;
//		INT incy = 1;
//
//		#pragma omp single
//		iter = 0;
//		#pragma omp barrier
//		while (iter < numIters) {
//
//			#pragma omp single
//			{
//				memset((void *) deriv, 0, N * sizeof(DOUBLE));
//				biasDeriv = 0;
//				randchoose(batch, numSamples, batchSize);
//			}
//			#pragma omp barrier
//
//	//		#pragma omp parallel for private(iterB, Ypred, alpha) shared(batch, X, Y, deriv, weights, bias) \
//				firstprivate(batchSize, DOTN, AXPYN, incx, incy, N, biasFlag) reduction(+: biasDeriv)
//			#pragma omp for
//			for (iterB = 0; iterB < batchSize; ++iterB) {
//				Ypred = DOT(&DOTN, weights, &incx, &X[batch[iterB] *N], &incy) + *bias;
//				if (Y[batch[iterB]] * Ypred < 1) {
//					alpha2 = Y[batch[iterB]];
//					#pragma omp critical
//					{
//						AXPY(&AXPYN, &alpha2, &X[batch[iterB] *N], &incx, deriv, &incy);
//						biasDeriv += (biasFlag) ? (alpha2) : 0;
//					}
//				}
//			}
//
//			#pragma omp single
//			{
//				eta = 1 / (lambda * ((DOUBLE) iter + 2.0));
//				alpha = eta / (DOUBLE) batchSize;
//				beta = 1.0 - eta * lambda;
//				AXPBY(&AXPBYN, &alpha, deriv, &incx, &beta, weights, &incy);
//				normw = NRM2(&NRM2N, weights, &incx);
//				alpha = invSqrtLambda / normw;
//				if (alpha < 1) {
//					SCAL(&SCALN, &alpha, weights, &incx);
//				}
//				*bias += eta / (DOUBLE) batchSize * biasDeriv;
//
//				if (returnAverageFlag) {
//					alpha = 1 / (DOUBLE) numIters;
//					AXPY(&AXPYN, &alpha, weights, &incx, weightsAverage, &incy);
//					biasAverage += alpha * (*bias);
//				}
//				++iter;
//			}
//			#pragma omp barrier
//		}
//	}
//
//	if (returnAverageFlag) {
//		datacpy(weights, weightsAverage, N);
//		*bias = biasAverage;
//	}
//
//	FREE(batch);
//	FREE(deriv);
//	FREE(weightsAverage);
//}

void pegasos_binary_svm(DOUBLE *weights, DOUBLE *bias, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
		INT N, INT numSamples, INT biasFlag, INT numIters, INT batchSize, INT returnAverageFlag) {

	INT *batch = (INT *) MALLOC(batchSize * sizeof(INT));
	DOUBLE *deriv = (DOUBLE *) MALLOC(N * sizeof(INT));
	DOUBLE *weightsAverage;
	if (returnAverageFlag == 1) {
		weightsAverage = (DOUBLE *) MALLOC(N * sizeof(DOUBLE));
	} else {
		weightsAverage = NULL;
	}

	srand(time(NULL));
	pegasos_svm_sub(weights, bias, X, Y, lambdap, NULL, N, numSamples, biasFlag, \
		numIters, batchSize, returnAverageFlag, batch, deriv, weightsAverage);

	FREE(batch);
	FREE(deriv);
	if (returnAverageFlag == 1) {
		FREE(weightsAverage);
	}
}

// TODO: Change multiclass functions to include a list of labels. That way, labels
// don't need to be in 0:1.
void pegasos_multiclass_svm(DOUBLE *weights, DOUBLE *bias, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, DOUBLE *classLabels, \
		INT N, INT numSamples, INT numTasks, INT biasFlag, INT numIters, INT batchSize, INT returnAverageFlag) {

	INT iterT;
	INT taskLabel;
	srand(time(NULL));
	#pragma omp parallel private(iterT, taskLabel) shared(weights, bias, X, Y, lambdap, classLabels) \
		firstprivate(N, numTasks, numSamples, biasFlag, numIters, batchSize, returnAverageFlag)
	{
		INT *batch = (INT *) CMALLOC(batchSize * sizeof(INT));
		DOUBLE *deriv = (DOUBLE *) CMALLOC(N * sizeof(INT));
		DOUBLE *weightsAverage;
		if (returnAverageFlag == 1) {
			weightsAverage = (DOUBLE *) CMALLOC(N * sizeof(DOUBLE));
		} else {
			weightsAverage = NULL;
		}

		#pragma omp for
		for (iterT = 0; iterT < numTasks; ++iterT) {
			taskLabel = (INT) classLabels[iterT];
			pegasos_svm_sub(&weights[N * iterT], &bias[iterT], X, Y, lambdap, \
				&taskLabel, N, numSamples, biasFlag, numIters, batchSize, returnAverageFlag, \
				batch, deriv, weightsAverage);
		}

		CFREE(batch);
		CFREE(deriv);
		CFREE(weightsAverage);
	}
}

void pegasos_multiclass_svm_alt(DOUBLE *weights, DOUBLE *bias, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, DOUBLE *classLabels, \
		INT N, INT numSamples, INT numTasks, INT biasFlag, INT numIters, INT batchSize, INT returnAverageFlag) {

	INT iter;
	INT iterB;
	INT iterT;
	INT taskLabel;
	INT AXPYN = N;
	INT AXPYN2 = numTasks;
	INT AXPYN3 = N * numTasks;
	INT AXPBYN = N * numTasks;
	INT NRM2N = N;
	INT SCALN = N;
	CHAR trans = 'T';
	INT GEMVM = N;
	INT GEMVN = numTasks;
	INT GEMVLDA = N;
	DOUBLE Ytask;
	DOUBLE alpha;
	DOUBLE alpha2;
	DOUBLE beta;
	DOUBLE normw;
	DOUBLE lambda = *lambdap;
	DOUBLE eta;
	DOUBLE invSqrtLambda = 1 / sqrt(lambda);
	INT incx = 1;
	INT incy = 1;

	INT *batch = (INT *) MALLOC(batchSize * sizeof(INT));
	DOUBLE *deriv = (DOUBLE *) MALLOC(N * numTasks * sizeof(DOUBLE));
	DOUBLE *Ypred = (DOUBLE *) MALLOC(numTasks * sizeof(DOUBLE));
	DOUBLE *biasDeriv;
	if (biasFlag == 1) {
		biasDeriv = (DOUBLE *) MALLOC(numTasks * sizeof(DOUBLE));
	}
	DOUBLE *weightsAverage;
	DOUBLE *biasAverage;

	memset((void *) weights, 0, N * numTasks * sizeof(DOUBLE));
	if (biasFlag == 1) {
		memset((void *) bias, 0, numTasks * sizeof(DOUBLE));
	}
	if (returnAverageFlag == 1) {
		weightsAverage = (DOUBLE *) MALLOC(N * numTasks * sizeof(DOUBLE));
		memset((void *) weightsAverage, 0, N * numTasks * sizeof(DOUBLE));
		biasAverage = (DOUBLE *) MALLOC(numTasks * sizeof(DOUBLE));
		memset((void *) biasAverage, 0, numTasks * sizeof(DOUBLE));
	}

	srand(time(NULL));
//	#pragma omp parallel private(iterT, Ytask, alpha2, taskLabel) default(shared)
	{
//		#pragma omp single
		iter = 0;
//		#pragma omp barrier
		while (iter < numIters) {

//			#pragma omp single
			{
				memset((void *) deriv, 0, N * numTasks * sizeof(DOUBLE));
				if (biasFlag == 1) {
					memset((void *) biasDeriv, 0, numTasks * sizeof(DOUBLE));
				}
				randchoose(batch, numSamples, batchSize);
				iterB = 0;
			}
//			#pragma omp barrier
			while (iterB < batchSize) {
//				#pragma omp single
				{
					alpha = 1;
					beta = 0;
					GEMV(&trans, &GEMVM, &GEMVN, &alpha, weights, &GEMVLDA, &X[batch[iterB] *N], &incx, &beta, Ypred, &incy);
					if (biasFlag == 1) {
						AXPY(&AXPYN2, &alpha, bias, &incx, Ypred, &incy);
					}
				}

//				#pragma omp for
				for (iterT = 0; iterT < numTasks; ++iterT) {
					taskLabel = (INT) classLabels[iterT];
					Ytask = ((INT) Y[batch[iterB]] == taskLabel) ? (1) : (- 1);
					if (Ytask * Ypred[iterT] < 1) {
						alpha2 = Ytask;
						AXPY(&AXPYN, &alpha2, &X[batch[iterB] *N], &incx, &deriv[iterT * N], &incy);
						if (biasFlag == 1) {
							biasDeriv[iterT] += alpha2;
						}
					}
				}

//				#pragma omp single
				++iterB;
//				#pragma omp barrier
			}

//			#pragma omp single
			{
				eta = 1 / (lambda * ((DOUBLE) iter + 2.0));
				alpha = eta / (DOUBLE) batchSize;
				beta = 1.0 - eta * lambda;
				AXPBY(&AXPBYN, &alpha, deriv, &incx, &beta, weights, &incy);
				for (iterT = 0; iterT < numTasks; ++iterT) {
					normw = NRM2(&NRM2N, &weights[iterT * N], &incx);
					alpha = invSqrtLambda / normw;
					if (alpha < 1) {
						SCAL(&SCALN, &alpha, &weights[iterT * N], &incx);
					}
				}
				if (biasFlag == 1) {
					alpha = eta / (DOUBLE) batchSize;
					AXPY(&AXPYN2, &alpha, biasDeriv, &incx, bias, &incy);
				}
				if (returnAverageFlag) {
					alpha = 1 / (DOUBLE) numIters;
					AXPY(&AXPYN3, &alpha, weights, &incx, weightsAverage, &incy);
					if (biasFlag == 1) {
						AXPY(&AXPYN2, &alpha, bias, &incx, biasAverage, &incy);
					}
				}
				++iter;
			}
//			#pragma omp barrier
		}
	}

	if (returnAverageFlag) {
		datacpy(weights, weightsAverage, N * numTasks);
		datacpy(bias, biasAverage, numTasks);
	}

	FREE(batch);
	FREE(deriv);
	if (biasFlag == 1) {
		FREE(biasDeriv);
	}
	if (returnAverageFlag == 1) {
		FREE(weightsAverage);
		if (biasFlag == 1) {
			FREE(biasAverage);
		}
	}
}

void squaredhinge_kernel_obj_grad_sub(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *wb, \
			DOUBLE *X, DOUBLE *Y, INT taskLabel, DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, \
			INT regularizationFlag, DOUBLE *Ypred, DOUBLE *KX) {

	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) CMALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}

	INT KXFlag = 0;
	if (KX == NULL) {
		KX = (DOUBLE *) CMALLOC(M * numSamples * sizeof(DOUBLE));
		KXFlag = 1;
	}

	CHAR side = 'L';
	CHAR uplo = 'U';
	INT SYMMM = M;
	INT SYMMN = numSamples;
	DOUBLE alpha = 1;
	INT SYMMLDA = M;
	INT SYMMLDB = M;
	DOUBLE beta = 0;
	INT SYMMLDC = M;

	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, kernelMatrix, &SYMMLDA, X, &SYMMLDB, &beta, KX, &SYMMLDC);
	DOUBLE *Xp = KX;

	DOUBLE *weights = wb;
	DOUBLE lambda = * lambdap;

	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = numSamples;
	alpha = 1;
	INT GEMVLDA = M;
	beta = 0;
	INT incx = 1;
	INT incy = 1;

	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Xp, &GEMVLDA, weights, &incx, &beta, Ypred, &incy);

	INT iterX;
	DOUBLE YYpred;
	DOUBLE Ytask;
	DOUBLE objtemp = 0;

	uplo = 'U';
	INT SYR2N = M;
	INT SYR2LDA = M;

	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * M * sizeof(DOUBLE));
	}

	for (iterX = 0; iterX < numSamples; ++iterX) {
		Ytask = ((INT) Y[iterX] == taskLabel) ? (1) : (- 1);
		YYpred = Ytask * Ypred[iterX];
		if (YYpred < 1) {
			objtemp += SQR(1 - YYpred);
			if (derivFlag == 1) {
				alpha = Ypred[iterX] - Ytask;
				SYR2(&uplo, &SYR2N, &alpha, weights, &incx, &X[iterX * M], &incy, deriv, &SYR2LDA);
			}
		}
	}
	*obj = objtemp;

	if (regularizationFlag == 1) {
		uplo = 'U';
		INT SYMVN = M;
		alpha = 1;
		INT SYMVLDA = M;
		beta = 0;
		incx = 1;
		incy = 1;
		SYMV(&uplo, &SYMVN, &alpha, kernelMatrix, &SYMVLDA, weights, &incx, &beta, KX, &incy);

		INT DOTN = M;
		*obj = objtemp + lambda * DOT(&DOTN, weights, &incx, KX, &incy);

		if (derivFlag == 1) {
			INT SYRN = M;
			INT SYRLDA = M;
			alpha = lambda;
			SYR(&uplo, &SYRN, &alpha, weights, &incx, deriv, &SYRLDA);
		}
	}

	if (derivFlag == 1) {
		INT iterM, iterN;
		for (iterM = 0; iterM < M; ++iterM) {
			for (iterN = iterM + 1; iterN < M; ++iterN) {
				deriv[iterM * M + iterN] = deriv[iterN * M + iterM];
			}
		}
	}

	if (YpredFlag == 1) {
		CFREE(Ypred);
	}

	if (KXFlag == 1) {
		CFREE(KX);
	}
}

void squaredhinge_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *wb, \
			DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, \
			INT regularizationFlag, DOUBLE *Ypred, DOUBLE *KX) {
	INT taskLabel = 1;
	squaredhinge_kernel_obj_grad_sub(obj, deriv, kernelMatrix, wb, X, Y, taskLabel, lambdap, M, numSamples, \
			derivFlag, regularizationFlag, Ypred, KX);
}

void multisquaredhinge_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
				DOUBLE *classLabels, INT M, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag) {

	INT iterT;
	DOUBLE *Ypred;
	DOUBLE *KX;
	DOUBLE lambda = *lambdap;
	DOUBLE *objTemp = (DOUBLE *) MALLOC(numTasks * 1 * sizeof(DOUBLE));
	DOUBLE *derivTemp;
	if (derivFlag == 1) {
		derivTemp = (DOUBLE *) MALLOC(M * M * numTasks * sizeof(DOUBLE));
	}
	INT taskLabel;

	#pragma omp parallel private(Ypred, KX, taskLabel, iterT) \
		shared(objTemp, derivTemp, kernelMatrix, W, X, Y, classLabels) \
		firstprivate(numTasks, M, numSamples, derivFlag, regularizationFlag, lambda)
	{
		Ypred = (DOUBLE *) CMALLOC(1 * numSamples * sizeof(DOUBLE));
		KX = (DOUBLE *) CMALLOC(M * numSamples * sizeof(DOUBLE));
		#pragma omp for
		for (iterT = 0; iterT < numTasks; ++iterT) {
			taskLabel = (INT) classLabels[iterT];
			squaredhinge_kernel_obj_grad_sub(&objTemp[iterT], &derivTemp[M * M * iterT], kernelMatrix, &W[iterT * M], \
					X, Y, taskLabel, &lambda, M, numSamples, derivFlag, regularizationFlag, Ypred, KX);
		}
		CFREE(Ypred);
		CFREE(KX);
	}

	INT AXPYN = M * M;
	DOUBLE alpha = 1;
	INT incx = 1;
	INT incy = 1;
	*obj = 0;
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * M * sizeof(DOUBLE));
	}
	for (iterT = 0; iterT < numTasks; ++iterT) {
		*obj += objTemp[iterT];
		if (derivFlag == 1) {
			AXPY(&AXPYN, &alpha, &derivTemp[M * M * iterT], &incx, deriv, &incy);
		}
	}
	FREE(objTemp);
	if (derivFlag == 1) {
		FREE(derivTemp);
	}
}

void huberhinge_kernel_obj_grad_sub(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *wb, \
			DOUBLE *X, DOUBLE *Y, INT taskLabel, DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, \
			INT regularizationFlag, DOUBLE *Ypred, DOUBLE *KX) {

	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) CMALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}

	INT KXFlag = 0;
	if ((KX == NULL) && (kernelMatrix != NULL)) {
		KX = (DOUBLE *) CMALLOC(M * numSamples * sizeof(DOUBLE));
		KXFlag = 1;
	}

	CHAR side = 'L';
	CHAR uplo = 'U';
	INT SYMMM = M;
	INT SYMMN = numSamples;
	DOUBLE alpha = 1;
	INT SYMMLDA = M;
	INT SYMMLDB = M;
	DOUBLE beta = 0;
	INT SYMMLDC = M;

	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, kernelMatrix, &SYMMLDA, X, &SYMMLDB, &beta, KX, &SYMMLDC);
	DOUBLE *Xp = KX;

	DOUBLE *weights = wb;
	DOUBLE lambda = * lambdap;

	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = numSamples;
	alpha = 1;
	INT GEMVLDA = M;
	beta = 0;
	INT incx = 1;
	INT incy = 1;

	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Xp, &GEMVLDA, weights, &incx, &beta, Ypred, &incy);
	INT iterX;
	DOUBLE YYpred;
	DOUBLE Ytask;
	DOUBLE objtemp = 0;

	uplo = 'U';
	INT SYR2N = M;
	INT SYR2LDA = M;

	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * M * sizeof(DOUBLE));
	}

//	#pragma omp parallel for private(iterX, alpha, YYpred) shared(Xp, X, Y, Ypred, deriv, biasterm) \
		reduction(+: objtemp, biasderiv) firstprivate(numSamples, AXPYN, incx, incy, derivFlag)
	for (iterX = 0; iterX < numSamples; ++iterX) {

		Ytask = ((INT) Y[iterX] == taskLabel) ? (1) : (- 1);
		YYpred = Ytask * Ypred[iterX];
		if (YYpred < -1) {
			objtemp += - 4.0 * YYpred;
			if (derivFlag == 1) {
				alpha = - 2.0 * Ytask;
//				#pragma omp critical
				SYR2(&uplo, &SYR2N, &alpha, weights, &incx, &X[iterX * M], &incy, deriv, &SYR2LDA);
			}
		} else if ((YYpred < 1.0) && (YYpred >= - 1.0)) {
			objtemp += SQR(1.0 - YYpred);

			if (derivFlag == 1) {
				alpha = Ypred[iterX] - Ytask;
//				#pragma omp critical
				SYR2(&uplo, &SYR2N, &alpha, weights, &incx, &X[iterX * M], &incy, deriv, &SYR2LDA);
			}
		}
	}
	*obj = objtemp;

	if (regularizationFlag == 1) {
		CHAR uplo = 'U';
		INT SYMVN = M;
		alpha = 1;
		INT SYMVLDA = M;
		beta = 0;
		incx = 1;
		incy = 1;
		SYMV(&uplo, &SYMVN, &alpha, kernelMatrix, &SYMVLDA, weights, &incx, &beta, KX, &incy);

		INT DOTN = M;
		*obj = objtemp + lambda * DOT(&DOTN, weights, &incx, KX, &incy);

		if (derivFlag == 1) {
			INT SYRN = M;
			INT SYRLDA = M;
			alpha = lambda;
			SYR(&uplo, &SYRN, &alpha, weights, &incx, deriv, &SYRLDA);
		}
	}

	if (derivFlag == 1) {
		INT iterM, iterN;
		for (iterM = 0; iterM < M; ++iterM) {
			for (iterN = iterM + 1; iterN < M; ++iterN) {
				deriv[iterM * M + iterN] = deriv[iterN * M + iterM];
			}
		}
	}

	if (YpredFlag == 1) {
		CFREE(Ypred);
	}

	if (KXFlag == 1) {
		CFREE(KX);
	}
}

void huberhinge_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *wb, \
			DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, \
			INT regularizationFlag, DOUBLE *Ypred, DOUBLE *KX) {
	INT taskLabel = 1;
	huberhinge_kernel_obj_grad_sub(obj, deriv, kernelMatrix, wb, X, Y, taskLabel, lambdap, M, numSamples, \
			derivFlag, regularizationFlag, Ypred, KX);
}

void multihuberhinge_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
				DOUBLE *classLabels, INT M, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag) {

	INT iterT;
	DOUBLE *Ypred;
	DOUBLE *KX;
	DOUBLE lambda = *lambdap;
	DOUBLE *objTemp = (DOUBLE *) MALLOC(numTasks * 1 * sizeof(DOUBLE));
	DOUBLE *derivTemp;
	if (derivFlag == 1) {
		derivTemp = (DOUBLE *) MALLOC(M * M * numTasks * sizeof(DOUBLE));
	}
	INT taskLabel;

	#pragma omp parallel private(Ypred, KX, taskLabel, iterT) \
		shared(objTemp, derivTemp, kernelMatrix, W, X, Y, classLabels) \
		firstprivate(numTasks, M, numSamples, derivFlag, regularizationFlag, lambda)
	{
		Ypred = (DOUBLE *) CMALLOC(1 * numSamples * sizeof(DOUBLE));
		KX = (DOUBLE *) CMALLOC(M * numSamples * sizeof(DOUBLE));
		#pragma omp for
		for (iterT = 0; iterT < numTasks; ++iterT) {
			taskLabel = (INT) classLabels[iterT];
			huberhinge_kernel_obj_grad_sub(&objTemp[iterT], &derivTemp[M * M * iterT], kernelMatrix, &W[iterT * M], \
					X, Y, taskLabel, &lambda, M, numSamples, derivFlag, regularizationFlag, Ypred, KX);
		}
		CFREE(Ypred);
		CFREE(KX);
	}

	INT AXPYN = M * M;
	DOUBLE alpha = 1;
	INT incx = 1;
	INT incy = 1;
	*obj = 0;
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * M * sizeof(DOUBLE));
	}
	for (iterT = 0; iterT < numTasks; ++iterT) {
		*obj += objTemp[iterT];
		if (derivFlag == 1) {
			AXPY(&AXPYN, &alpha, &derivTemp[M * M * iterT], &incx, deriv, &incy);
		}
	}
	FREE(objTemp);
	if (derivFlag == 1) {
		FREE(derivTemp);
	}
}

//										  T const * data,
//                                          vl_size dimension,
//                                          vl_size numSamples,
//                                          vl_int8 const * labels,
//                                          double regularizer,
//                                          double biasMultiplier,
//                                          vl_uindex startingIteration,
//                                          vl_size numIterations,
//                                          VlRand * randomGenerator)
//
//  for (iteration = startingIteration ;
//       iteration < startingIteration + numIterations ;
//       ++ iteration) {
//    /* pick a sample  */
//    vl_uindex k = vl_rand_uindex(randomGenerator, numSamples) ;
//    x = data + dimension * k ;
//    y = labels[k] ;
//
//    /* project on the weight vector */
//    acc = dotFn(dimension, x, model) ;
//    if (biasMultiplier) acc += biasMultiplier * model[dimension] ;
//    acc *= scale ;
//
//    /* learning rate */
//    eta = 1.0 / (iteration * lambda) ;
//
//    if (y * acc < (T) 1.0) {
//      /* margin violated */
//      T a = scale * (1 - eta * lambda)  ;
//      T b = y * eta ;
//
//      acc = 0 ;
//      for (i = 0 ; i < dimension ; ++i) {
//        model[i] = a * model[i] + b * x[i] ;
//        acc += model[i] * model[i] ;
//      }
//      if (biasMultiplier) {
//        model[dimension] = a * model[dimension] + b * biasMultiplier ;
//        acc += model[dimension] * model[dimension] ;
//      }
//      scale = VL_MIN((T)1.0 / (sqrtLambda * sqrt(acc + VL_EPSILON_D)), (T)1.0) ;
//    } else {
//      /* margin not violated */
//      scale *= 1 - eta * lambda ;
//    }
//  }
//
//  /* denormalize representation */
//  for (i = 0 ; i < dimension + (biasMultiplier ? 1 : 0) ; ++i) {
//    model[i] *= scale ;
//  }
//}

//void minimize_kernel_basis(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *Xdata, DOUBLE *Acodes, INT N, INT K, INT numSamples, \
//				CHAR kernelType, DOUBLE *pparam1, DOUBLE *pparam2) {
//
//	DOUBLE INTERV = 0.1;
//	DOUBLE EXT = 3.0;
//	INT MAX = 20;
//	DOUBLE RATIO = (DOUBLE) 10;
//	DOUBLE SIG = 0.1;
//	DOUBLE RHO = SIG / (DOUBLE) 2;
//	INT MN = N * K;
//
//	CHAR lamch_opt = 'U';
//	DOUBLE realmin = LAMCH(&lamch_opt);
//
//	DOUBLE red = 1;
//
//	INT i = 0;
//	INT ls_failed = 0;
//	DOUBLE f0;
//	DOUBLE *df0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
//	DOUBLE *dftemp = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
//	DOUBLE *df3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
//	DOUBLE *s = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
//	DOUBLE d0;
//	INT derivFlag = 1;
//
//	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
//	datacpy(X, Xorig, MN);
//
//	DOUBLE *KDD = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
//	DOUBLE *KDX = (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
//	DOUBLE *KDDDX = (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
//	DOUBLE *normMat1 = (DOUBLE *) MALLOC(IMAX(K, numSamples) * 1 * sizeof(DOUBLE));
//	DOUBLE *oneVec = (DOUBLE *) MALLOC(IMAX(K, numSamples) * 1 * sizeof(DOUBLE));
//	DOUBLE *ak = (DOUBLE *) MALLOC(numSamples * 1 * sizeof(DOUBLE));
//	DOUBLE *tempMat1 = (DOUBLE *) MALLOC(numSamples * K * sizeof(DOUBLE));
//	DOUBLE *tempMat2 = (DOUBLE *) MALLOC(numSamples * N * sizeof(DOUBLE));
//
//	basis_kernel_obj_grad(&f0, df0, X, Xdata, Acodes, N, K, numSamples, kernelType, pparam1, pparam2, \
//			derivFlag, KDD, KDX, KDDDX, ak, normMat1, oneVec, tempMat1, tempMat2);
///*
//	dual_obj_grad(&f0, df0, X, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);
//*/
//
//	INT incx = 1;
//	INT incy = 1;
//
//	datacpy(s, df0, MN );
//	DOUBLE alpha = -1;
//	SCAL(&MN, &alpha, s, &incx);
//
//	d0 = - DOT(&MN, s, &incx, s, &incy);
//
//	DOUBLE x1;
//	DOUBLE x2;
//	DOUBLE x3;
//	DOUBLE x4;
//	DOUBLE *X0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
//	DOUBLE *X3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
//	DOUBLE F0;
//	DOUBLE *dF0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
//	INT Mmin;
//	DOUBLE f1;
//	DOUBLE f2;
//	DOUBLE f3;
//	DOUBLE f4;
//	DOUBLE d1;
//	DOUBLE d2;
//	DOUBLE d3;
//	DOUBLE d4;
//	INT success;
//	DOUBLE A;
//	DOUBLE B;
//	DOUBLE sqrtquantity;
//	DOUBLE tempnorm;
//	DOUBLE tempinprod1;
//	DOUBLE tempinprod2;
//	DOUBLE tempscalefactor;
//
//	x3 = red / (1 - d0);
//
//	while (i++ < length) {
//
//		datacpy(X0, X, MN);
//		datacpy(dF0, df0, MN);
//		F0 = f0;
//		Mmin = MAX;
//
//		while (1) {
//			x2 = 0;
//			f2 = f0;
//			d2 = d0;
//			f3 = f0;
//
//			datacpy(df3, df0, MN);
//
//			success = 0;
//			while ((!success) && (Mmin > 0)) {
//				Mmin = Mmin - 1;
//
//				datacpy(X3, X, MN);
//				alpha = x3;
//				AXPY(&MN, &alpha, s, &incx, X3, &incy);
//
//				basis_kernel_obj_grad(&f3, df3, X3, Xdata, Acodes, N, K, numSamples, kernelType, pparam1, pparam2, \
//						derivFlag, KDD, KDX, KDDDX, ak, normMat1, oneVec, tempMat1, tempMat2);
///*
//				dual_obj_grad(&f3, df3, X3, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);
//*/
//
//				if (ISNAN(f3) || ISINF(f3)) {  /* any(isnan(df3)+isinf(df3)) */
//					x3 = (x2 + x3) * 0.5;
//				} else {
//					success = 1;
//				}
//			}
//
//			if (f3 < F0) {
//
//				datacpy(X0, X, MN);
//				alpha = x3;
//				AXPY(&MN, &alpha, s, &incx, X0, &incy);
//				datacpy(dF0, df3, MN);
//				F0 = f3;
//			}
//
//			d3 = DOT(&MN, df3, &incx, s, &incy);
//
//			if ((d3 > SIG * d0) || (f3 > f0 + x3 * RHO * d0) || (Mmin == 0)) {
//				break;
//			}
//
//			x1 = x2;
//			f1 = f2;
//			d1 = d2;
//			x2 = x3;
//			f2 = f3;
//			d2 = d3;
//			A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1);
//			B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1);
//			sqrtquantity = B * B - A * d1 * (x2 - x1);
//
//			if (sqrtquantity < 0) {
//				x3 = x2 * EXT;
//			} else {
//				x3 = x1 - d1 * SQR(x2 - x1) / (B + SQRT(sqrtquantity));
//				if (ISNAN(x3) || ISINF(x3) || (x3 < 0)) {
//					x3 = x2 * EXT;
//				} else if (x3 > x2 * EXT) {
//					x3 = x2 * EXT;
//				} else if (x3 < x2 + INTERV * (x2 - x1)) {
//					x3 = x2 + INTERV * (x2 - x1);
//				}
//			}
//		}
//
//		while (((ABS(d3) > - SIG * d0) || (f3 > f0 + x3 * RHO * d0)) && (Mmin > 0)) {
//			if ((d3 > 0) || (f3 > f0 + x3 * RHO * d0)) {
//				x4 = x3;
//				f4 = f3;
//				d4 = d3;
//			} else {
//				x2 = x3;
//				f2 = f3;
//				d2 = d3;
//			}
//
//			if (f4 > f0) {
//				x3 = x2 - (0.5 * d2 * SQR(x4 - x2)) / (f4 - f2 - d2 * (x4 - x2));
//			} else {
//				A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2);
//				B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2);
//				x3 = x2 + (SQRT(B * B - A * d2 * SQR(x4 - x2)) - B) / A;
//			}
//
//			if (ISNAN(x3) || ISINF(x3)) {
//				x3 = (x2 + x4) * 0.5;
//			}
//			x3 = IMAX(IMIN(x3, x4 - INTERV * (x4 - x2)), x2 + INTERV * (x4 - x2));
//
//			datacpy(X3, X, MN);
//			alpha = x3;
//			AXPY(&MN, &alpha, s, &incx, X3, &incy);
//
//			basis_kernel_obj_grad(&f3, df3, X3, Xdata, Acodes, N, K, numSamples, kernelType, pparam1, pparam2, \
//					derivFlag, KDD, KDX, KDDDX, ak, normMat1, oneVec, tempMat1, tempMat2);
///*
//			dual_obj_grad(&f3, df3, X3, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);
//*/
//
//			if (f3 < F0) {
//				datacpy(X0, X, MN);
//				alpha = x3;
//				AXPY(&MN, &alpha, s, &incx, X0, &incy);
//				datacpy(dF0, df3, MN);
//				F0 = f3;
//			}
//
//			Mmin = Mmin - 1;
//			d3 = DOT(&MN, df3, &incx, s, &incy);
//		}
//
//		if ((ABS(d3) < - SIG * d0) && (f3 < f0 + x3 * RHO * d0)) {
//			alpha = x3;
//			AXPY(&MN, &alpha, s, &incx, X, &incy);
//			f0 = f3;
//
//			datacpy(dftemp, df3, MN);
//			alpha = -1;
//			AXPY(&MN, &alpha, df0, &incx, dftemp, &incy);
//			tempinprod1 = DOT(&MN, dftemp, &incx, df3, &incy);
//			tempnorm = NRM2(&MN, df0, &incx);
//			tempinprod2 = SQR(tempnorm);
//			tempscalefactor = tempinprod1 / tempinprod2;
//
//			alpha = tempscalefactor;
//			SCAL(&MN, &alpha, s, &incx);
//			alpha = -1;
//			AXPY(&MN, &alpha, df3, &incx, s, &incy);
//			datacpy(df0, df3, MN);
//			d3 = d0;
//			d0 = DOT(&MN, df0, &incx, s, &incy);
//
//			if (d0 > 0) {
//				datacpy(s, df0, MN);
//				alpha = -1;
//				SCAL(&MN, &alpha, s, &incx);
//				tempnorm = NRM2(&MN, s, &incx);
//				d0 = - SQR(tempnorm);
//			}
//			x3 = x3 * IMIN(RATIO, d3 / (d0 - realmin));
//			ls_failed = 0;
//		} else {
//			datacpy(X, X0, MN);
//			datacpy(df0, dF0, MN);
//			f0 = F0;
//
//			if ((ls_failed == 1) || (i > length)) {
//				break;
//			}
//			datacpy(s, df0, MN);
//			alpha = -1;
//			SCAL(&MN, &alpha, s, &incx);
//			tempnorm = NRM2(&MN, s, &incx);
//			d0 = - SQR(tempnorm);
//			x3 = 1 / (1 - d0);
//
//			ls_failed = 1;
//		}
//	}
//
//	datacpy(Xopt, X, MN);
//
//	FREE(KDD);
//	FREE(KDX);
//	FREE(KDDDX);
//	FREE(normMat1);
//	FREE(oneVec);
//	FREE(ak);
//	FREE(tempMat1);
//	FREE(tempMat2);
//	FREE(df0);
//	FREE(dftemp);
//	FREE(df3);
//	FREE(s);
//	FREE(X);
//	FREE(X0);
//	FREE(X3);
//	FREE(dF0);
//}
