/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "l2kernel_learn_basis.h"
#include "kernel_gram.h"
#include "useinterfaces.h"
#include "utils.h"
#include "useblas.h"

/* TODO: Parallelize, similarly to basis_exp_obj_grad */
void basis_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *A, INT N, INT K, INT numSamples, \
				KERNEL_TYPE kernelType, DOUBLE *pparam1, DOUBLE *pparam2, INT derivFlag, DOUBLE *KDD, DOUBLE *KDX, DOUBLE *KDDDX, \
				DOUBLE *ak, DOUBLE *normMat1, DOUBLE *oneVec, DOUBLE *tempMat1, DOUBLE *tempMat2) {
	
	INT KDDflag = 0;
	if (KDD == NULL) {
		KDD = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		KDDflag = 1;
	}
	
	INT KDXflag = 0;
	if (KDX == NULL) {
		KDX = (DOUBLE *) CMALLOC(K * numSamples * sizeof(DOUBLE));
		KDXflag = 1;
	}
	
	INT KDDDXflag = 0;
	if (KDDDX == NULL) {
		KDDDX = (DOUBLE *) CMALLOC(K * numSamples * sizeof(DOUBLE));
		KDDDXflag = 1;
	}

	INT normMat1flag = 0;
	if (normMat1 == NULL) {
		normMat1 = (DOUBLE *) CMALLOC(IMAX(K, numSamples) * 1 * sizeof(DOUBLE));
		normMat1flag = 1;
	}

	INT oneVecflag = 0;
	if (oneVec == NULL) {
		oneVec = (DOUBLE *) CMALLOC(IMAX(K, numSamples) * 1 * sizeof(DOUBLE));
		oneVecflag = 1;
	}
	
	INT akflag = 0;
	INT tempMat1flag = 0;
	INT tempMat2flag = 0;
	if (derivFlag == 1) {
		if (ak == NULL) {
			ak = (DOUBLE *) CMALLOC(numSamples * 1 * sizeof(DOUBLE));
			akflag = 1;
		}

		if (tempMat1 == NULL) {
			tempMat1 = (DOUBLE *) CMALLOC(numSamples * K * sizeof(DOUBLE));
			tempMat1flag = 1;
		}

		if (tempMat2 == NULL) {
			tempMat2 = (DOUBLE *) CMALLOC(numSamples * N * sizeof(DOUBLE));
			tempMat2flag = 1;
		}
	}

	kernel_gram(KDD, D, NULL, N, K, 0, kernelType, pparam1, pparam2, normMat1, oneVec);
	kernel_gram(KDX, D, X, N, K, numSamples, kernelType, pparam1, pparam2, normMat1, oneVec);
	
	datacpy(KDDDX, KDX, K * numSamples);
	
	CHAR side = 'L';
	CHAR uplo = 'U';
	INT SYMMM = K;
	INT SYMMN = numSamples;
	DOUBLE alpha = 1.0;
	INT SYMMLDA = K;
	INT SYMMLDB = K;
	DOUBLE beta = - 2.0;
	INT SYMMLDC = K;
	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, KDD, &SYMMLDA, A, &SYMMLDB, &beta, KDDDX, &SYMMLDC);
	
	INT DOTN = K;
	INT incx = 1;
	INT incy = 1;
	INT iterX;
	DOUBLE objtemp;
	DOUBLE objval = 0;
	
	#pragma omp parallel for private(iterX, objtemp) shared(A, KDDDX) \
			reduction(+: objval) firstprivate(K, numSamples, incx, incy, DOTN)
	for (iterX = 0; iterX < numSamples; ++iterX) {
		objtemp = DOT(&DOTN, &A[K * iterX], &incx, &KDDDX[K * iterX], &incy);
		objval += objtemp;
	}
	*obj = objval;
	
	if (derivFlag == 1) {
		
		INT AXPYN = K * numSamples;
		alpha = 1.0;
		AXPY(&AXPYN, &alpha, KDX, &incx, KDDDX, &incy);
		
		INT iterK;
		INT iterK2;
		DOTN = numSamples;
		DOUBLE mult;
		AXPYN = N;
		DOTN = numSamples;
		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = N;
		INT GEMMN = numSamples;
		INT GEMMK = K;
		INT GEMMLDA = N;
		INT GEMMLDB = K;
		INT GEMMLDC = N;
		CHAR trans = 'N';
		INT GEMVM = N;
		INT GEMVN = numSamples;
		INT GEMVLDA = N;
		
		DOUBLE s;
		if (pparam1 == NULL) {
			s = 1;
		} else {
			s = *pparam1;
		}
		
		for (iterK = 0; iterK < K; ++iterK) {
			
			memset((void *) tempMat2, 0, numSamples * N * sizeof(DOUBLE));
			for (iterX = 0; iterX < numSamples; ++iterX) {
				ak[iterX] = A[iterX * K + iterK];
				oneVec[iterX] = KDDDX[iterX * K + iterK];
				alpha = KDX[K * iterX + iterK];
				AXPY(&AXPYN, &alpha, &X[N * iterX], &incx, &tempMat2[N * iterX], &incy);
				for (iterK2 = 0; iterK2 < K; ++iterK2) {
					tempMat1[iterX * K + iterK2] = A[iterX * K + iterK2] * KDD[iterK * K + iterK2];
				}
			}
			
			mult = DOT(&DOTN, ak, &incx, oneVec, &incy);
			
			alpha = 1;
			beta = -1;
			GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, D, &GEMMLDA, tempMat1, &GEMMLDB, &beta, tempMat2, &GEMMLDC);
			
			alpha = 2.0 / SQR(s);
			beta = - alpha * mult;
			datacpy(&deriv[N * iterK], &D[N * iterK], N);
			GEMV(&trans, &GEMVM, &GEMVN, &alpha, tempMat2, &GEMVLDA, ak, &incx, &beta, &deriv[N * iterK], &incy);
			
		}
	}
		
	if (KDDflag == 1) {
		CFREE(KDD);
	}
	
	if (KDXflag == 1) {
		CFREE(KDX);
	}
	
	if (KDDDXflag == 1) {
		CFREE(KDDDX);
	}

	if (normMat1flag == 1) {
		CFREE(normMat1);
	}

	if (oneVecflag == 1) {
		CFREE(oneVec);
	}
	
	if (derivFlag == 1) {
		if (akflag == 1) {
			CFREE(ak);
		}

		if (tempMat1flag == 1) {
			CFREE(tempMat1);
		}

		if (tempMat2flag == 1) {
			CFREE(tempMat2);
		}
	}
}

void minimize_kernel_basis(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *Xdata, DOUBLE *Acodes, INT N, INT K, INT numSamples, \
				KERNEL_TYPE kernelType, DOUBLE *pparam1, DOUBLE *pparam2) {
	
	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
	INT MN = N * K;
	
	CHAR lamch_opt = 'U';
	DOUBLE realmin = LAMCH(&lamch_opt);

	DOUBLE red = 1;

	INT i = 0;
	INT ls_failed = 0;
	DOUBLE f0;
	DOUBLE *df0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *dftemp = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *df3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *s = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE d0;
	INT derivFlag = 1;

	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	
	DOUBLE *KDD = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
	DOUBLE *KDX = (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
	DOUBLE *KDDDX = (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
	DOUBLE *normMat1 = (DOUBLE *) MALLOC(IMAX(K, numSamples) * 1 * sizeof(DOUBLE));
	DOUBLE *oneVec = (DOUBLE *) MALLOC(IMAX(K, numSamples) * 1 * sizeof(DOUBLE));
	DOUBLE *ak = (DOUBLE *) MALLOC(numSamples * 1 * sizeof(DOUBLE));
	DOUBLE *tempMat1 = (DOUBLE *) MALLOC(numSamples * K * sizeof(DOUBLE));
	DOUBLE *tempMat2 = (DOUBLE *) MALLOC(numSamples * N * sizeof(DOUBLE));
	
	basis_kernel_obj_grad(&f0, df0, X, Xdata, Acodes, N, K, numSamples, kernelType, pparam1, pparam2, \
			derivFlag, KDD, KDX, KDDDX, ak, normMat1, oneVec, tempMat1, tempMat2);
/*
	dual_obj_grad(&f0, df0, X, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);
*/
	
	INT incx = 1;
	INT incy = 1;

	datacpy(s, df0, MN );
	DOUBLE alpha = -1;
	SCAL(&MN, &alpha, s, &incx);
	
	d0 = - DOT(&MN, s, &incx, s, &incy);
	
	DOUBLE x1;
	DOUBLE x2;
	DOUBLE x3;
	DOUBLE x4;
	DOUBLE *X0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *X3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE F0;
	DOUBLE *dF0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	INT Mmin;
	DOUBLE f1;
	DOUBLE f2;
	DOUBLE f3;
	DOUBLE f4;
	DOUBLE d1;
	DOUBLE d2;
	DOUBLE d3;
	DOUBLE d4;
	INT success;
	DOUBLE A;
	DOUBLE B;
	DOUBLE sqrtquantity;
	DOUBLE tempnorm;
	DOUBLE tempinprod1;
	DOUBLE tempinprod2;
	DOUBLE tempscalefactor;
	
	x3 = red / (1 - d0);            

	while (i++ < length) {

		datacpy(X0, X, MN);
		datacpy(dF0, df0, MN);
		F0 = f0;
		Mmin = MAX;
		
		while (1) {
			x2 = 0;
			f2 = f0;
			d2 = d0;
			f3 = f0;

			datacpy(df3, df0, MN);
			
			success = 0;
			while ((!success) && (Mmin > 0)) {
				Mmin = Mmin - 1;
				
				datacpy(X3, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X3, &incy);
				
				basis_kernel_obj_grad(&f3, df3, X3, Xdata, Acodes, N, K, numSamples, kernelType, pparam1, pparam2, \
						derivFlag, KDD, KDX, KDDDX, ak, normMat1, oneVec, tempMat1, tempMat2);
/*
				dual_obj_grad(&f3, df3, X3, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);	
*/

				if (ISNAN(f3) || ISINF(f3)) {  /* any(isnan(df3)+isinf(df3)) */
					x3 = (x2 + x3) * 0.5;
				} else {
					success = 1;
				}
			}
			
			if (f3 < F0) {

				datacpy(X0, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X0, &incy);
				datacpy(dF0, df3, MN);
				F0 = f3;
			}	
			
			d3 = DOT(&MN, df3, &incx, s, &incy);

			if ((d3 > SIG * d0) || (f3 > f0 + x3 * RHO * d0) || (Mmin == 0)) {
				break;
			}
			
			x1 = x2; 
			f1 = f2; 
			d1 = d2;
			x2 = x3; 
			f2 = f3; 
			d2 = d3;
			A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1);
			B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1);
			sqrtquantity = B * B - A * d1 * (x2 - x1);

			if (sqrtquantity < 0) {
				x3 = x2 * EXT;
			} else {
				x3 = x1 - d1 * SQR(x2 - x1) / (B + SQRT(sqrtquantity));
				if (ISNAN(x3) || ISINF(x3) || (x3 < 0)) {
					x3 = x2 * EXT;
				} else if (x3 > x2 * EXT) {
					x3 = x2 * EXT;
				} else if (x3 < x2 + INTERV * (x2 - x1)) {
					x3 = x2 + INTERV * (x2 - x1);
				}
			}		
		}                

		while (((ABS(d3) > - SIG * d0) || (f3 > f0 + x3 * RHO * d0)) && (Mmin > 0)) {
			if ((d3 > 0) || (f3 > f0 + x3 * RHO * d0)) {
				x4 = x3;
				f4 = f3;
				d4 = d3;
			} else {
				x2 = x3;
				f2 = f3;
				d2 = d3;
			}

			if (f4 > f0) {
				x3 = x2 - (0.5 * d2 * SQR(x4 - x2)) / (f4 - f2 - d2 * (x4 - x2));
			} else {
				A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2);
				B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2);
				x3 = x2 + (SQRT(B * B - A * d2 * SQR(x4 - x2)) - B) / A;
			}

			if (ISNAN(x3) || ISINF(x3)) {
				x3 = (x2 + x4) * 0.5;
			}
			x3 = IMAX(IMIN(x3, x4 - INTERV * (x4 - x2)), x2 + INTERV * (x4 - x2));

			datacpy(X3, X, MN);
			alpha = x3;
			AXPY(&MN, &alpha, s, &incx, X3, &incy);			

			basis_kernel_obj_grad(&f3, df3, X3, Xdata, Acodes, N, K, numSamples, kernelType, pparam1, pparam2, \
					derivFlag, KDD, KDX, KDDDX, ak, normMat1, oneVec, tempMat1, tempMat2);
/*
			dual_obj_grad(&f3, df3, X3, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);
*/

			if (f3 < F0) {
				datacpy(X0, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X0, &incy);
				datacpy(dF0, df3, MN);
				F0 = f3;
			}

			Mmin = Mmin - 1;
			d3 = DOT(&MN, df3, &incx, s, &incy);
		}
		
		if ((ABS(d3) < - SIG * d0) && (f3 < f0 + x3 * RHO * d0)) {
			alpha = x3;
			AXPY(&MN, &alpha, s, &incx, X, &incy);
			f0 = f3;
			
			datacpy(dftemp, df3, MN);
			alpha = -1;
			AXPY(&MN, &alpha, df0, &incx, dftemp, &incy);
			tempinprod1 = DOT(&MN, dftemp, &incx, df3, &incy);
			tempnorm = NRM2(&MN, df0, &incx);
			tempinprod2 = SQR(tempnorm);
			tempscalefactor = tempinprod1 / tempinprod2;

			alpha = tempscalefactor;
			SCAL(&MN, &alpha, s, &incx);
			alpha = -1;
			AXPY(&MN, &alpha, df3, &incx, s, &incy);
			datacpy(df0, df3, MN);
			d3 = d0;
			d0 = DOT(&MN, df0, &incx, s, &incy);

			if (d0 > 0) {
				datacpy(s, df0, MN);
				alpha = -1;
				SCAL(&MN, &alpha, s, &incx);
				tempnorm = NRM2(&MN, s, &incx);
				d0 = - SQR(tempnorm);
			}
			x3 = x3 * IMIN(RATIO, d3 / (d0 - realmin));
			ls_failed = 0;
		} else {
			datacpy(X, X0, MN);
			datacpy(df0, dF0, MN);
			f0 = F0;
			
			if ((ls_failed == 1) || (i > length)) {
				break;
			}
			datacpy(s, df0, MN);
			alpha = -1;
			SCAL(&MN, &alpha, s, &incx);
			tempnorm = NRM2(&MN, s, &incx);
			d0 = - SQR(tempnorm);
			x3 = 1 / (1 - d0);
			
			ls_failed = 1;
		}
	}

	datacpy(Xopt, X, MN);
	
	FREE(KDD);
	FREE(KDX);
	FREE(KDDDX);
	FREE(normMat1);
	FREE(oneVec);
	FREE(ak);
	FREE(tempMat1);
	FREE(tempMat2);
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}
