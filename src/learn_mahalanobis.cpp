/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useblas.h"
#include "useinterfaces.h"
#include "learn_mahalanobis.h"
#include "utils.h"

void mahalanobis_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, \
					INT N, INT derivFlag, DOUBLE *GtG, DOUBLE *ObjMat, DOUBLE *MDDt2) {
	
	int GtGFlag = 0;
	if (GtG == NULL) {
		GtG = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		GtGFlag = 1;
	}
	
	CHAR side = 'L';
	CHAR uplo = 'U';
	INT SYMMM = N;
	INT SYMMN = N;
	DOUBLE alpha = 1;
	INT SYMMLDA = N;
	INT SYMMLDB = N;
	DOUBLE beta = 0;
	INT SYMMLDC = N;
	
	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, X, &SYMMLDA, VL, &SYMMLDB, &beta, GtG, &SYMMLDC);

	CHAR transa = 'T';
	CHAR transb = 'N';
	INT GEMMM = N;
	INT GEMMN = N;
	INT GEMMK = N;
	alpha = -1;
	INT GEMMLDA = N;	
	INT GEMMLDB = N;
	beta = 0;
	INT GEMMLDC = N;
	
	int ObjMatFlag = 0;
	if (ObjMat == NULL) {
		ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		ObjMatFlag = 1;
	}
	
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, VL, &GEMMLDA, GtG, &GEMMLDB, &beta, ObjMat, &GEMMLDC);
	
	INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		ObjMat[iterN * N + iterN] += L[iterN];
	}
	
	CHAR norm = 'F';
	uplo = 'U';
	INT LANSYN = N;
	INT LANSYLDA = N;
	
	*obj = LANSY(&norm, &uplo, &LANSYN, ObjMat, &LANSYLDA, NULL);
	*obj = SQR(*obj);
	
	if (derivFlag == 1) {
		int MDDt2Flag = 0;
		if (MDDt2 == NULL) {
			MDDt2 = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
			MDDt2Flag = 1;
		}

		side = 'L';
		uplo = 'U';
		SYMMM = N;
		SYMMN = N;
		alpha = 1;
		SYMMLDA = N;
		SYMMLDB = N;
		beta = 0;
		SYMMLDC = N;

		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, X, &SYMMLDA, DDt2, &SYMMLDB, &beta, MDDt2, &SYMMLDC);

		datacpy(deriv, DDt3, N * N);

		side = 'L';
		uplo = 'U';
		SYMMM = N;
		SYMMN = N;
		alpha = 2;
		SYMMLDA = N;
		SYMMLDB = N;
		beta = -2;
		SYMMLDC = N;

		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, DDt2, &SYMMLDA, MDDt2, &SYMMLDB, &beta, deriv, &SYMMLDC);
		
		if (MDDt2Flag == 1) {
			FREE(MDDt2);
		}
	}
	
	if (GtGFlag == 1) {
		FREE(GtG);
	}
	
	if (ObjMatFlag == 1) {
		FREE(ObjMat);
	}
}

void minimize_mahalanobis(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, \
		DOUBLE *L, INT N) {

	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
	INT M = N;
	INT MN = M * N;
	
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
	
	DOUBLE *GtG = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *MDDt2 = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT derivFlag = 1;
	
	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	mahalanobis_obj_grad(&f0, df0, X, DDt2, DDt3, VL, L, N, derivFlag, GtG, ObjMat, MDDt2);
	
	INT incx = 1;
	INT incy = 1;

	datacpy(s, df0, MN);
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
	
				mahalanobis_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, N, derivFlag, GtG, ObjMat, MDDt2);

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
				
			mahalanobis_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, N, derivFlag, GtG, ObjMat, MDDt2);
			
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
	
	FREE(GtG);
	FREE(ObjMat);
	FREE(MDDt2);
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}

void mahalanobis_unweighted_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *D, DOUBLE *DDt, \
					INT N, INT K, INT derivFlag, DOUBLE *MD, DOUBLE *ObjMat, DOUBLE *MDDt) {

	INT MDFlag = 0;
	if (MD == NULL) {
		MD = (DOUBLE *) MALLOC(N * K * sizeof(DOUBLE));
		MDFlag = 1;
	}

	CHAR transa = 'N';
	CHAR transb = 'N';
	INT GEMMM = N;
	INT GEMMN = K;
	INT GEMMK = N;
	DOUBLE alpha = 1;
	INT GEMMLDA = N;
	INT GEMMLDB = N;
	DOUBLE beta = 0;
	INT GEMMLDC = N;

	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X, &GEMMLDA, D, &GEMMLDB, &beta, MD, &GEMMLDC);

	transa = 'T';
	transb = 'N';
	GEMMM = K;
	GEMMN = K;
	GEMMK = N;
	alpha = -1;
	GEMMLDA = N;
	GEMMLDB = N;
	beta = 0;
	GEMMLDC = K;

	INT ObjMatFlag = 0;
	if (ObjMat == NULL) {
		ObjMat = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
		ObjMatFlag = 1;
	}

	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, D, &GEMMLDA, MD, &GEMMLDB, &beta, ObjMat, &GEMMLDC);

	INT iterK;
	for (iterK = 0; iterK < K; ++iterK) {
		ObjMat[iterK * K + iterK] += 1;
	}

	CHAR norm = 'F';
	CHAR uplo = 'U';
	INT LANSYN = K;
	INT LANSYLDA = K;

	*obj = LANSY(&norm, &uplo, &LANSYN, ObjMat, &LANSYLDA, NULL);
	*obj = SQR(*obj);

	if (derivFlag == 1) {
		INT MDDtFlag = 0;
		if (MDDt == NULL) {
			MDDt = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
			MDDtFlag = 1;
		}

		CHAR side = 'L';
		uplo = 'U';
		INT SYMMM = N;
		INT SYMMN = N;
		alpha = 1;
		INT SYMMLDA = N;
		INT SYMMLDB = N;
		beta = 0;
		INT SYMMLDC = N;

		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, X, &SYMMLDA, DDt, &SYMMLDB, &beta, MDDt, &SYMMLDC);

		datacpy(deriv, DDt, N * N);

		side = 'L';
		uplo = 'U';
		SYMMM = N;
		SYMMN = N;
		alpha = 2;
		SYMMLDA = N;
		SYMMLDB = N;
		beta = -2;
		SYMMLDC = N;

		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, DDt, &SYMMLDA, MDDt, &SYMMLDB, &beta, deriv, &SYMMLDC);

		if (MDDtFlag == 1) {
			FREE(MDDt);
		}
	}

	if (MDFlag == 1) {
		FREE(MD);
	}

	if (ObjMatFlag == 1) {
		FREE(ObjMat);
	}
}

void mahalanobis_ml_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *XXt, DOUBLE *XAtAXt, \
					DOUBLE *normAAtSq, INT N, INT derivFlag, DOUBLE *MXXt, DOUBLE *MXXtSq, DOUBLE *MXAtAXt) {

	INT MXXtFlag = 0;
	if (MXXt == NULL) {
		MXXt = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		MXXtFlag = 1;
	}

	INT MXXtSqFlag = 0;
	if (MXXtSq == NULL) {
		MXXtSq = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		MXXtSqFlag = 1;
	}

	INT MXAtAXtFlag = 0;
	if (MXAtAXt == NULL) {
		MXAtAXt = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		MXAtAXtFlag = 1;
	}

	CHAR side = 'R';
	CHAR uplo = 'U';
	INT SYMMM = N;
	INT SYMMN = N;
	DOUBLE alpha = 1;
	INT SYMMLDA = N;
	INT SYMMLDB = N;
	DOUBLE beta = 0;
	INT SYMMLDC = N;

	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, XXt, &SYMMLDA, X, &SYMMLDB, &beta, MXXt, &SYMMLDC);

	CHAR transa = 'N';
	CHAR transb = 'N';
	INT GEMMM = N;
	INT GEMMN = N;
	INT GEMMK = N;
	alpha = 1;
	INT GEMMLDA = N;
	INT GEMMLDB = N;
	beta = 0;
	INT GEMMLDC = N;

	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, MXXt, &GEMMLDA, MXXt, &GEMMLDB, &beta, MXXtSq, &GEMMLDC);

	side = 'L';
	uplo = 'U';
	SYMMM = N;
	SYMMN = N;
	alpha = 1;
	SYMMLDA = N;
	SYMMLDB = N;
	beta = 0;
	SYMMLDC = N;

	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, X, &SYMMLDA, XAtAXt, &SYMMLDB, &beta, MXAtAXt, &SYMMLDC);

	DOUBLE objTemp = 0;
	INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		objTemp += MXXtSq[iterN * N + iterN] - 2 * MXAtAXt[iterN * N + iterN];
	}
	*obj = objTemp + *normAAtSq;

	if (derivFlag == 1) {

		datacpy(deriv, XAtAXt, N * N);

		transa = 'N';
		transb = 'N';
		GEMMM = N;
		GEMMN = N;
		GEMMK = N;
		alpha = 2;
		GEMMLDA = N;
		GEMMLDB = N;
		beta = - 2;
		GEMMLDC = N;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, XXt, &GEMMLDA, MXXt, &GEMMLDB, &beta, deriv, &GEMMLDC);
	}

	if (MXXtFlag == 1) {
		FREE(MXXt);
	}

	if (MXXtSqFlag == 1) {
		FREE(MXXtSq);
	}

	if (MXAtAXtFlag == 1) {
		FREE(MXAtAXt);
	}
}
