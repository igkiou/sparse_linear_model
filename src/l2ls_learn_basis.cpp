/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "useblas.h"
#include "useinterfaces.h"
#include "l2ls_learn_basis.h"

void dual_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *dualLambda, DOUBLE *SSt, DOUBLE *SXt, DOUBLE *SXtXSt, DOUBLE trXXt, \
					DOUBLE c, INT N, INT K, INT derivFlag, DOUBLE *SStLambda, DOUBLE *tempMatrix) {
	
	INT maxNK = IMAX(N, K);
	INT SStLambdaFlag = 0;
	if (SStLambda == NULL) { 
		SStLambda = (DOUBLE *) MALLOC(maxNK * K * sizeof(DOUBLE));
		SStLambdaFlag = 1;
	}

	INT tempMatrixFlag = 0;
	if (tempMatrix == NULL) { 
		tempMatrix = (DOUBLE *) MALLOC(maxNK * K * sizeof(DOUBLE));
		tempMatrixFlag = 1;
	}
	
	datacpy(SStLambda, SSt, K * K);
	
	INT iterK;
	
/*
	#pragma omp parallel for private(iterK) shared(SStLambda, dualLambda, K)
*/
	for (iterK = 0; iterK < K; ++iterK) {
		SStLambda[iterK * K + iterK] += dualLambda[iterK];
	}
	
	CHAR uplo = 'U';
	INT POTRSN = K;
	INT POTRSLDA = K;
	INT INFO;
	
	POTRF(&uplo, &POTRSN, SStLambda, &POTRSLDA, &INFO);
	
	datacpy(tempMatrix, SXtXSt, K * K);
	
	INT POTRSNRHS = K;
	INT POTRSLDB = K;
	
	POTRS(&uplo, &POTRSN, &POTRSNRHS, SStLambda, &POTRSLDA, tempMatrix, &POTRSLDB, &INFO);

	DOUBLE objTemp = 0;
	
/*
	#pragma omp parallel for private(iterK) shared(tempMatrix, K) reduction(-: objTemp)
*/
	for (iterK = 0; iterK < K; ++iterK) {
		objTemp = objTemp - tempMatrix[iterK * K + iterK];
	}
	
	INT ASUMN = K;
	INT incx = 1;
	DOUBLE sumDualLambda = ASUM(&ASUMN, dualLambda, &incx);
	
	objTemp += trXXt - c * sumDualLambda;
	*obj = - objTemp;

	if (derivFlag == 1) {
		
		datacpy(tempMatrix, SXt, K * N);
		
		POTRSNRHS = N;
		POTRSLDB = K;
	
		POTRS(&uplo, &POTRSN, &POTRSNRHS, SStLambda, &POTRSLDA, tempMatrix, &POTRSLDB, &INFO);
		
		transpose(tempMatrix, SStLambda, K, N);
		
		INT NRM2N = N;
		DOUBLE tempNorm;
		#pragma omp parallel for private(iterK, tempNorm) shared(SStLambda, deriv, K, c)
		for (iterK = 0; iterK < K; ++iterK) {
			tempNorm = NRM2(&NRM2N, &SStLambda[iterK * N], &incx);
			deriv[iterK] = - SQR(tempNorm) + c;
		}
	}
	
	if (SStLambdaFlag == 1) {
		FREE(SStLambda);
	}
	
	if (tempMatrixFlag == 1) {
		FREE(tempMatrix);
	}
}

void minimize_dual(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *SSt, DOUBLE *SXt, DOUBLE *SXtXSt, DOUBLE trXXt, \
					DOUBLE c, INT N, INT K) {

	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
	INT MN = K * 1;
	
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
	
	INT maxNK = IMAX(N, K);
	DOUBLE *SStLambda = (DOUBLE *) MALLOC(maxNK * K * sizeof(DOUBLE));
	DOUBLE *tempMatrix = (DOUBLE *) MALLOC(maxNK * K * sizeof(DOUBLE));
	
	dual_obj_grad(&f0, df0, X, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);
	
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
				
				dual_obj_grad(&f3, df3, X3, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);	

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

			dual_obj_grad(&f3, df3, X3, SSt, SXt, SXtXSt, trXXt, c, N, K, derivFlag, SStLambda, tempMatrix);

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
	
	FREE(SStLambda);
	FREE(tempMatrix);
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}

void l2ls_learn_basis_dual(DOUBLE *Dopt, DOUBLE *Dorig, DOUBLE *X, DOUBLE *S, DOUBLE l2norm, INT length, INT N, INT K, INT numSamples) {

	DOUBLE *SSt = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
	
	CHAR uplo = 'U';
	CHAR trans = 'N';
	INT SYRKN = K;
	INT SYRKK = numSamples;
	DOUBLE alpha = 1;
	INT SYRKLDA = K;
	DOUBLE beta = 0;
	INT SYRKLDC = K;
	
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, S, &SYRKLDA, &beta, SSt, &SYRKLDC);
	
	DOUBLE *XSt = (DOUBLE *) MALLOC(N * K * sizeof(DOUBLE));

	CHAR transa = 'N';
	CHAR transb = 'T';
	INT GEMMM = N;
	INT GEMMN = K;
	INT GEMMK = numSamples;
	alpha = 1;
	INT GEMMLDA = N;
	INT GEMMLDB = K;
	beta = 0;
	INT GEMMLDC = N;
	
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X, &GEMMLDA, S, &GEMMLDB, &beta, XSt, &GEMMLDC);
	DOUBLE *SXt = (DOUBLE *) MALLOC(N * K * sizeof(DOUBLE));
	transpose(XSt, SXt, N, K);
	
	INT iterK;	
	DOUBLE *dualLambdaOrig = (DOUBLE *) MALLOC(K * sizeof(DOUBLE));
	if (Dorig == NULL) {
		srand(time(NULL));
		for (iterK = 0; iterK < K; ++iterK) {
			dualLambdaOrig[iterK] = 10 * (DOUBLE) rand() / (DOUBLE) RAND_MAX;
		}
	} else {
		
		INT maxNK = IMAX(N, K);
		DOUBLE *B = (DOUBLE *) MALLOC(maxNK * maxNK * sizeof(DOUBLE));
		for (iterK = 0; iterK < K; ++iterK) {
			datacpy(&B[iterK * maxNK], &XSt[iterK * N], K);
		}
		
		INT GELSYM = N;
		INT GELSYN = K;
		INT GELSYNRHS = K;
		INT GELSYLDA = N;
		INT GELSYLDB = maxNK;
		INT *jpvt = (INT *) MALLOC(K * sizeof(INT));
		DOUBLE rcond;
		INT rank;
		INT lwork = -1;
		DOUBLE work_temp;
		DOUBLE *work;
		INT INFO;

		GELSY(&GELSYM, &GELSYN, &GELSYNRHS, Dorig, &GELSYLDA, B, &GELSYLDB, jpvt, &rcond, &rank, &work_temp, &lwork, &INFO);
		
		lwork = (INT) work_temp;
		work = (DOUBLE*) MALLOC(lwork * sizeof(DOUBLE));

		
		GELSY(&GELSYM, &GELSYN, &GELSYNRHS, Dorig, &GELSYLDA, XSt, &GELSYLDB, jpvt, &rcond, &rank, work, &lwork, &INFO);

		for (iterK = 0; iterK < K; ++iterK) {
			dualLambdaOrig[K] = B[iterK * K + iterK] - SSt[iterK * K + iterK];
		}
		
		FREE(work);
		FREE(B);
		FREE(jpvt);
	}

	DOUBLE *SXtXSt = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
	
	uplo = 'U';
	trans = 'N';
	SYRKN = K;
	SYRKK = N;
	alpha = 1;
	SYRKLDA = K;
	beta = 0;
	SYRKLDC = K;
	
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, SXt, &SYRKLDA, &beta, SXtXSt, &SYRKLDC);

	DOUBLE c = SQR(l2norm);

	CHAR norm = 'F';
	INT LANGEM = N;
	INT LANGEN = numSamples;
	INT LANGELDA = N;
	
	DOUBLE trXXt = LANGE(&norm, &LANGEM, &LANGEN, X, &LANGELDA, NULL);
	trXXt = SQR(trXXt);
	
/*
	DOUBLE *dualLambdaOpt = (DOUBLE *) MALLOC(K * sizeof(DOUBLE));
*/
	DOUBLE *dualLambdaOpt = XSt;
	minimize_dual(dualLambdaOpt, dualLambdaOrig, length, SSt, SXt, SXtXSt, trXXt, c, N, K);

	for (iterK = 0; iterK < K; ++iterK) {
		SSt[iterK * K + iterK] += dualLambdaOpt[iterK];
	}

	uplo = 'U';
	INT POTRSN = K;
	INT POTRSLDA = K;
	INT INFO;
	POTRF(&uplo, &POTRSN, SSt, &POTRSLDA, &INFO);
	
	INT POTRSNRHS = N;
	INT POTRSLDB = K;
	
	POTRS(&uplo, &POTRSN, &POTRSNRHS, SSt, &POTRSLDA, SXt, &POTRSLDB, &INFO);

	transpose(SXt, Dopt, K, N);
	
	FREE(SSt);
	FREE(XSt);
	FREE(SXt);
	FREE(dualLambdaOrig);
	FREE(SXtXSt);
}
