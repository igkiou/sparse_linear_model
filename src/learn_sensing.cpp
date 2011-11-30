/*
 * Mex implementation of LEARN_SENSING.
 *
 * Requires LAPACK and BLAS. Has been tested with both MKL's and MATLAB's 
 * implementations. 
 *
 */

/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useblas.h"
#include "useinterfaces.h"
#include "utils.h"
#include "learn_sensing.h"

void random_sensing(DOUBLE *Phi, INT M, INT N) {
	
	INT iter;
	INT numel = M * N;
	DOUBLE std = 1 / SQRT((DOUBLE) M);

	srand(time(NULL));
	for (iter = 0; iter + 1 < numel; iter = iter + 2) {
		rand_normal_custom(&Phi[iter], &Phi[iter + 1], std);
	}
	
	if (iter < numel) {
		rand_normal_custom(&Phi[iter], NULL, std);
	}
}
	
void learn_sensing(DOUBLE *Phi, DOUBLE *D, INT M, INT N, INT K) {
	
	/* 
	 * Check initial Phi
	 */
	if (Phi == NULL) {
		random_sensing(Phi, M, N);
	}
	
	/* 
	 * Get eps and sfmin
	 */
	CHAR lamch_opt;
	lamch_opt = 'E';
	DOUBLE eps = LAMCH(&lamch_opt);
	lamch_opt = 'S';
	DOUBLE sfmin = LAMCH(&lamch_opt);
	
	/* 
	 * Precalculate DDt, eigenvectors and eigenvalues 
	 */
	DOUBLE *DDt = (DOUBLE*) MALLOC(N * N * sizeof(DOUBLE));
	
	/* Setup SYRK parameters */
	CHAR trans = 'N';
	CHAR uplo = 'U';
	INT SYRKN = N;
	INT SYRKK = K;
	DOUBLE alpha = 1;
	INT SYRKLDA = N;
	DOUBLE beta = 0;
	INT SYRKLDC = N;
	
	/* Run SYRK */
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, D, &SYRKLDA, &beta, DDt, &SYRKLDC);
	
	/* Setup SYEVR parameters */
	CHAR jobz = 'V';
	CHAR range = 'A';
	uplo = 'U';
	INT SYEVRN = N;
	INT SYEVLDA = N;
	DOUBLE VL, VU;
	INT IL, IU;
	DOUBLE abstol = sfmin;
	INT SYEVLDZ = N;
	INT lwork = -1;
	INT liwork = -1;
	DOUBLE *work;
	INT *iwork;
	DOUBLE work_temp;
	INT iwork_temp;
	INT SYEVRM;
	INT INFO;
	
	DOUBLE *lvec = (DOUBLE*) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE*) MALLOC(N * N * sizeof(DOUBLE));
	INT *ISUPPZ = (INT*) MALLOC(2 * N * sizeof(INT));
	
	/* Run SYEVR */
	SYEVR(&jobz, &range, &uplo, &SYEVRN, DDt, &SYEVLDA, &VL, &VU, &IL, &IU, &abstol, &SYEVRM, \
			lvec, Vr, &SYEVLDZ, ISUPPZ, &work_temp, &lwork, &iwork_temp, &liwork, &INFO);
	
	lwork = (INT) work_temp;
	work = (DOUBLE*) MALLOC(lwork * sizeof(DOUBLE));
	liwork = (INT) iwork_temp;
	iwork = (INT*) MALLOC(liwork * sizeof(INT));
	
	SYEVR(&jobz, &range, &uplo, &SYEVRN, DDt, &SYEVLDA, &VL, &VU, &IL, &IU, &abstol, &SYEVRM, \
			lvec, Vr, &SYEVLDZ, ISUPPZ, work, &lwork, iwork, &liwork, &INFO);	
	
	/* Reverse eigenvalues and eigenvectors and convert to diagonal */
	DOUBLE *L = (DOUBLE*) MALLOC(N * N * sizeof(DOUBLE));
	memset((void*) L, 0, N * N * sizeof(DOUBLE));
	DOUBLE *V = (DOUBLE*) MALLOC(N * N * sizeof(DOUBLE));
	
	/* Define iteration counters */
	INT iterM, iterN;
	
	for (iterN = 0; iterN < N; ++iterN) {
	
		L[iterN * N + iterN] = lvec[N - 1 - iterN];
		datacpy(&V[iterN * N], &Vr[(N - 1 - iterN) * N], N);
	}
	
	/* 
	 * Find rank of DDt 
	 */
	
	DOUBLE tolerance = (DOUBLE) N * eps;
	
	for (iterN = 0; iterN < N; ++iterN) {
		if (lvec[N - 1 - iterN] <= tolerance) {
			break;
		}
	}
	
	INT rank = iterN;
		
	/*
	 * Start main loop
	 */
	DOUBLE *Gammat = (DOUBLE*) MALLOC(N * M * sizeof(DOUBLE));
	DOUBLE *Vmat = (DOUBLE*) MALLOC(N * M * sizeof(DOUBLE));
	DOUBLE *Ej = (DOUBLE*) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *u = (DOUBLE*) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE xi;
	DOUBLE sqrtxi;
	
	/* Setup GEMM parameters */
	CHAR transa = 'T';
	CHAR transb = 'T';
	INT GEMMM = N;
	INT GEMMN = M;
	INT GEMMK = N;
	alpha = 1;
	INT GEMMLDA = N;	
	INT GEMMLDB = M;
	beta = 0;
	INT GEMMLDC = N;
	
	/* Run GEMM */
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, V, &GEMMLDA, Phi, &GEMMLDB, &beta, Gammat, &GEMMLDC);

	/* Define LACPY parameters */
	INT LACPYM;
	INT LACPYN;
	INT LACPYLDA;
	INT LACPYLDB;

	/* Define and setup SCAL parameters */
	INT SCALN = M;
	INT incx = N;

	/* Define and setup SYR parameters */
	INT SYRN = N;
	INT INCX = 1;
	INT SYRLDA = N;

	DOUBLE tempVal;
	for (iterM = 0; iterM < M; ++iterM) {
	
		/* Setup LACPY parameters */		
		uplo = 'A';
		LACPYM = N;
		LACPYN = M;
		LACPYLDA = N;
		LACPYLDB = N;
		
		/* Run LACPY, Vmat = Gamma' */
		LACPY(&uplo, &LACPYM, &LACPYN, Gammat, &LACPYLDA, Vmat, &LACPYLDB);

		/* Setup TRMM parameters */
		uplo = 'U';
		transa = 'N';
		alpha = 1;

		/* Run TRMM, Vmat = L * Gamma' */
		for (iterN = 0; iterN < N; ++iterN) {
			tempVal = L[iterN * N + iterN];
			SCAL(&SCALN, &tempVal, &Vmat[iterN], &incx);
		}

		/* Setup LACPY parameters */		
		uplo = 'U';
		LACPYM = N;
		LACPYN = N;
		LACPYLDA = N;
		LACPYLDB = N;
		
		/* Run LACPY, Ej = L */
		LACPY(&uplo, &LACPYM, &LACPYN, L, &LACPYLDA, Ej, &LACPYLDB);
		
		/* Setup SYRK parameters */
		trans = 'N';
		uplo = 'U';
		SYRKN = N;
		SYRKK = M;
		alpha = - 1;
		SYRKLDA = N;
		beta = 1;
		SYRKLDC = N;
		
		/* Run SYRK, Ej = Ej - Vmat * Vmat' */
		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, Vmat, &SYRKLDA, &beta, Ej, &SYRKLDC);
		
		/* Setup SYR parameters */
		uplo = 'U';
		alpha = 1;
		
		/* Run SYR, Ej = Ej + Vmat(:, j) * Vmat(:, j)' */		
		SYR(&uplo, &SYRN, &alpha, &Vmat[iterM * N], &INCX, Ej, &SYRLDA);
				
		/* Setup SYEVR parameters */
		jobz = 'V';
		range = 'I';
		uplo = 'U';
		SYEVRN = N;
		SYEVLDA = N;
		VL = 0;
		VU = 0;
		IL = N;
		IU = N;
		abstol = sfmin;
		SYEVLDZ = N;

		/* Run SYEVR */
		SYEVR(&jobz, &range, &uplo, &SYEVRN, Ej, &SYEVLDA, &VL, &VU, &IL, &IU, &abstol, &SYEVRM, \
				&xi, u, &SYEVLDZ, ISUPPZ, work, &lwork, iwork, &liwork, &INFO);	
		
		sqrtxi = SQRT(xi);
		for (iterN = 0; iterN < rank; ++iterN) {		
			Gammat[iterM * N + iterN] = sqrtxi * u[iterN] / lvec[N - 1 - iterN];
		}
	}
	
	/* Setup GEMM parameters */
	transa = 'T';
	transb = 'T';
	GEMMM = M;
	GEMMN = N;
	GEMMK = N;
	alpha = 1;
	GEMMLDA = N;	
	GEMMLDB = N;
	beta = 0;
	GEMMLDC = M;
	
	/* Run GEMM */
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Gammat, &GEMMLDA, V, &GEMMLDB, &beta, Phi, &GEMMLDC);
	
	/*
	 * Free memory
	 */
	FREE(DDt);
	FREE(lvec);
	FREE(Vr);
	FREE(ISUPPZ);
	FREE(work);
	FREE(iwork);
	FREE(L);
	FREE(V);
	FREE(Gammat);
	FREE(Vmat);
	FREE(Ej);
	FREE(u);
}

void minimize(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, INT M, INT N) {

	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
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

	DOUBLE *Gamma = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *PhiDDt2 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT derivFlag = 1;
	
	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	eig_obj_grad(&f0, df0, X, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
	
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
				
				eig_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);

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
				
			eig_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
			
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
#ifdef __DEBUG__
			PRINTF("End of if.\n\n");
#endif		
		}
	}

	datacpy(Xopt, X, MN);
	
	FREE(Gamma);
	FREE(ObjMat);
	FREE(PhiDDt2);
	FREE(DDt3temp);
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}

void eig_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, \
					INT M, INT N, INT derivFlag, DOUBLE *Gamma, DOUBLE *ObjMat, DOUBLE *PhiDDt2, DOUBLE *DDt3temp) {
	
	int GammaFlag = 0;
	if (Gamma == NULL) {
		Gamma = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
		GammaFlag = 1;
	}
	
	CHAR transa = 'N';
	CHAR transb = 'N';
	INT GEMMM = M;
	INT GEMMN = N;
	INT GEMMK = N;
	DOUBLE alpha = 1;
	INT GEMMLDA = M;	
	INT GEMMLDB = N;
	DOUBLE beta = 0;
	INT GEMMLDC = M;
	
	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, X, &GEMMLDA, VL, &GEMMLDB, &beta, Gamma, &GEMMLDC);
	
	int ObjMatFlag = 0;
	if (ObjMat == NULL) {
		ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
		ObjMatFlag = 1;
	}
	
	CHAR uplo = 'U';
	CHAR trans = 'T';
	INT SYRKN = N;
	INT SYRKK = M;
	alpha = -1;
	INT SYRKLDA = M;
	beta = 0;
	INT SYRKLDC = N;
	
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, Gamma, &SYRKLDA, &beta, ObjMat, &SYRKLDC);
	
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
		int PhiDDt2Flag = 0;
		if (PhiDDt2 == NULL) {
			PhiDDt2 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
			PhiDDt2Flag = 1;
		}

		CHAR side = 'R';
		uplo = 'U';
		INT SYMMM = M;
		INT SYMMN = N;
		alpha = 1;
		INT SYMMLDA = N;
		INT SYMMLDB = M;
		beta = 0;
		INT SYMMLDC = M;

		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, DDt2, &SYMMLDA, X, &SYMMLDB, &beta, PhiDDt2, &SYMMLDC);

		int DDt3tempFlag = 0;
		if (DDt3temp == NULL) {
			DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
			DDt3tempFlag = 1;
		}

		uplo = 'U';
		INT LACPYM = N;
		INT LACPYN = N;
		INT LACPYLDA = N;
		INT LACPYLDB = N;

		LACPY(&uplo, &LACPYM, &LACPYN, DDt3, &LACPYLDA, DDt3temp, &LACPYLDB);

		uplo = 'U';
		trans = 'T';
		SYRKN = N;
		SYRKK = M;
		alpha = 1;
		SYRKLDA = M;
		beta = -1;
		SYRKLDC = N;

		SYRK(&uplo, &trans, &N, &M, &alpha, PhiDDt2, &M, &beta, DDt3temp, &N);

		alpha = 4;
		side = 'R';

		side = 'R';
		uplo = 'U';
		SYMMM = M;
		SYMMN = N;
		alpha = 4;
		SYMMLDA = N;
		SYMMLDB = M;
		beta = 0;
		SYMMLDC = M;

		SYMM(&side, &uplo, &M, &N, &alpha, DDt3temp, &N, X, &M, &beta, deriv, &M);
	
		if (PhiDDt2Flag == 1) {
			FREE(PhiDDt2);
		}
		if (DDt3tempFlag == 1) {
			FREE(DDt3temp);
		}
	}
	
	if (GammaFlag == 1) {
		FREE(Gamma);
	}
	
	if (ObjMatFlag == 1) {
		FREE(ObjMat);
	}
}

void eig_lap_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *XLXt, DOUBLE *DDt2, DOUBLE *DDt3,\
					DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT numSamples, INT derivFlag,\
					DOUBLE *PhiXLXt, DOUBLE *PhiXLXtPhit, DOUBLE *Gamma, DOUBLE *ObjMat, DOUBLE *PhiDDt2, DOUBLE *DDt3temp) {
	
	INT PhiXLXtFlag = 0;
	if (PhiXLXt == NULL) { 
		PhiXLXt = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
		PhiXLXtFlag = 1;
	}

	CHAR side = 'R';
	CHAR uplo = 'U';
	INT SYMMM = M;
	INT SYMMN = N;
	DOUBLE alpha = 2 * alphaReg / (DOUBLE) SQR(numSamples);
	INT SYMMLDA = N;	
	INT SYMMLDB = M;
	DOUBLE beta = 0;
	INT SYMMLDC = M;

	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, XLXt, &SYMMLDA, Phi, &SYMMLDB, &beta, PhiXLXt, &SYMMLDC);
	
	INT PhiXLXtPhitFlag = 0;
	if (PhiXLXtPhit == NULL) { 
		PhiXLXtPhit = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
		PhiXLXtPhitFlag = 1;
	}
	
	CHAR transa = 'N';
	CHAR transb = 'T';
	INT GEMMM = M;
	INT GEMMN = M;
	INT GEMMK = N;
	alpha = 0.5;
	INT GEMMLDA = M;	
	INT GEMMLDB = M;
	beta = 0;
	INT GEMMLDC = M;

	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, PhiXLXt, &GEMMLDA, Phi, &GEMMLDB, &beta, PhiXLXtPhit, &GEMMLDC);

	DOUBLE objtemp = 0;
	INT iterM;
	for (iterM = 0; iterM < M; ++iterM) {
		objtemp += PhiXLXtPhit[iterM * M + iterM];
	}
	objtemp = objtemp;
	
	*obj = objtemp;
	
	if (PhiXLXtPhitFlag == 1) {
		FREE(PhiXLXtPhit);
	}
	
	if (betaReg > 0) {
		int GammaFlag = 0;
		if (Gamma == NULL) {
			Gamma = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
			GammaFlag = 1;
		}
	
		transa = 'N';
		transb = 'N';
		GEMMM = M;
		GEMMN = N;
		GEMMK = N;
		alpha = 1;
		GEMMLDA = M;	
		GEMMLDB = N;
		beta = 0;
		GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Phi, &GEMMLDA, VL, &GEMMLDB, &beta, Gamma, &GEMMLDC);
		
		int ObjMatFlag = 0;
		if (ObjMat == NULL) {
			ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
			ObjMatFlag = 1;
		}

		uplo = 'U';
		CHAR trans = 'T';
		INT SYRKN = N;
		INT SYRKK = M;
		alpha = -1;
		INT SYRKLDA = M;
		beta = 0;
		INT SYRKLDC = N;

		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, Gamma, &SYRKLDA, &beta, ObjMat, &SYRKLDC);

		INT iterN;
		for (iterN = 0; iterN < N; ++iterN) {
			ObjMat[iterN * N + iterN] += L[iterN];
		}

		CHAR norm = 'F';
		uplo = 'U';
		INT LANSYN = N;
		INT LANSYLDA = N;

		objtemp = LANSY(&norm, &uplo, &LANSYN, ObjMat, &LANSYLDA, NULL);
		objtemp = SQR(objtemp);
		*obj = *obj + betaReg * objtemp;
		
		if (GammaFlag == 1) {
			FREE(Gamma);
		}

		if (ObjMatFlag == 1) {
			FREE(ObjMat);
		}
	}

	if (derivFlag == 1) {
		
		datacpy(deriv, PhiXLXt, M * N);
		
		if (betaReg > 0) {
			int PhiDDt2Flag = 0;
			if (PhiDDt2 == NULL) {
				PhiDDt2 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
				PhiDDt2Flag = 1;
			}

			side = 'R';
			uplo = 'U';
			SYMMM = M;
			SYMMN = N;
			alpha = 1;
			SYMMLDA = N;
			SYMMLDB = M;
			beta = 0;
			SYMMLDC = M;

			SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, DDt2, &SYMMLDA, Phi, &SYMMLDB, &beta, PhiDDt2, &SYMMLDC);

			int DDt3tempFlag = 0;
			if (DDt3temp == NULL) {
				DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
				DDt3tempFlag = 1;
			}

			uplo = 'U';
			INT LACPYM = N;
			INT LACPYN = N;
			INT LACPYLDA = N;
			INT LACPYLDB = N;

			LACPY(&uplo, &LACPYM, &LACPYN, DDt3, &LACPYLDA, DDt3temp, &LACPYLDB);

			uplo = 'U';
			CHAR trans = 'T';
			INT SYRKN = N;
			INT SYRKK = M;
			alpha = 1;
			INT SYRKLDA = M;
			beta = -1;
			INT SYRKLDC = N;

			SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, PhiDDt2, &SYRKLDA, &beta, DDt3temp, &SYRKLDC);

			alpha = 4;
			side = 'R';

			side = 'R';
			uplo = 'U';
			SYMMM = M;
			SYMMN = N;
			alpha = 4 * betaReg;
			SYMMLDA = N;
			SYMMLDB = M;
			beta = 1;
			SYMMLDC = M;

			SYMM(&side, &uplo, &M, &N, &alpha, DDt3temp, &N, Phi, &M, &beta, deriv, &M);
	
			if (PhiDDt2Flag == 1) {
				FREE(PhiDDt2);
			}
			if (DDt3tempFlag == 1) {
				FREE(DDt3temp);
			}
		}
	}
	
	if (PhiXLXtFlag == 1) {
		FREE(PhiXLXt);
	}	
}		

void eig_lsqr_obj_grad_largedata(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *XXt, DOUBLE *YXt, DOUBLE trYYt, DOUBLE *DDt2,\
					DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT derivFlag,\
					DOUBLE *AXXt, DOUBLE *AXXtAt, DOUBLE *Gamma, DOUBLE *ObjMat, DOUBLE *PhiDDt2, DOUBLE *DDt3temp) {
	
	INT AXXtFlag = 0;
	if (AXXt == NULL) { 
		AXXt = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
		AXXtFlag = 1;
	}
	
	datacpy(AXXt, YXt, M * N);

	CHAR side = 'R';
	CHAR uplo = 'U';
	INT SYMMM = M;
	INT SYMMN = N;
	DOUBLE alpha = 1;
	INT SYMMLDA = N;	
	INT SYMMLDB = M;
	DOUBLE beta = -2;
	INT SYMMLDC = M;

	SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, XXt, &SYMMLDA, Phi, &SYMMLDB, &beta, AXXt, &SYMMLDC);

	INT AXXtAtFlag = 0;
	if (AXXtAt == NULL) { 
		AXXtAt = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
		AXXtAtFlag = 1;
	}
	
	CHAR transa = 'N';
	CHAR transb = 'T';
	INT GEMMM = M;
	INT GEMMN = M;
	INT GEMMK = N;
	alpha = 1;
	INT GEMMLDA = M;	
	INT GEMMLDB = M;
	beta = 0;
	INT GEMMLDC = M;

	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Phi, &GEMMLDA, AXXt, &GEMMLDB, &beta, AXXtAt, &GEMMLDC);

	if (alphaReg > 0) {

		transa = 'N';
		transb = 'T';
		GEMMM = M;
		GEMMN = M;
		GEMMK = N;
		alpha = alphaReg;
		GEMMLDA = M;	
		GEMMLDB = M;
		beta = 1;
		GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Phi, &GEMMLDA, Phi, &GEMMLDB, &beta, AXXtAt, &GEMMLDC);
	}
	
	DOUBLE objtemp = trYYt;
	INT iterM;
	for (iterM = 0; iterM < M; ++iterM) {
		objtemp = objtemp + AXXtAt[iterM * M + iterM];
	}
	*obj = objtemp;
	
	if (betaReg > 0) {
		int GammaFlag = 0;
		if (Gamma == NULL) {
			Gamma = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
			GammaFlag = 1;
		}
	
		transa = 'N';
		transb = 'N';
		GEMMM = M;
		GEMMN = N;
		GEMMK = N;
		alpha = 1;
		GEMMLDA = M;	
		GEMMLDB = N;
		beta = 0;
		GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Phi, &GEMMLDA, VL, &GEMMLDB, &beta, Gamma, &GEMMLDC);
		
		int ObjMatFlag = 0;
		if (ObjMat == NULL) {
			ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
			ObjMatFlag = 1;
		}

		uplo = 'U';
		CHAR trans = 'T';
		INT SYRKN = N;
		INT SYRKK = M;
		alpha = -1;
		INT SYRKLDA = M;
		beta = 0;
		INT SYRKLDC = N;

		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, Gamma, &SYRKLDA, &beta, ObjMat, &SYRKLDC);

		INT iterN;
		for (iterN = 0; iterN < N; ++iterN) {
			ObjMat[iterN * N + iterN] += L[iterN];
		}

		CHAR norm = 'F';
		uplo = 'U';
		INT LANSYN = N;
		INT LANSYLDA = N;

		objtemp = LANSY(&norm, &uplo, &LANSYN, ObjMat, &LANSYLDA, NULL);
		objtemp = SQR(objtemp);
		
		*obj = *obj + betaReg * objtemp;
		
		if (GammaFlag == 1) {
			FREE(Gamma);
		}

		if (ObjMatFlag == 1) {
			FREE(ObjMat);
		}
	}

	if (derivFlag == 1) {
		
		/* AXXt = A * XXt - 2 * YXt */

		INT SCALN = M * N;
		alpha = 2;
		INT incx = 1;
		INT incy = 1;
		SCAL(&SCALN, &alpha, AXXt, &incx);

		/* AXXt = 2 * A * XXt - 4 * YXt */
		
		INT AXPYN = M * N;
		alpha = 2;
		daxpy(&AXPYN, &alpha, YXt, &incx, AXXt, &incy);

		/* AXXt = 2 * A * XXt - 2 * YXt */
		
		if (alphaReg > 0) {
			alpha = 2 * alphaReg;
			daxpy(&AXPYN, &alpha, Phi, &incx, AXXt, &incy);
		}
		
		datacpy(deriv, AXXt, M * N);
		
		if (betaReg > 0) {
			int PhiDDt2Flag = 0;
			if (PhiDDt2 == NULL) {
				PhiDDt2 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
				PhiDDt2Flag = 1;
			}

			side = 'R';
			uplo = 'U';
			SYMMM = M;
			SYMMN = N;
			alpha = 1;
			SYMMLDA = N;
			SYMMLDB = M;
			beta = 0;
			SYMMLDC = M;

			SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, DDt2, &SYMMLDA, Phi, &SYMMLDB, &beta, PhiDDt2, &SYMMLDC);

			int DDt3tempFlag = 0;
			if (DDt3temp == NULL) {
				DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
				DDt3tempFlag = 1;
			}

			uplo = 'U';
			INT LACPYM = N;
			INT LACPYN = N;
			INT LACPYLDA = N;
			INT LACPYLDB = N;

			LACPY(&uplo, &LACPYM, &LACPYN, DDt3, &LACPYLDA, DDt3temp, &LACPYLDB);

			uplo = 'U';
			CHAR trans = 'T';
			INT SYRKN = N;
			INT SYRKK = M;
			alpha = 1;
			INT SYRKLDA = M;
			beta = -1;
			INT SYRKLDC = N;
			
			SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, PhiDDt2, &SYRKLDA, &beta, DDt3temp, &SYRKLDC);

			alpha = 4;
			side = 'R';

			side = 'R';
			uplo = 'U';
			SYMMM = M;
			SYMMN = N;
			alpha = 4 * betaReg;
			SYMMLDA = N;
			SYMMLDB = M;
			beta = 1;
			SYMMLDC = M;

			SYMM(&side, &uplo, &M, &N, &alpha, DDt3temp, &N, Phi, &M, &beta, deriv, &M);
	
			if (PhiDDt2Flag == 1) {
				FREE(PhiDDt2);
			}
			if (DDt3tempFlag == 1) {
				FREE(DDt3temp);
			}
		}
	}
	
	if (AXXtFlag == 1) {
		FREE(AXXt);
	}
	
	if (AXXtAtFlag == 1) {
		FREE(AXXtAt);
	}		
}

void eig_lsqr_obj_grad_smalldata(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *DDt2, DOUBLE *DDt3,\
					DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT numSamples, INT derivFlag,\
					DOUBLE *Err, DOUBLE *Gamma, DOUBLE *ObjMat, DOUBLE *PhiDDt2, DOUBLE *DDt3temp) {
	
	INT ErrFlag = 0;
	if (Err == NULL) { 
		Err = (DOUBLE *) MALLOC(M * numSamples * sizeof(DOUBLE));
		ErrFlag = 1;
	}
	
	datacpy(Err, Y, M * numSamples);

	CHAR transa = 'N';
	CHAR transb = 'N';
	INT GEMMM = M;
	INT GEMMN = numSamples;
	INT GEMMK = N;
	DOUBLE alpha = 1;
	INT GEMMLDA = M;	
	INT GEMMLDB = N;
	DOUBLE beta = -1;
	INT GEMMLDC = M;

	GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Phi, &GEMMLDA, X, &GEMMLDB, &beta, Err, &GEMMLDC);

	CHAR norm = 'F';
	INT LANGEM = M;
	INT LANGEN = numSamples;
	INT LANGELDA = M;
	
	*obj = LANGE(&norm, &LANGEM, &LANGEN, Err, &LANGELDA, NULL);
	*obj = SQR(*obj);
	
	DOUBLE objtemp;
	if (alphaReg > 0) {
		
		norm = 'F';
		LANGEM = M;
		LANGEN = N;
		LANGELDA = M;
	
		objtemp = LANGE(&norm, &LANGEM, &LANGEN, Phi, &LANGELDA, NULL);
		objtemp = SQR(objtemp);
		*obj = *obj + alphaReg * objtemp;
	}
	
	if (betaReg > 0) {
		int GammaFlag = 0;
		if (Gamma == NULL) {
			Gamma = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
			GammaFlag = 1;
		}
	
		transa = 'N';
		transb = 'N';
		GEMMM = M;
		GEMMN = N;
		GEMMK = N;
		alpha = 1;
		GEMMLDA = M;	
		GEMMLDB = N;
		beta = 0;
		GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Phi, &GEMMLDA, VL, &GEMMLDB, &beta, Gamma, &GEMMLDC);
		
		int ObjMatFlag = 0;
		if (ObjMat == NULL) {
			ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
			ObjMatFlag = 1;
		}

		CHAR uplo = 'U';
		CHAR trans = 'T';
		INT SYRKN = N;
		INT SYRKK = M;
		alpha = -1;
		INT SYRKLDA = M;
		beta = 0;
		INT SYRKLDC = N;

		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, Gamma, &SYRKLDA, &beta, ObjMat, &SYRKLDC);

		INT iterN;
		for (iterN = 0; iterN < N; ++iterN) {
			ObjMat[iterN * N + iterN] += L[iterN];
		}

		norm = 'F';
		uplo = 'U';
		INT LANSYN = N;
		INT LANSYLDA = N;

		objtemp = LANSY(&norm, &uplo, &LANSYN, ObjMat, &LANSYLDA, NULL);
		objtemp = SQR(objtemp);
		*obj = *obj + betaReg * objtemp;
		
		if (GammaFlag == 1) {
			FREE(Gamma);
		}

		if (ObjMatFlag == 1) {
			FREE(ObjMat);
		}
	}

	if (derivFlag == 1) {
		
		transa = 'N';
		transb = 'T';
		GEMMM = M;
		GEMMN = N;
		GEMMK = numSamples;
		GEMMLDA = M;	
		GEMMLDB = N;
		GEMMLDC = M;
		alpha = 2;
		
		if (alphaReg > 0) {
			beta = 2 * alphaReg;
			datacpy(deriv, Phi, M * N);
		} else {
			beta = 0;
		}
			
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Err, &GEMMLDA, X, &GEMMLDB, &beta, deriv, &GEMMLDC);
		
		if (betaReg > 0) {
			int PhiDDt2Flag = 0;
			if (PhiDDt2 == NULL) {
				PhiDDt2 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
				PhiDDt2Flag = 1;
			}

			CHAR side = 'R';
			CHAR uplo = 'U';
			INT SYMMM = M;
			INT SYMMN = N;
			alpha = 1;
			INT SYMMLDA = N;
			INT SYMMLDB = M;
			beta = 0;
			INT SYMMLDC = M;

			SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, DDt2, &SYMMLDA, Phi, &SYMMLDB, &beta, PhiDDt2, &SYMMLDC);

			int DDt3tempFlag = 0;
			if (DDt3temp == NULL) {
				DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
				DDt3tempFlag = 1;
			}

			uplo = 'U';
			INT LACPYM = N;
			INT LACPYN = N;
			INT LACPYLDA = N;
			INT LACPYLDB = N;

			LACPY(&uplo, &LACPYM, &LACPYN, DDt3, &LACPYLDA, DDt3temp, &LACPYLDB);

			uplo = 'U';
			CHAR trans = 'T';
			INT SYRKN = N;
			INT SYRKK = M;
			alpha = 1;
			INT SYRKLDA = M;
			beta = -1;
			INT SYRKLDC = N;

			SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, PhiDDt2, &SYRKLDA, &beta, DDt3temp, &SYRKLDC);

			alpha = 4;
			side = 'R';

			side = 'R';
			uplo = 'U';
			SYMMM = M;
			SYMMN = N;
			alpha = 4 * betaReg;
			SYMMLDA = N;
			SYMMLDB = M;
			beta = 1;
			SYMMLDC = M;

			SYMM(&side, &uplo, &M, &N, &alpha, DDt3temp, &N, Phi, &M, &beta, deriv, &M);
	
			if (PhiDDt2Flag == 1) {
				FREE(PhiDDt2);
			}
			if (DDt3tempFlag == 1) {
				FREE(DDt3temp);
			}
		}
	}
	
	if (ErrFlag == 1) {
		FREE(Err);
	}	
}		

void minimize_eig_lsqr_largedata(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *XXt, DOUBLE *YXt, DOUBLE trYYt,\
					DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N) {
	
	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
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

	DOUBLE *AXXt = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *AXXtAt = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
	DOUBLE *Gamma = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *PhiDDt2 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT derivFlag = 1;
	
	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	
	eig_lsqr_obj_grad_largedata(&f0, df0, X, XXt, YXt, trYYt, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, derivFlag,\
					AXXt, AXXtAt, Gamma, ObjMat, PhiDDt2, DDt3temp);
/*
	eig_obj_grad(&f0, df0, X, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
*/
	
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
		PRINTF("Iter: %d\n\n", i);
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
				
				eig_lsqr_obj_grad_largedata(&f3, df3, X3, XXt, YXt, trYYt, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, derivFlag,\
					AXXt, AXXtAt, Gamma, ObjMat, PhiDDt2, DDt3temp);
/*
				eig_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
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
			
			eig_lsqr_obj_grad_largedata(&f3, df3, X3, XXt, YXt, trYYt, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, derivFlag,\
					AXXt, AXXtAt, Gamma, ObjMat, PhiDDt2, DDt3temp);
/*
			eig_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
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
			
/*
			if ((ls_failed == 1) || (i > length)) {
*/
			if (i > length) {
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
	
	FREE(AXXt);
	FREE(AXXtAt);
	FREE(Gamma);
	FREE(ObjMat);
	FREE(PhiDDt2);
	FREE(DDt3temp);
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}

void minimize_eig_lsqr_smalldata(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *Xdata, DOUBLE *Y, DOUBLE *DDt2,\
					DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT numSamples) {

	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
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

	DOUBLE *Err = (DOUBLE *) MALLOC(M * numSamples * sizeof(DOUBLE));
	DOUBLE *Gamma = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *PhiDDt2 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT derivFlag = 1;
	
	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	
	eig_lsqr_obj_grad_smalldata(&f0, df0, X, Xdata, Y, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, numSamples, derivFlag,\
					Err, Gamma, ObjMat, PhiDDt2, DDt3temp);
/*
	eig_obj_grad(&f0, df0, X, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
*/
	
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
				
				
	
				eig_lsqr_obj_grad_smalldata(&f3, df3, X3, Xdata, Y, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, numSamples, derivFlag,\
					Err, Gamma, ObjMat, PhiDDt2, DDt3temp);
/*
				eig_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
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
			
			eig_lsqr_obj_grad_smalldata(&f3, df3, X3, Xdata, Y, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, numSamples, derivFlag,\
					Err, Gamma, ObjMat, PhiDDt2, DDt3temp);
/*
			eig_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
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
	
	FREE(Err);
	FREE(Gamma);
	FREE(ObjMat);
	FREE(PhiDDt2);
	FREE(DDt3temp);
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}

void minimize_eig_lap(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *XLXt, DOUBLE *DDt2, DOUBLE *DDt3, \
					DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT numSamples) {

	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
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

	DOUBLE *PhiXLXt = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *PhiXLXtPhit = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
	DOUBLE *Gamma = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *ObjMat = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	DOUBLE *PhiDDt2 = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
	DOUBLE *DDt3temp = (DOUBLE *) MALLOC(N * N * sizeof(DOUBLE));
	INT derivFlag = 1;
	
	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	
	eig_lap_obj_grad(&f0, df0, X, XLXt, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, numSamples, derivFlag,\
					PhiXLXt, PhiXLXtPhit, Gamma, ObjMat, PhiDDt2, DDt3temp);
/*
	eig_obj_grad(&f0, df0, X, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
*/
	
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
		PRINTF("Iter: %d\n\n", i);
		
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
				
				eig_lap_obj_grad(&f3, df3, X3, XLXt, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, numSamples, derivFlag,\
					PhiXLXt, PhiXLXtPhit, Gamma, ObjMat, PhiDDt2, DDt3temp);
/*
				eig_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
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
			
			eig_lap_obj_grad(&f3, df3, X3, XLXt, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, numSamples, derivFlag,\
					PhiXLXt, PhiXLXtPhit, Gamma, ObjMat, PhiDDt2, DDt3temp);
/*
			eig_obj_grad(&f3, df3, X3, DDt2, DDt3, VL, L, M, N, derivFlag, Gamma, ObjMat, PhiDDt2, DDt3temp);
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
	
	FREE(PhiXLXt);
	FREE(PhiXLXtPhit);
	FREE(Gamma);
	FREE(ObjMat);
	FREE(PhiDDt2);
	FREE(DDt3temp);
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}

void orth_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, INT M, INT N, INT derivFlag, DOUBLE *PhiPhit) {
	
	int PhiPhitFlag = 0;
	if (PhiPhit == NULL) {
		PhiPhit = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
		PhiPhitFlag = 1;
	}
	
	memset(PhiPhit, 0, M * M * sizeof(DOUBLE));
	INT iterM;
	
	for (iterM = 0; iterM < M; ++iterM) {
		PhiPhit[M * iterM + iterM] = 1;
	}
	
	CHAR uplo = 'U';
	CHAR trans = 'N';
	INT SYRKN = M;
	INT SYRKK = N;
	DOUBLE alpha = 1;
	INT SYRKLDA = M;
	DOUBLE beta = -1;
	INT SYRKLDC = M;
	
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, X, &SYRKLDA, &beta, PhiPhit, &SYRKLDC);
	
	CHAR norm = 'F';
	uplo = 'U';
	INT LANSYN = M;
	INT LANSYLDA = M;
	
	*obj = LANSY(&norm, &uplo, &LANSYN, PhiPhit, &LANSYLDA, NULL);
	*obj = SQR(*obj);
	
	if (derivFlag == 1) {

		CHAR side = 'L';
		uplo = 'U';
		INT SYMMM = M;
		INT SYMMN = N;
		alpha = 4;
		INT SYMMLDA = M;
		INT SYMMLDB = M;
		beta = 0;
		INT SYMMLDC = M;

		SYMM(&side, &uplo, &SYMMM, &SYMMN, &alpha, PhiPhit, &SYMMLDA, X, &SYMMLDB, &beta, deriv, &SYMMLDC);
	}
	
	if (PhiPhitFlag == 1) {
		FREE(PhiPhit);
	}
}

void minimize_orth(DOUBLE *Xopt, DOUBLE *Xorig, INT length, INT M, INT N) {

	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
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
	INT derivFlag = 1;

	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	
	DOUBLE *PhiPhit = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
	orth_obj_grad(&f0, df0, X, M, N, derivFlag, PhiPhit);
	
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
				
				orth_obj_grad(&f3, df3, X3, M, N, derivFlag, PhiPhit);

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

			orth_obj_grad(&f3, df3, X3, M, N, derivFlag, PhiPhit);
			
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
	
	FREE(PhiPhit);
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}
