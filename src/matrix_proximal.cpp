/*
 * matrix_proximal.c
 *
 *  Created on: Mar 25, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useblas.h"
#ifdef USE_CUDA
#include "usecuda.h"
#include "cudakernels.h"
#endif
#include "useinterfaces.h"
#include "matrix_proximal.h"
#include "utils.h"

// TODO: Change so that adaptive depending on M and N
// TODO: Check what standard I use for copy of original data, and get rid of it
// if not necessary.

#ifdef USE_CUDA
void nuclear_proximal_cuda(CUHANDLE handle, CUDOUBLE *X, CUDOUBLE *h_norm, \
							CUDOUBLE tau, CUINT M, CUINT N, CUDOUBLE *sv, \
							CUDOUBLE *svecsmall, CUDOUBLE *sveclarge) {

	CUINT MINMN = IMIN(M, N);
	CUINT MAXMN = IMAX(M, N);

	CUINT svFlag = 0;
	if (sv == NULL) {
		cumalloc(handle, (void**)&sv, MINMN * 1 * sizeof(CUDOUBLE));
		svFlag = 1;
	}

	CUINT svecsmallFlag = 0;
	if (svecsmall == NULL) {
		cumalloc(handle, (void**)&svecsmall, MINMN * MINMN * sizeof(CUDOUBLE));
		svecsmallFlag = 1;
	}

	CUINT sveclargeFlag = 0;
	if (sveclarge == NULL) {
		cumalloc(handle, (void**)&sveclarge, MAXMN * MINMN * sizeof(CUDOUBLE));
		sveclargeFlag = 1;
	}

	CUCHAR jobu = 'S';
	CUCHAR jobvt = 'S';
	CUDOUBLE *u;
	CUDOUBLE *vt;
	if (MAXMN == M) {
		u = sveclarge;
		vt = svecsmall;
	} else {
		u = svecsmall;
		vt = sveclarge;
	}
	CUINT GESVDM = M;
	CUINT GESVDN = N;
	CUINT GESVDLDA = M;
	CUINT GESVDLDU = M;
	CUINT GESVDLDVT = MINMN;

	CUGESVD(handle, jobu, jobvt, GESVDM, GESVDN, X, GESVDLDA, sv, u, GESVDLDU, \
			vt, GESVDLDVT);

	cuSoftThreshold(sv, tau, MINMN);
	if (h_norm != NULL) {
		CUINT ASUMN = MINMN;
		CUINT incx = 1;
		CUASUM(handle, ASUMN, sv, incx, h_norm);
	}

	/*
	 * TODO: Only multiply for singular vectors corresponding to non-zero singular values.
	 * TODO: Or, maybe deflate X instead of GEMM.
	 * TODO: Remove if clause? Doesn't appear to be necessary.
	 */
	if (MAXMN == M) {
		CUINT SCALN = M;
		CUINT incx = 1;
		INT iterMN;
		cublasSetPointerModeDevice(handle);
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			CUSCAL(handle, SCALN, &sv[iterMN], &u[iterMN * M], incx);
		}
		cublasSetPointerModeHost(handle);

		CUCHAR transa = 'N';
		CUCHAR transb = 'N';
		CUINT GEMMM = M;
		CUINT GEMMN = N;
		CUINT GEMMK = MINMN;
		CUDOUBLE alpha = 1;
		CUINT GEMMLDA = M;
		CUINT GEMMLDB = MINMN;
		CUDOUBLE beta = 0;
		CUINT GEMMLDC = M;

		CUGEMM(handle, transa, transb, GEMMM, GEMMN, GEMMK, &alpha, u, GEMMLDA, \
			vt, GEMMLDB, &beta, X, GEMMLDC);
	} else {
		CUINT SCALN = M;
		CUINT incx = 1;
		INT iterMN;
		cublasSetPointerModeDevice(handle);
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			CUSCAL(handle, SCALN, &sv[iterMN], &u[iterMN * M], incx);
		}
		cublasSetPointerModeHost(handle);

		CUCHAR transa = 'N';
		CUCHAR transb = 'N';
		CUINT GEMMM = M;
		CUINT GEMMN = N;
		CUINT GEMMK = MINMN;
		CUDOUBLE alpha = 1;
		CUINT GEMMLDA = M;
		CUINT GEMMLDB = MINMN;
		CUDOUBLE beta = 0;
		CUINT GEMMLDC = M;

		CUGEMM(handle, transa, transb, GEMMM, GEMMN, GEMMK, &alpha, u, GEMMLDA, \
			vt, GEMMLDB, &beta, X, GEMMLDC);
	}

	if (svFlag == 1) {
		cufree(handle, sv);
	}

	if (svecsmallFlag == 1) {
		cufree(handle, svecsmall);
	}

	if (sveclargeFlag == 1) {
		cufree(handle, sveclarge);
	}
}
#endif

void nuclear_proximal(DOUBLE *X, DOUBLE *norm, DOUBLE tau, INT M, INT N, DOUBLE *sv, \
		DOUBLE *svecsmall, DOUBLE *sveclarge, DOUBLE *work, INT lwork) {

	INT MINMN = IMIN(M, N);
	INT MAXMN = IMAX(M, N);

	INT svFlag = 0;
	if (sv == NULL) {
		sv = (DOUBLE *) CMALLOC(MINMN * 1 * sizeof(DOUBLE));
		svFlag = 1;
	}

	INT svecsmallFlag = 0;
	if (svecsmall == NULL) {
		svecsmall = (DOUBLE *) CMALLOC(MINMN * MINMN * sizeof(DOUBLE));
		svecsmallFlag = 1;
	}

	INT sveclargeFlag = 0;
	if (sveclarge == NULL) {
		sveclarge = (DOUBLE *) CMALLOC(MAXMN * MINMN * sizeof(DOUBLE));
		sveclargeFlag = 1;
	}

	CHAR jobu = 'S';
	CHAR jobvt = 'S';
	DOUBLE *u;
	DOUBLE *vt;
	if (MAXMN == M) {
		u = sveclarge;
		vt = svecsmall;
	} else {
		u = svecsmall;
		vt = sveclarge;
	}
	INT GESVDM = M;
	INT GESVDN = N;
	INT GESVDLDA = M;
	INT GESVDLDU = M;
	INT GESVDLDVT = MINMN;
	INT info;

	if (lwork == -1) {
		GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, X, &GESVDLDA, sv, u, &GESVDLDU, vt, &GESVDLDVT, work, &lwork, &info);

		if (svFlag == 1) {
			CFREE(sv);
		}

		if (svecsmallFlag == 1) {
			CFREE(svecsmall);
		}

		if (sveclargeFlag == 1) {
			CFREE(sveclarge);
		}
		return;
	}

	INT workFlag = 0;
	if (lwork == 0) {
		DOUBLE workTemp;
		lwork = -1;
		GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, X, &GESVDLDA, sv, u, &GESVDLDU, vt, &GESVDLDVT, &workTemp, &lwork, &info);

		lwork = (INT) workTemp;
		work = (DOUBLE *) CMALLOC(lwork * 1 * sizeof(DOUBLE));
		workFlag = 1;
	}

	GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, X, &GESVDLDA, sv, u, &GESVDLDU, vt, &GESVDLDVT, work, &lwork, &info);

	INT iterMN;
	DOUBLE normtemp = 0;
	for (iterMN = 0; iterMN < MINMN; ++iterMN) {
		sv[iterMN] = sv[iterMN] - tau;
		(sv[iterMN] < 0) ? (sv[iterMN] = 0) : (normtemp += sv[iterMN]);
	}

	if (norm != NULL) {
		*norm = normtemp;
	}

	/*
	 * TODO: Only multiply for singular vectors corresponding to non-zero singular values.
	 * TODO: Or, maybe deflate X instead of GEMM.
	 * TODO: Remove if clause? Doesn't appear to be necessary.
	 */
	if (MAXMN == M) {
		INT SCALN = M;
		INT incx = 1;
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			SCAL(&SCALN, &sv[iterMN], &u[iterMN * M], &incx);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = M;
		INT GEMMN = N;
		INT GEMMK = MINMN;
		DOUBLE alpha = 1;
		INT GEMMLDA = M;
		INT GEMMLDB = MINMN;
		DOUBLE beta = 0;
		INT GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, u, &GEMMLDA, vt, &GEMMLDB, &beta, X, &GEMMLDC);
	} else {
		INT SCALN = M;
		INT incx = 1;
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			SCAL(&SCALN, &sv[iterMN], &u[iterMN * M], &incx);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = M;
		INT GEMMN = N;
		INT GEMMK = MINMN;
		DOUBLE alpha = 1;
		INT GEMMLDA = M;
		INT GEMMLDB = MINMN;
		DOUBLE beta = 0;
		INT GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, u, &GEMMLDA, vt, &GEMMLDB, &beta, X, &GEMMLDC);
	}

	if (svFlag == 1) {
		CFREE(sv);
	}

	if (svecsmallFlag == 1) {
		CFREE(svecsmall);
	}

	if (sveclargeFlag == 1) {
		CFREE(sveclarge);
	}

	if (workFlag == 1) {
		CFREE(work);
	}
}

void nuclear_proximal_gesdd(DOUBLE *X, DOUBLE *norm, DOUBLE tau, INT M, INT N, \
		DOUBLE *sv, DOUBLE *svecsmall, DOUBLE *sveclarge, DOUBLE *work, \
		INT lwork, INT *iwork) {

	INT MINMN = IMIN(M, N);
	INT MAXMN = IMAX(M, N);

	INT svFlag = 0;
	if (sv == NULL) {
		sv = (DOUBLE *) CMALLOC(MINMN * 1 * sizeof(DOUBLE));
		svFlag = 1;
	}

	INT svecsmallFlag = 0;
	if (svecsmall == NULL) {
		svecsmall = (DOUBLE *) CMALLOC(MINMN * MINMN * sizeof(DOUBLE));
		svecsmallFlag = 1;
	}

	INT sveclargeFlag = 0;
	if (sveclarge == NULL) {
		sveclarge = (DOUBLE *) CMALLOC(MAXMN * MINMN * sizeof(DOUBLE));
		sveclargeFlag = 1;
	}

	INT iworkFlag = 0;
	if (iwork == NULL) {
		iwork = (INT *) CMALLOC(16 * MINMN * sizeof(INT));
		iworkFlag = 1;
	}

	CHAR jobz = 'S';
	DOUBLE *u;
	DOUBLE *vt;
	if (MAXMN == M) {
		u = sveclarge;
		vt = svecsmall;
	} else {
		u = svecsmall;
		vt = sveclarge;
	}
	INT GESDDM = M;
	INT GESDDN = N;
	INT GESDDLDA = M;
	INT GESDDLDU = M;
	INT GESDDLDVT = MINMN;
	INT info;

	if (lwork == -1) {
		GESDD(&jobz, &GESDDM, &GESDDN, X, &GESDDLDA, sv, u, &GESDDLDU, vt, &GESDDLDVT, work, &lwork, iwork, &info);

		if (svFlag == 1) {
			CFREE(sv);
		}

		if (svecsmallFlag == 1) {
			CFREE(svecsmall);
		}

		if (sveclargeFlag == 1) {
			CFREE(sveclarge);
		}

		if (iworkFlag == 1) {
			CFREE(iwork);
		}
		return;
	}

	INT workFlag = 0;
	if (lwork == 0) {
		DOUBLE workTemp;
		lwork = -1;
		GESDD(&jobz, &GESDDM, &GESDDN, X, &GESDDLDA, sv, u, &GESDDLDU, vt, &GESDDLDVT, &workTemp, &lwork, iwork, &info);

		lwork = (INT) workTemp;
		work = (DOUBLE *) CMALLOC(lwork * 1 * sizeof(DOUBLE));
		workFlag = 1;
	}

	GESDD(&jobz, &GESDDM, &GESDDN, X, &GESDDLDA, sv, u, &GESDDLDU, vt, &GESDDLDVT, work, &lwork, iwork, &info);

	INT iterMN;
	DOUBLE normtemp = 0;
	for (iterMN = 0; iterMN < MINMN; ++iterMN) {
		sv[iterMN] = sv[iterMN] - tau;
		(sv[iterMN] < 0) ? (sv[iterMN] = 0) : (normtemp += sv[iterMN]);
	}

	if (norm != NULL) {
		*norm = normtemp;
	}

	/*
	 * TODO: Only multiply for singular vectors corresponding to non-zero singular values.
	 * TODO: Or, maybe deflate X instead of GEMM.
	 * TODO: Remove if clause? Doesn't appear to be necessary.
	 */
	if (MAXMN == M) {
		INT SCALN = M;
		INT incx = 1;
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			SCAL(&SCALN, &sv[iterMN], &u[iterMN * M], &incx);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = M;
		INT GEMMN = N;
		INT GEMMK = MINMN;
		DOUBLE alpha = 1;
		INT GEMMLDA = M;
		INT GEMMLDB = MINMN;
		DOUBLE beta = 0;
		INT GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, u, &GEMMLDA, vt, &GEMMLDB, &beta, X, &GEMMLDC);
	} else {
		INT SCALN = M;
		INT incx = 1;
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			SCAL(&SCALN, &sv[iterMN], &u[iterMN * M], &incx);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = M;
		INT GEMMN = N;
		INT GEMMK = MINMN;
		DOUBLE alpha = 1;
		INT GEMMLDA = M;
		INT GEMMLDB = MINMN;
		DOUBLE beta = 0;
		INT GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, u, &GEMMLDA, vt, &GEMMLDB, &beta, X, &GEMMLDC);
	}

	if (svFlag == 1) {
		CFREE(sv);
	}

	if (svecsmallFlag == 1) {
		CFREE(svecsmall);
	}

	if (sveclargeFlag == 1) {
		CFREE(sveclarge);
	}

	if (iworkFlag == 1) {
		CFREE(iwork);
	}

	if (workFlag == 1) {
		CFREE(work);
	}
}

/* 	NOTE: alternative 1, slower due to non-unitary stride for vt.
	INT iterMN;
	DOUBLE normtemp = 0;
	memset((void *) X, 0, M * N * sizeof(DOUBLE));
	INT GERM = M;
	INT GERN = N;
	DOUBLE alpha;
	INT incx = 1;
	INT incy = MINMN;
	INT GERLDA = M;

	for (iterMN = 0; iterMN < MINMN; ++iterMN) {
		sv[iterMN] = sv[iterMN] - tau;
		if (sv[iterMN] < 0) {
			sv[iterMN] = 0;
		} else {
			normtemp += sv[iterMN];
			alpha = sv[iterMN];
			GER(&GERM, &GERN, &alpha, &u[iterMN * M], &incx, &vt[iterMN], &incy, X, &GERLDA);
		}
	}
	*norm = normtemp;
 */

/*	NOTE: alternative 2, incorrect, due to need to use rows of vt instead of columns. Perhaps with two transpositions?
	INT iterMN;
	DOUBLE normtemp = 0;
	INT rank = -1;
	for (iterMN = 0; iterMN < MINMN; ++iterMN) {
		sv[iterMN] = sv[iterMN] - tau;
		if (sv[iterMN] < 0) {
			sv[iterMN] = 0;
			if (rank < 0) {
				rank = iterMN;
			}
		} else {
			normtemp += sv[iterMN];
		}
//		(sv[iterMN] < 0) ? (sv[iterMN] = 0) : (normtemp += sv[iterMN]);
	}
	*norm = normtemp;
	if (rank < 0) {
		rank = MINMN;
	}

	if (rank == 0) {
		memset((void *) X, 0, M * N * sizeof(DOUBLE));
	} else if (MAXMN == M) {
		INT SCALN = M;
		INT incx = 1;
		for (iterMN = 0; iterMN < rank; ++iterMN) {
			SCAL(&SCALN, &sv[iterMN], &u[iterMN * M], &incx);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = M;
		INT GEMMN = N;
		INT GEMMK = rank;
		DOUBLE alpha = 1;
		INT GEMMLDA = M;
		INT GEMMLDB = rank;
		DOUBLE beta = 0;
		INT GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, u, &GEMMLDA, vt, &GEMMLDB, &beta, X, &GEMMLDC);
	} else {
		INT SCALN = M;
		INT incx = 1;
		for (iterMN = 0; iterMN < rank; ++iterMN) {
			SCAL(&SCALN, &sv[iterMN], &u[iterMN * M], &incx);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = M;
		INT GEMMN = N;
		INT GEMMK = rank;
		DOUBLE alpha = 1;
		INT GEMMLDA = M;
		INT GEMMLDB = rank;
		DOUBLE beta = 0;
		INT GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, u, &GEMMLDA, vt, &GEMMLDB, &beta, X, &GEMMLDC);
	}
 */

void nuclear_hard_thresholding(DOUBLE *X, DOUBLE *norm, INT rank, INT M, INT N, DOUBLE *sv, \
		DOUBLE *svecsmall, DOUBLE *sveclarge, DOUBLE *work, INT lwork) {

	INT MINMN = IMIN(M, N);
	INT MAXMN = IMAX(M, N);

	INT svFlag = 0;
	if (sv == NULL) {
		sv = (DOUBLE *) CMALLOC(MINMN * 1 * sizeof(DOUBLE));
		svFlag = 1;
	}

	INT svecsmallFlag = 0;
	if (svecsmall == NULL) {
		svecsmall = (DOUBLE *) CMALLOC(MINMN * MINMN * sizeof(DOUBLE));
		svecsmallFlag = 1;
	}

	INT sveclargeFlag = 0;
	if (sveclarge == NULL) {
		sveclarge = (DOUBLE *) CMALLOC(MAXMN * MINMN * sizeof(DOUBLE));
		sveclargeFlag = 1;
	}

	CHAR jobu = 'S';
	CHAR jobvt = 'S';
	DOUBLE *u;
	DOUBLE *vt;
	if (MAXMN == M) {
		u = sveclarge;
		vt = svecsmall;
	} else {
		u = svecsmall;
		vt = sveclarge;
	}
	INT GESVDM = M;
	INT GESVDN = N;
	INT GESVDLDA = M;
	INT GESVDLDU = M;
	INT GESVDLDVT = MINMN;
	INT info;

	if (lwork == -1) {
		GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, X, &GESVDLDA, sv, u, &GESVDLDU, vt, &GESVDLDVT, work, &lwork, &info);

		if (svFlag == 1) {
			CFREE(sv);
		}

		if (svecsmallFlag == 1) {
			CFREE(svecsmall);
		}

		if (sveclargeFlag == 1) {
			CFREE(sveclarge);
		}
		return;
	}

	INT workFlag = 0;
	if (lwork == 0) {
		DOUBLE workTemp;
		lwork = -1;
		GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, X, &GESVDLDA, sv, u, &GESVDLDU, vt, &GESVDLDVT, &workTemp, &lwork, &info);

		lwork = (INT) workTemp;
		work = (DOUBLE *) CMALLOC(lwork * 1 * sizeof(DOUBLE));
		workFlag = 1;
	}

	GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, X, &GESVDLDA, sv, u, &GESVDLDU, vt, &GESVDLDVT, work, &lwork, &info);

	INT iterMN;
	DOUBLE normtemp = 0;
	for (iterMN = 0; iterMN < rank; ++iterMN) {
		normtemp += sv[iterMN];
	}

	if (norm != NULL) {
		*norm = normtemp;
	}

	for (iterMN = rank; iterMN < MINMN; ++iterMN) {
		sv[iterMN] = 0;
	}

	/*
	 * TODO: Only multiply for singular vectors corresponding to non-zero singular values.
	 */
	if (MAXMN == M) {
		INT SCALN = M;
		INT incx = 1;
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			SCAL(&SCALN, &sv[iterMN], &u[iterMN * M], &incx);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = M;
		INT GEMMN = N;
		INT GEMMK = MINMN;
		DOUBLE alpha = 1;
		INT GEMMLDA = M;
		INT GEMMLDB = MINMN;
		DOUBLE beta = 0;
		INT GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, u, &GEMMLDA, vt, &GEMMLDB, &beta, X, &GEMMLDC);
	} else {
		INT SCALN = M;
		INT incx = 1;
		for (iterMN = 0; iterMN < MINMN; ++iterMN) {
			SCAL(&SCALN, &sv[iterMN], &u[iterMN * M], &incx);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = M;
		INT GEMMN = N;
		INT GEMMK = MINMN;
		DOUBLE alpha = 1;
		INT GEMMLDA = M;
		INT GEMMLDB = MINMN;
		DOUBLE beta = 0;
		INT GEMMLDC = M;

		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, u, &GEMMLDA, vt, &GEMMLDB, &beta, X, &GEMMLDC);
	}

	if (svFlag == 1) {
		CFREE(sv);
	}

	if (svecsmallFlag == 1) {
		CFREE(svecsmall);
	}

	if (sveclargeFlag == 1) {
		CFREE(sveclarge);
	}

	if (workFlag == 1) {
		CFREE(work);
	}
}

void nuclear_psd_proximal(DOUBLE *X, DOUBLE *norm, DOUBLE tau, INT M, DOUBLE *eigv, \
		DOUBLE *eigvec, DOUBLE *work, INT lwork) {

	CHAR jobz = 'V';
	CHAR uplo = 'U';
	INT SYEVN = M;
	INT SYEVLDA = M;
	INT info;

	if (lwork == - 1) {
		SYEV(&jobz, &uplo, &SYEVN, eigvec, &SYEVLDA, eigv, work, &lwork, &info);
		return;
	}

	INT eigvFlag = 0;
	if (eigv == NULL) {
		eigv = (DOUBLE *) MALLOC(M * 1 * sizeof(DOUBLE));
		eigvFlag = 1;
	}

	INT eigvecFlag = 0;
	if (eigvec == NULL) {
		eigvec = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
		eigvecFlag = 1;
	}

	datacpy(eigvec, X, M * M);
	INT workFlag = 0;
	if (lwork == 0) {
		DOUBLE workTemp;
		lwork = -1;
		SYEV(&jobz, &uplo, &SYEVN, eigvec, &SYEVLDA, eigv, &workTemp, &lwork, &info);

		lwork = (INT) workTemp;
		work = (DOUBLE *) MALLOC(lwork * 1 * sizeof(DOUBLE));
		workFlag = 1;
	}

	// TODO: Perhaps replace with SYEVR?
	SYEV(&jobz, &uplo, &SYEVN, eigvec, &SYEVLDA, eigv, work, &lwork, &info);

	INT iterM;
	DOUBLE normtemp = 0;
	DOUBLE alpha;
	INT SCALN = M;
	INT incx = 1;
	INT rank = -1;
	for (iterM = 0; iterM < M; ++iterM) {
		eigv[iterM] = eigv[iterM] - tau;
		if (eigv[iterM] < 0) {
			eigv[iterM] = 0;
		} else {
			if (rank < 0) {
				rank = iterM;
			}
			normtemp += eigv[iterM];
			alpha = SQRT(eigv[iterM]);
			SCAL(&SCALN, &alpha, &eigvec[iterM * M], &incx);
		}
	}
	if (norm != NULL) {
		*norm = normtemp;
	}

	if (rank < 0) {
		rank = M;
	}
//	printf("rank %d\n", M-rank);
	uplo = 'U';
	CHAR trans = 'N';
	INT SYRKN = M;
	INT SYRKK = M - rank;
	alpha = 1;
	INT SYRKLDA = M;
	DOUBLE beta = 0;
	INT SYRKLDC = M;
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, &eigvec[rank * M], &SYRKLDA, &beta, X, &SYRKLDC);

/* 	NOTE: alternative 1, somewhat slower than version above.
	INT iterM;
	DOUBLE normtemp = 0;
	memset((void *) X, 0, M * M * sizeof(DOUBLE));
	uplo = 'U';
	INT SYRN = M;
	DOUBLE alpha;
	INT SYRLDA = M;
	INT incx = 1;
	for (iterM = 0; iterM < M; ++iterM) {
		eigv[iterM] = eigv[iterM] - tau;
		if (eigv[iterM] < 0) {
			eigv[iterM] = 0;
		} else {
			normtemp += eigv[iterM];
			alpha = eigv[iterM];
			SYR(&uplo, &SYRN, &alpha, &eigvec[iterM * M], &incx, X, &SYRLDA);
		}
	}
	*norm = normtemp;
 */

	INT iterN;
	for (iterM = 0; iterM < M; ++iterM) {
		for (iterN = iterM + 1; iterN < M; ++iterN) {
			X[iterM * M + iterN] = X[iterN * M + iterM];
		}
	}

	if (eigvFlag == 1) {
		FREE(eigv);
	}

	if (eigvecFlag == 1) {
		FREE(eigvec);
	}

	if (workFlag == 1) {
		FREE(work);
	}
}

void matrix_psd_projection(DOUBLE *X, INT M, DOUBLE *eigv, DOUBLE *eigvec, \
							DOUBLE *work, INT lwork) {
	nuclear_psd_proximal(X, NULL, 0, M, eigv, eigvec, work, lwork);
}

void nuclear_psd_hard_thresholding(DOUBLE *X, DOUBLE *norm, INT rank, INT M, DOUBLE *eigv, \
		DOUBLE *eigvec, DOUBLE *work, INT lwork) {

	CHAR jobz = 'V';
	CHAR uplo = 'U';
	INT SYEVN = M;
	INT SYEVLDA = M;
	INT info;

	if (lwork == - 1) {
		SYEV(&jobz, &uplo, &SYEVN, eigvec, &SYEVLDA, eigv, work, &lwork, &info);
		return;
	}

	INT eigvFlag = 0;
	if (eigv == NULL) {
		eigv = (DOUBLE *) MALLOC(M * 1 * sizeof(DOUBLE));
		eigvFlag = 1;
	}

	INT eigvecFlag = 0;
	if (eigvec == NULL) {
		eigvec = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
		eigvecFlag = 1;
	}

	datacpy(eigvec, X, M * M);
	INT workFlag = 0;
	if (lwork == 0) {
		DOUBLE workTemp;
		lwork = -1;
		SYEV(&jobz, &uplo, &SYEVN, eigvec, &SYEVLDA, eigv, &workTemp, &lwork, &info);

		lwork = (INT) workTemp;
		work = (DOUBLE *) MALLOC(lwork * 1 * sizeof(DOUBLE));
		workFlag = 1;
	}

	// TODO: Perhaps replace with SYEVR?
	SYEV(&jobz, &uplo, &SYEVN, eigvec, &SYEVLDA, eigv, work, &lwork, &info);

	INT iterM;
	DOUBLE normtemp = 0;
	DOUBLE alpha;
	INT SCALN = M;
	INT incx = 1;
	for (iterM = 0; iterM < M; ++iterM) {
		if ((eigv[iterM] < 0) || (iterM < M - rank)){
			eigv[iterM] = 0;
		} else {
			normtemp += eigv[iterM];
			alpha = SQRT(eigv[iterM]);
			SCAL(&SCALN, &alpha, &eigvec[iterM * M], &incx);
		}
	}
	if (norm != NULL) {
		*norm = normtemp;
	}

	uplo = 'U';
	CHAR trans = 'N';
	INT SYRKN = M;
	INT SYRKK = rank;
	alpha = 1;
	INT SYRKLDA = M;
	DOUBLE beta = 0;
	INT SYRKLDC = M;
	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, &eigvec[(M - rank) * M], &SYRKLDA, &beta, X, &SYRKLDC);

/* 	NOTE: alternative 1, somewhat slower than version above.
	INT iterM;
	DOUBLE normtemp = 0;
	memset((void *) X, 0, M * M * sizeof(DOUBLE));
	uplo = 'U';
	INT SYRN = M;
	DOUBLE alpha;
	INT SYRLDA = M;
	INT incx = 1;
	for (iterM = 0; iterM < M; ++iterM) {
		eigv[iterM] = eigv[iterM] - tau;
		if (eigv[iterM] < 0) {
			eigv[iterM] = 0;
		} else {
			normtemp += eigv[iterM];
			alpha = eigv[iterM];
			SYR(&uplo, &SYRN, &alpha, &eigvec[iterM * M], &incx, X, &SYRLDA);
		}
	}
	*norm = normtemp;
 */

	INT iterN;
	for (iterM = 0; iterM < M; ++iterM) {
		for (iterN = iterM + 1; iterN < M; ++iterN) {
			X[iterM * M + iterN] = X[iterN * M + iterM];
		}
	}

	if (eigvFlag == 1) {
		FREE(eigv);
	}

	if (eigvecFlag == 1) {
		FREE(eigvec);
	}

	if (workFlag == 1) {
		FREE(work);
	}
}

void nuclear_approx_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *rp, \
					INT M, INT N, INT derivFlag, DOUBLE *svdVec, DOUBLE *vtMat, \
					DOUBLE *dataBuffer, DOUBLE *derivVec, DOUBLE *work, INT lwork) {

	INT MN = IMIN(M, N);

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
		INT GESVDM = M;
		INT GESVDN = N;
		INT GESVDLDA = M;
		INT GESVDLDU = M;
		INT GESVDLDVT = MN;
		INT INFO;

		GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, dataBuffer, &GESVDLDA, svdVec, NULL, &GESVDLDU, vtMat, &GESVDLDVT, work, &lwork, &INFO);
		return;
	}

	INT svdVecFlag = 0;
	if (svdVec == NULL) {
		svdVec = (DOUBLE *) MALLOC(1 * MN * sizeof(DOUBLE));
		svdVecFlag = 1;
	}

	INT derivVecFlag = 0;
	if ((derivVec == NULL) && (derivFlag == 1)) {
		derivVec = (DOUBLE *) MALLOC(1 * MN * sizeof(DOUBLE));
		derivVecFlag = 1;
	}

	INT vtMatFlag = 0;
	if ((vtMat == NULL) && (derivFlag == 1)) {
		vtMat = (DOUBLE *) MALLOC(MN * N * sizeof(DOUBLE));
		vtMatFlag = 1;
	}

	INT dataBufferFlag = 0;
	if (dataBuffer == NULL) {
		dataBuffer = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
		dataBufferFlag = 1;
	}

	INT workFlag = 0;
	if (work == NULL) {
		workFlag = 1;
	}

	CHAR jobu;
	CHAR jobvt;
	if (derivFlag == 1) {
		jobu = 'O';
		jobvt = 'S';
	} else {
		jobu = 'N';
		jobvt = 'N';
	}
	datacpy(dataBuffer, X, M * N);

	INT GESVDM = M;
	INT GESVDN = N;
	INT GESVDLDA = M;
	INT GESVDLDU = M;
	INT GESVDLDVT = MN;
	INT INFO;

	if (workFlag == 1) {
		lwork = -1;
		DOUBLE work_temp;
		GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, dataBuffer, &GESVDLDA, svdVec, NULL, &GESVDLDU, vtMat, &GESVDLDVT, &work_temp, &lwork, &INFO);

		lwork = (INT) work_temp;
		work = (DOUBLE*) MALLOC(lwork * sizeof(DOUBLE));
	}

	GESVD(&jobu, &jobvt, &GESVDM, &GESVDN, dataBuffer, &GESVDLDA, svdVec, NULL, &GESVDLDU, vtMat, &GESVDLDVT, work, &lwork, &INFO);

	abs_smooth_obj_grad(svdVec, derivVec, svdVec, rp, MN, derivFlag);
	INT ASUMN = MN;
	INT incx = 1;
	*obj = ASUM(&ASUMN, svdVec, &incx);

	if (derivFlag == 1) {

		INT iterMN;
		INT SCALN = M;
		INT incx = 1;
		DOUBLE alpha;
		for (iterMN = 0; iterMN < MN; ++iterMN) {
			alpha = derivVec[iterMN];
			SCAL(&SCALN, &alpha, &dataBuffer[iterMN * M], &incx);
		}

		CHAR transa = 'N';
		CHAR transb = 'N';
		INT GEMMM = M;
		INT GEMMN = N;
		INT GEMMK = MN;
		alpha = 1.0;
		INT GEMMLDA = M;
		INT GEMMLDB = MN;
		DOUBLE beta = 0.0;
		INT GEMMLDC = M;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, dataBuffer, &GEMMLDA, vtMat, &GEMMLDB, &beta, deriv, &GEMMLDC);
	}

	if (svdVecFlag == 1) {
		FREE(svdVec);
	}

	if (derivVecFlag == 1) {
		FREE(derivVec);
	}

	if (vtMatFlag == 1) {
		FREE(vtMat);
	}

	if (dataBufferFlag == 1) {
		FREE(dataBuffer);
	}

	if (workFlag == 1) {
		FREE(work);
	}
}

void abs_smooth_obj_grad(DOUBLE *objVec, DOUBLE *derivVec, DOUBLE *xVec, DOUBLE *rp, INT N, INT derivFlag) {

	DOUBLE r = *rp;
	DOUBLE rhalf = r / 2.0;
	DOUBLE invrtwo = 1 / (2.0 * r);
	DOUBLE invr = 1 / r;
	DOUBLE x;
	INT iterN;

	if (derivFlag == 0) {
		#pragma omp parallel for private(iterN, x) shared(objVec, xVec) firstprivate(N, r, invrtwo, rhalf)
		for (iterN = 0; iterN < N; ++iterN) {
			x = xVec[iterN];
			if (x < -r) {
				objVec[iterN] = - x;
			} else if (x > r) {
				objVec[iterN] = x;
			} else {
				objVec[iterN] = SQR(x) * invrtwo + rhalf;
			}
		}
	} else {
		#pragma omp parallel for private(iterN, x) shared(objVec, xVec) firstprivate(N, r, invrtwo, invr, rhalf)
		for (iterN = 0; iterN < N; ++iterN) {
			x = xVec[iterN];
			if (x < -r) {
				objVec[iterN] = - x;
				derivVec[iterN] = - 1;
			} else if (x > r) {
				objVec[iterN] = x;
				derivVec[iterN] = 1;
			} else {
				objVec[iterN] = SQR(x) * invrtwo + rhalf;
				derivVec[iterN] = x * invr;
			}
		}
	}

}

//void nuclear_psd_hard_thresholding(DOUBLE *X, DOUBLE *norm, INT rank, INT M, DOUBLE *eigv, \
//		DOUBLE *eigvec, DOUBLE *work, INT lwork, INT *iwork, INT liwork) {
//
//	CHAR jobz = 'V';
//	CHAR range = 'I';
//	CHAR uplo = 'U';
//	INT SYEVRN = M;
//	INT SYEVLDA = M;
//	DOUBLE VL, VU;
//	INT IL = M - rank + 1;
//	INT IU = M;
//	CHAR lamch_opt = 'S';
//	DOUBLE sfmin = LAMCH(&lamch_opt);
//	DOUBLE abstol = sfmin;
//	INT SYEVLDZ = M;
//	INT *ISUPPZ = NULL;
//	INT SYEVRM;
//	INT info;
//	const INT SYEVRM_expected = IU - IL + 1;
//
//	if (lwork == - 1) {
//		SYEVR(&jobz, &range, &uplo, &SYEVRN, X, &SYEVLDA, &VL, &VU, &IL, &IU, &abstol, &SYEVRM, \
//				eigv, eigvec, &SYEVLDZ, ISUPPZ, work, &lwork, iwork, &liwork, &info);
//		return;
//	}
//
//	INT eigvFlag = 0;
//	if (eigv == NULL) {
//		eigv = (DOUBLE *) MALLOC(M * 1 * sizeof(DOUBLE));
//		eigvFlag = 1;
//	}
//
//	INT eigvecFlag = 0;
//	if (eigvec == NULL) {
//		eigvec = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
//		eigvecFlag = 1;
//	}
//
//	datacpy(eigvec, X, M * M);
//	INT workFlag = 0;
//	INT iworkFlag = 0;
//	if (lwork == 0) {
//		DOUBLE workTemp;
//		INT iworkTemp;
//		lwork = -1;
//		liwork = -1;
//		SYEVR(&jobz, &range, &uplo, &SYEVRN, X, &SYEVLDA, &VL, &VU, &IL, &IU, &abstol, &SYEVRM, \
//				eigv, eigvec, &SYEVLDZ, ISUPPZ, &workTemp, &lwork, &iworkTemp, &liwork, &info);
//
//		lwork = (INT) workTemp;
//		work = (DOUBLE *) MALLOC(lwork * 1 * sizeof(DOUBLE));
//		workFlag = 1;
//
//		liwork = (INT) iworkTemp;
//		iwork = (INT *) MALLOC(liwork * 1 * sizeof(INT));
//		iworkFlag = 1;
//	}
//
//	SYEVR(&jobz, &range, &uplo, &SYEVRN, X, &SYEVLDA, &VL, &VU, &IL, &IU, &abstol, &SYEVRM, \
//			eigv, eigvec, &SYEVLDZ, ISUPPZ, work, &lwork, iwork, &liwork, &info);
//	if (SYEVRM != SYEVRM_expected) {
//		PRINTF("Error, only %d eigenvalues were found, when %d were expected. ", SYEVRM, SYEVRM_expected);
//		ERROR("LAPACK execution error.");
//	}
//
//	INT iterM;
//	DOUBLE normtemp = 0;
//	DOUBLE alpha;
//	INT SCALN = M;
//	INT incx = 1;
//	for (iterM = 0; iterM < rank; ++iterM) {
//		eigv[iterM] = (eigv[iterM > 0]) ? (eigv[iterM]) : (0);
//		normtemp += eigv[iterM];
//		alpha = SQRT(eigv[iterM]);
//		SCAL(&SCALN, &alpha, &eigvec[iterM * M], &incx);
//	}
//
//	if (norm != NULL) {
//		*norm = normtemp;
//	}
//
//	uplo = 'U';
//	CHAR trans = 'N';
//	INT SYRKN = M;
//	INT SYRKK = rank;
//	alpha = 1;
//	INT SYRKLDA = M;
//	DOUBLE beta = 0;
//	INT SYRKLDC = M;
//	SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, eigvec, &SYRKLDA, &beta, X, &SYRKLDC);
//
//	INT iterN;
//	for (iterM = 0; iterM < M; ++iterM) {
//		for (iterN = iterM + 1; iterN < M; ++iterN) {
//			X[iterM * M + iterN] = X[iterN * M + iterM];
//		}
//	}
//
//	if (eigvFlag == 1) {
//		FREE(eigv);
//	}
//
//	if (eigvecFlag == 1) {
//		FREE(eigvec);
//	}
//
//	if (workFlag == 1) {
//		FREE(work);
//	}
//
//	if (iworkFlag == 1) {
//		FREE(iwork);
//	}
//}
