#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useinterfaces.h"
#include "useblas.h"
#include "l1_featuresign.h"
#include "exponential.h"

void l1_featuresign_sub(DOUBLE *x, DOUBLE *A, DOUBLE *b, DOUBLE lambda, INT K, \
						DOUBLE *grad, INT *actset, DOUBLE *xa, DOUBLE *ba, \
						DOUBLE *Aa, DOUBLE *signa, DOUBLE *vect, DOUBLE *xnew, INT *idxset, \
						INT *sset, DOUBLE *vecti, DOUBLE *bi, DOUBLE *xnewi, DOUBLE *Ai, \
						DOUBLE *xmin, DOUBLE *d, DOUBLE *t, DOUBLE *xs, INT init) {
	
	INT gradflag = 0;
	if (grad == NULL) {
		grad = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		gradflag = 1;
	}
	
	INT actsetflag = 0;
	if (actset == NULL) {
		actset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		actsetflag = 1;
	}
	
	INT xaflag = 0;
	if (xa == NULL) {
		xa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xaflag = 1;
	}
	
	INT baflag = 0;
	if (ba == NULL) {
		ba = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		baflag = 1;
	}
	
	INT Aaflag = 0;
	if (Aa == NULL) {
		Aa = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		Aaflag = 1;
	}
	
	INT signaflag = 0;
	if (signa == NULL) {
		signa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		signaflag = 1;
	}
	
	INT vectflag = 0;
	if (vect == NULL) {
		vect = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		vectflag = 1;
	}
	
	INT xnewflag = 0;
	if (xnew == NULL) {
		xnew = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnewflag = 1;
	}
	
	INT idxsetflag = 0;
	if (idxset == NULL) {
		idxset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		idxsetflag = 1;
	}
	
	INT ssetflag = 0;
	if (sset == NULL) {
		sset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		ssetflag = 1;
	}
	
	INT vectiflag = 0;
	if (vecti == NULL) {
		vecti = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		vectiflag = 1;
	}
	
	INT biflag = 0;
	if (bi == NULL) {
		bi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		biflag = 1;
	}
	
	INT xnewiflag = 0;
	if (xnewi == NULL) {
		xnewi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnewiflag = 1;
	}
	
	INT Aiflag = 0;
	if (Ai == NULL) {
		Ai = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		Aiflag = 1;
	}
	
	INT xminflag = 0;
	if (xmin == NULL) {
		xmin = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xminflag = 1;
	}
	
	INT dflag = 0;
	if (d == NULL) {
		d = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		dflag = 1;
	}
	
	INT tflag = 0;
	if (t == NULL) {
		t = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		tflag = 1;
	}
	
	INT xsflag = 0;
	if (xs == NULL) {
		xs = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xsflag = 1;
	}
	
	DOUBLE EPS = POW(10, -9);
	
	CHAR uplo = 'U';
	INT SYMVN = K;
	DOUBLE alpha = 1;
	INT SYMVLDA = K;
	INT incx = 1;
	DOUBLE beta = 1;
	INT incy = 1;

	if (init == 0) {
		memset((void *) x, 0, K * sizeof(DOUBLE));
	}
	datacpy(grad, b, K);
	SYMV(&uplo, &SYMVN, &alpha, A, &SYMVLDA, x, &incx, &beta, grad, &incy);
	
	INT iterN;
	INT iterK;
	INT iterK2;
	DOUBLE ma = 0;
	INT mi = 0;
	
	for (iterK = 0; iterK < K; ++iterK) {
		if (ABS(grad[iterK]) > ma) {
			ma = ABS(grad[iterK]);
			mi = iterK;
		}
	}
	
	INT actcnt;
	INT idxcnt;
	INT scnt;
	DOUBLE dotprod;
	DOUBLE abssum;
	DOUBLE o_new;
	DOUBLE o_min;
	INT zd;
	DOUBLE o_s;
	
	INT AXPYN;
	INT SCALN;
	INT POTRSN;
	INT POTRSNRHS;
	INT POTRSLDA;
	INT POTRSLDB;
	INT INFO;
	INT DOTN;
	INT ASUMN;

	while (1) {
		
		if (grad[mi] > lambda + EPS) {
			x[mi] = (lambda - grad[mi]) / A[mi * K + mi];
		} else if (grad[mi] < - lambda - EPS) {
			x[mi] = (- lambda - grad[mi]) / A[mi * K + mi];
		} else {
			for (iterK = 0; iterK < K; ++iterK) {
				if (x[iterK] != 0) {
					break;
				}
			}
			if (iterK == K) {	
				break;
			}
		}
			
		while (1) {
			
			actcnt = 0; /* a */
			for (iterK = 0; iterK < K; ++iterK) {
				if (x[iterK] != 0) {
					actset[actcnt++] = iterK;
				}
			}
			
			for (iterK = 0; iterK < actcnt; ++iterK) {
				xa[iterK] = x[actset[iterK]];
				ba[iterK] = b[actset[iterK]];
				signa[iterK] = SIGN(x[actset[iterK]]);
				for (iterN = 0; iterN < iterK + 1; ++iterN) {
					Aa[actcnt * iterK + iterN] = A[K * actset[iterK] + actset[iterN]];
				}
			}
			
			datacpy(vect, ba, actcnt);
			
			AXPYN = actcnt;
			alpha = lambda;
			AXPY(&AXPYN, &alpha, signa, &incx, vect, &incy);

			SCALN = actcnt;
			alpha = -1;
			SCAL(&SCALN, &alpha, vect, &incx);
			
			datacpy(xnew, vect, actcnt);
			
			uplo = 'U';
			POTRSN = actcnt;
			POTRSNRHS = 1;
			POTRSLDA = actcnt;
			POTRSLDB = actcnt;
			
			/*
			 * TODO: Accelerate this. Perhaps find way to get Cholesky
			 * factorization of submatrix from one of full matrix.
			 */
			datacpy(Ai, Aa, actcnt * actcnt);
			POTRF(&uplo, &POTRSN, Ai, &POTRSLDA, &INFO);
			if (INFO != 0) {
				printf("Error, INFO = %d. LAPACK POTRF error.", INFO);
			}	
			POTRS(&uplo, &POTRSN, &POTRSNRHS, Ai, &POTRSLDA, xnew, &POTRSLDB, &INFO);
			if (INFO != 0) {
				printf("Error, INFO = %d. LAPACK POTRF error.", INFO);
			}	

			idxcnt = 0;
			for (iterK = 0; iterK < actcnt; ++iterK) {
				if (xnew[iterK] != 0) {
					idxset[idxcnt++] = iterK;
				}
			}
			
			for (iterK = 0; iterK < idxcnt; ++iterK) {
				vecti[iterK] = vect[idxset[iterK]];
				bi[iterK] = ba[idxset[iterK]];
				xnewi[iterK] = xnew[idxset[iterK]];
			}
			
			AXPYN = idxcnt;
			alpha = 0.5;
			AXPY(&AXPYN, &alpha, vecti, &incx, bi, &incy);
			
			DOTN = idxcnt;
			dotprod = DOT(&DOTN, bi, &incx, xnewi, &incy);
			
			ASUMN = idxcnt;
			abssum = ASUM(&ASUMN, xnewi, &incx);
			o_new = dotprod + lambda * abssum;

			scnt = 0;
			for (iterK = 0; iterK < actcnt; ++iterK) {
				if (xnew[iterK] * xa[iterK] <= 0) {
					sset[scnt++] = iterK;
				}
			}
			
			if (scnt == 0) {
				for (iterK = 0; iterK < actcnt; ++iterK) {
					x[actset[iterK]] = xnew[iterK];
				}
				break;
			}
			
			o_min = o_new;
			datacpy(xmin, xnew, actcnt);
			
			datacpy(d, xnew, actcnt);
			AXPYN = actcnt;
			alpha = -1;
			AXPY(&AXPYN, &alpha, xa, &incx, d, &incy);
			
			
			for (iterK = 0; iterK < actcnt; ++iterK) {
				t[iterK] = d[iterK] / xa[iterK];
			}
			
			for (iterK = 0; iterK < scnt; ++iterK) {
				zd = sset[iterK];
				
				datacpy(xs, xa, actcnt);
				AXPYN = actcnt;
				alpha = - ((DOUBLE) 1) / t[zd];
				AXPY(&AXPYN, &alpha, d, &incx, xs, &incy);
				xs[zd] = 0;
				
				idxcnt = 0;
				for (iterK2 = 0; iterK2 < actcnt; ++iterK2) {
					if (xs[iterK2] != 0) {
						idxset[idxcnt++] = iterK2;
					}
				}
				
				for (iterK2 = 0; iterK2 < idxcnt; ++iterK2) {
					vecti[iterK2] = xs[idxset[iterK2]]; /* vecti = xs(idx) */
					bi[iterK2] = ba[idxset[iterK2]]; /* bi = ba(idx) */
					for (iterN = 0; iterN < iterK2 + 1; ++iterN) {
						Ai[idxcnt * iterK2 + iterN] = Aa[actcnt * idxset[iterK2] + idxset[iterN]];
					}
				}
				
				uplo = 'U';
				SYMVN = idxcnt;
				alpha = 0.5;
				SYMVLDA = idxcnt;
				beta = 1;
				
				SYMV(&uplo, &SYMVN, &alpha, Ai, &SYMVLDA, vecti, &incx, &beta, bi, &incy);

				DOTN = idxcnt;
				dotprod = DOT(&DOTN, vecti, &incx, bi, &incy);

				ASUMN = idxcnt;
				abssum = ASUM(&ASUMN, vecti, &incx);
				o_s = dotprod + lambda * abssum;
				
				if (o_s < o_min) {		
					datacpy(xmin, xs, actcnt);
					o_min = o_s;
				}
			}
			
			for (iterK = 0; iterK < actcnt; ++iterK) {
				x[actset[iterK]] = xmin[iterK];
			}
		}
		
		uplo = 'U';
		SYMVN = K;
		alpha = 1;
		SYMVLDA = K;
		incx = 1;
		beta = 1;
		incy = 1;

		datacpy(grad, b, K);
		SYMV(&uplo, &SYMVN, &alpha, A, &SYMVLDA, x, &incx, &beta, grad, &incy);
		
		ma = 0;
		mi = 0;
			
		for (iterK = 0; iterK < K; ++iterK) {
			if ((x[iterK] == 0) && (ABS(grad[iterK]) > ma)) {
				ma = ABS(grad[iterK]);
				mi = iterK;
			}
		}

		if (ma <= lambda + EPS) {
			break;
		}
	}
	
	if (gradflag == 1) {
		CFREE(grad);
	}
	
	if (actsetflag == 1) {
		CFREE(actset);
	}
	
	if (xaflag == 1) {
		CFREE(xa);
	}
	
	if (baflag == 1) {
		CFREE(ba);
	}
	
	if (Aaflag == 1) {
		CFREE(Aa);
	}
	
	if (signaflag == 1) {
		CFREE(signa);
	}
	
	if (vectflag == 1) {
		CFREE(vect);
	}
	
	if (xnewflag == 1) {
		CFREE(xnew);
	}
	
	if (idxsetflag == 1) {
		CFREE(idxset);
	}
	
	if (ssetflag == 1) {
		CFREE(sset);
	}
	
	if (vectiflag == 1) {
		CFREE(vecti);
	}
	
	if (biflag == 1) {
		CFREE(bi);
	}
	
	if (xnewiflag == 1) {
		CFREE(xnewi);
	}
	
	if (Aiflag == 1) {
		CFREE(Ai);
	}
	
	if (xminflag == 1) {
		CFREE(xmin);
	}
	
	if (dflag == 1) {
		CFREE(d);
	}
	
	if (tflag == 1) {
		CFREE(t);
	}
	
	if (xsflag == 1) {
		CFREE(xs);
	}
}

void l1qp_featuresign(DOUBLE *S, DOUBLE *X, DOUBLE *D, DOUBLE *lambdap, INT N, INT K, \
			INT numSamples, DOUBLE *regparamp, DOUBLE *KDD, DOUBLE *KDX) {
	
	DOUBLE lambda = *lambdap;
	DOUBLE regparam;
	if (regparamp == NULL) {
		regparam = 0.0001;
	} else {
		regparam = *regparamp;
	}
	
	INT iterK;
	INT iterX;
	
	CHAR uplo = 'U';
	CHAR trans = 'T';
	INT SYRKN = K;
	INT SYRKK = N;
	DOUBLE alpha = 1;
	INT SYRKLDA = N;
	DOUBLE beta = 0;
	INT SYRKLDC = K;

	DOUBLE *A;
	INT AFlag = 0;
	if (KDD == NULL) {
		A = (DOUBLE *) MALLOC(K * K * sizeof(DOUBLE));
		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, D, &SYRKLDA, &beta, A, &SYRKLDC);
		AFlag = 1;
	} else {
		A = KDD;
	}
	
	for (iterK = 0; iterK < K; ++iterK) {
		A[iterK * K + iterK] += 2 * regparam;
	}

	CHAR transa = 'T';
	CHAR transb = 'N';
	INT GEMMM = K;
	INT GEMMN = numSamples;
	INT GEMMK = N;
	alpha = - 1;
	INT GEMMLDA = N;
	INT GEMMLDB = N;
	beta = 0;
	INT GEMMLDC = K;
	
	INT SCALN = K * numSamples;
	INT incx = 1;

	DOUBLE *bAll;
	INT bAllFlag = 0;
	if (KDX == NULL) {
		bAll = (DOUBLE *) MALLOC(K * numSamples * sizeof(DOUBLE));
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, D, &GEMMLDA, X, &GEMMLDB, &beta, bAll, &GEMMLDC);
		bAllFlag = 1;
	} else {
		bAll = KDX;
		SCAL(&SCALN, &alpha, bAll, &incx);
	}

	DOUBLE *grad;
	INT *actset;
	DOUBLE *xa;
	DOUBLE *ba;
	DOUBLE *Aa;
	DOUBLE *signa;
	DOUBLE *vect;
	DOUBLE *xnew;
	INT *idxset;
	INT *sset;
	DOUBLE *vecti;
	DOUBLE *bi;
	DOUBLE *xnewi;
	DOUBLE *Ai;
	DOUBLE *xmin;
	DOUBLE *d;
	DOUBLE *t;
	DOUBLE *xs;
	
	#pragma omp parallel private(iterX, grad, actset, xa, ba, Aa, signa, vect, xnew, idxset, sset, \
			vecti, bi, xnewi, Ai, xmin, d, t, xs) shared(S, A, bAll) firstprivate(K, lambda, numSamples)
	{
		
		grad = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		actset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		xa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		ba = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		Aa = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		signa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		vect = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnew = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		idxset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		sset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		vecti = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		bi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnewi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		Ai = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		xmin = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		d = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		t = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xs = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
	
		#pragma omp for
		for (iterX = 0; iterX < numSamples; ++iterX) {
			l1_featuresign_sub(&S[K * iterX], A, &bAll[K * iterX], lambda, K, \
								grad, actset, xa, ba, \
								Aa, signa, vect, xnew, idxset, \
								sset, vecti, bi, xnewi, Ai, xmin, d, t, xs, 0);
		}
		
		CFREE(grad);
		CFREE(actset);
		CFREE(xa);
		CFREE(ba);
		CFREE(Aa);
		CFREE(signa);
		CFREE(vect);
		CFREE(xnew);
		CFREE(idxset);
		CFREE(sset);
		CFREE(vecti);
		CFREE(bi);
		CFREE(xnewi);
		CFREE(Ai);
		CFREE(xmin);
		CFREE(d);
		CFREE(t);
		CFREE(xs);
	}
	
	if (AFlag == 1) {
		FREE(A);
	} else {
		for (iterK = 0; iterK < K; ++iterK) {
			A[iterK * K + iterK] -= 2 * regparam;
		}
	}

	if (bAllFlag == 1) {
		FREE(bAll);
	} else {
		alpha = - 1.0;
		SCAL(&SCALN, &alpha, bAll, &incx);
	}
}

void l1kernel_featuresign(DOUBLE *S, DOUBLE *KDX, DOUBLE *KDD, DOUBLE *lambdap, \
			INT K, INT numSamples, DOUBLE *regparamp) {
	
	DOUBLE lambda = *lambdap;
	DOUBLE regparam;
	if (regparamp == NULL) {
		regparam = 0.0001;
	} else {
		regparam = *regparamp;
	}
	
	INT iterK;
	INT iterX;
	
	DOUBLE *A = KDD;

	for (iterK = 0; iterK < K; ++iterK) {
		A[iterK * K + iterK] += 2 * regparam;
	}

	DOUBLE *bAll = KDX;

	INT SCALN = K * numSamples;
	DOUBLE alpha = -1;
	INT incx = 1;
	SCAL(&SCALN, &alpha, bAll, &incx);

	DOUBLE *grad;
	INT *actset;
	DOUBLE *xa;
	DOUBLE *ba;
	DOUBLE *Aa;
	DOUBLE *signa;
	DOUBLE *vect;
	DOUBLE *xnew;
	INT *idxset;
	INT *sset;
	DOUBLE *vecti;
	DOUBLE *bi;
	DOUBLE *xnewi;
	DOUBLE *Ai;
	DOUBLE *xmin;
	DOUBLE *d;
	DOUBLE *t;
	DOUBLE *xs;
	
	#pragma omp parallel private(iterX, grad, actset, xa, ba, Aa, signa, vect, xnew, idxset, sset, \
			vecti, bi, xnewi, Ai, xmin, d, t, xs) shared(S, A, bAll) firstprivate(K, lambda, numSamples)
	{
		
		grad = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		actset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		xa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		ba = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		Aa = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		signa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		vect = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnew = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		idxset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		sset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		vecti = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		bi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnewi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		Ai = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		xmin = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		d = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		t = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xs = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
	
		#pragma omp for
		for (iterX = 0; iterX < numSamples; ++iterX) {
			l1_featuresign_sub(&S[K * iterX], A, &bAll[K * iterX], lambda, K, \
								grad, actset, xa, ba, \
								Aa, signa, vect, xnew, idxset, \
								sset, vecti, bi, xnewi, Ai, xmin, d, t, xs, 0);
		}
		
		CFREE(grad);
		CFREE(actset);
		CFREE(xa);
		CFREE(ba);
		CFREE(Aa);
		CFREE(signa);
		CFREE(vect);
		CFREE(xnew);
		CFREE(idxset);
		CFREE(sset);
		CFREE(vecti);
		CFREE(bi);
		CFREE(xnewi);
		CFREE(Ai);
		CFREE(xmin);
		CFREE(d);
		CFREE(t);
		CFREE(xs);
	}
	
	for (iterK = 0; iterK < K; ++iterK) {
		A[iterK * K + iterK] -= 2 * regparam;
	}

	alpha = - 1.0;
	SCAL(&SCALN, &alpha, bAll, &incx);
}

void l1exp_featuresign_sub(DOUBLE *s, DOUBLE *x, DOUBLE *Dt, DOUBLE lambda, INT N, INT K, EXPONENTIAL_TYPE family, DOUBLE regparam, \
						DOUBLE *grad, INT *actset, DOUBLE *xa, DOUBLE *ba, \
						DOUBLE *Aa, DOUBLE *signa, DOUBLE *vect, DOUBLE *xnew, INT *idxset, \
						INT *sset, DOUBLE *vecti, DOUBLE *bi, DOUBLE *xnewi, DOUBLE *Ai, \
						DOUBLE *xmin, DOUBLE *d, DOUBLE *t, DOUBLE *xs, DOUBLE *Ds, DOUBLE *xtilde, \
						DOUBLE *Dttilde, DOUBLE *A, DOUBLE *b, DOUBLE *shat, \
						DOUBLE *aPrime, DOUBLE *aDoublePrime, DOUBLE *deriv) {
	
	INT gradflag = 0;
	if (grad == NULL) {
		grad = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		gradflag = 1;
	}
	
	INT actsetflag = 0;
	if (actset == NULL) {
		actset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		actsetflag = 1;
	}
	
	INT xaflag = 0;
	if (xa == NULL) {
		xa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xaflag = 1;
	}
	
	INT baflag = 0;
	if (ba == NULL) {
		ba = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		baflag = 1;
	}
	
	INT Aaflag = 0;
	if (Aa == NULL) {
		Aa = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		Aaflag = 1;
	}
	
	INT signaflag = 0;
	if (signa == NULL) {
		signa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		signaflag = 1;
	}
	
	INT vectflag = 0;
	if (vect == NULL) {
		vect = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		vectflag = 1;
	}
	
	INT xnewflag = 0;
	if (xnew == NULL) {
		xnew = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnewflag = 1;
	}
	
	INT idxsetflag = 0;
	if (idxset == NULL) {
		idxset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		idxsetflag = 1;
	}
	
	INT ssetflag = 0;
	if (sset == NULL) {
		sset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		ssetflag = 1;
	}
	
	INT vectiflag = 0;
	if (vecti == NULL) {
		vecti = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		vectiflag = 1;
	}
	
	INT biflag = 0;
	if (bi == NULL) {
		bi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		biflag = 1;
	}
	
	INT xnewiflag = 0;
	if (xnewi == NULL) {
		xnewi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnewiflag = 1;
	}
	
	INT Aiflag = 0;
	if (Ai == NULL) {
		Ai = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		Aiflag = 1;
	}
	
	INT xminflag = 0;
	if (xmin == NULL) {
		xmin = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xminflag = 1;
	}
	
	INT dflag = 0;
	if (d == NULL) {
		d = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		dflag = 1;
	}
	
	INT tflag = 0;
	if (t == NULL) {
		t = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		tflag = 1;
	}
	
	INT xsflag = 0;
	if (xs == NULL) {
		xs = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xsflag = 1;
	}
	
	INT Dsflag = 0;
	if (Ds == NULL) {
		Ds = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		Dsflag = 1;
	}
	
	INT xtildeflag = 0;
	if (xtilde == NULL) {
		xtilde = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		xtildeflag = 1;
	}
	
/*
	INT Dtildeflag = 0;
	if (Dtilde == NULL) {
		Dtilde = (DOUBLE *) CMALLOC(N * K * sizeof(DOUBLE));
		Dtildeflag = 1;
	}
*/
	
	INT Dttildeflag = 0;
	if (Dttilde == NULL) {
		Dttilde = (DOUBLE *) CMALLOC(K * N * sizeof(DOUBLE));
		Dttildeflag = 1;
	}
	
	INT Aflag = 0;
	if (A == NULL) {
		A = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		Aflag = 1;
	}
	
	INT bflag = 0;
	if (b == NULL) {
		b = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		bflag = 1;
	}
	
	INT shatflag = 0;
	if (shat == NULL) {
		shat = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		shatflag = 1;
	}
	
	INT aPrimeflag = 0;
	if (aPrime == NULL) {
		aPrime = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		aPrimeflag = 1;
	}
	
	INT aDoublePrimeflag = 0;
	if (aDoublePrime == NULL) {
		aDoublePrime = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		aDoublePrimeflag = 1;
	}
	
	INT derivflag = 0;
	if (deriv == NULL) {
		deriv = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		derivflag = 1;
	}
	
/*
	INT LMatrixSqrtflag = 0;
	if (LMatrixSqrt == NULL) {
		LMatrixSqrt = (DOUBLE *) CMALLOC(N * N * sizeof(DOUBLE));
		LMatrixSqrtflag = 1;
	}
*/
	
	CHAR trans = 'N';
/*
	CHAR trans2 = 'T';
*/
	
	INT GEMVM = K;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = K;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
/*
	INT DOTN = N;
*/
	INT DOTN2 = K;
	INT ASUMN = K;
	INT SCALN = K;
	INT AXPYN = K;
	INT AXPYN2 = N;
	
	CHAR uplo = 'U';
/*
	CHAR side = 'L';
	CHAR transa = 'N';
	CHAR diag = 'N';
	INT TRMMM = N;
	INT TRMMN = K;
	INT TRMMLDA = N;
	INT TRMMLDB = N;
*/
	
/*
	INT SYRKN = K;
	INT SYRKK = N;
	INT SYRKLDA = N;
	INT SYRKLDC = K;
*/
	
	INT SYRKN = K;
	INT SYRKK = N;
	INT SYRKLDA = K;
	INT SYRKLDC = K;
	
/*
	INT GEMVM2 = N;
	INT GEMVN2 = K;
	DOUBLE alpha2 = -1;
	DOUBLE beta2 = 0;
	INT GEMVLDA2 = N;
	DOUBLE beta2 = 0;
*/

	INT GEMVM2 = K;
	INT GEMVN2 = N;
	DOUBLE alpha2 = -1;
	DOUBLE beta2 = 0;
	INT GEMVLDA2 = K;
	
	DOUBLE LMatrixSqrtEl;
	
	DOUBLE BETA = 0.9;
	DOUBLE ALPHA = 0.3;
	DOUBLE EPS = POW(10, -12);
	INT MAXITER = 500;
	INT MAXITEROUT = 200;
	DOUBLE tf;
	DOUBLE tAdd;
	DOUBLE p;
	
	DOUBLE objVal;
	DOUBLE objValOld;
	DOUBLE aVal;
	DOUBLE l1Norm;
	DOUBLE preObjVal;
	DOUBLE postObjVal;
	INT iterX;
	INT iterK;
	INT iterBT;
	INT iterOut;
/*
	DOUBLE LMatrixSqrtInvEl;
*/
	
	memset((void *) s, 0, K * sizeof(DOUBLE));
	memset((void *) Ds, 0, N * 1 * sizeof(DOUBLE));
	link_func(&aVal, aPrime, aDoublePrime, Ds, N, 1, family, 1, 1, 1);
	objVal = aVal;
	l1Norm = 0;
	iterOut = 0;
	while (1) {
/*
		for (iterX = 0; iterX < N; ++iterX) {
			LMatrixSqrtEl = SQRT(aDoublePrime[iterX]);
			LMatrixSqrtInvEl = 1 / LMatrixSqrtEl;
			if (ISNAN(LMatrixSqrtInvEl)) {
				LMatrixSqrtInvEl = 0;
			}
			LMatrixSqrt[iterX * N + iterX] = LMatrixSqrtEl;
			xtilde[iterX] = LMatrixSqrtInvEl * (x[iterX] - aPrime[iterX]) + LMatrixSqrtEl * Ds[iterX];
		}
		
		datacpy(Dtilde, D, K * N);
		TRMM(&side, &uplo, &transa, &diag, &TRMMM, &TRMMN, &alpha, LMatrixSqrt, &TRMMLDA, Dtilde, &TRMMLDB);

		SYRK(&uplo, &trans2, &SYRKN, &SYRKK, &alpha, Dtilde, &SYRKLDA, &beta, A, &SYRKLDC);
	
		for (iterK = 0; iterK < K; ++iterK) {
			A[iterK * K + iterK] += 2 * regparam;
		}
	
		GEMV(&trans2, &GEMVM2, &GEMVN2, &alpha2, Dtilde, &GEMVLDA, xtilde, &incx, &beta2, b, &incy);
*/

		datacpy(Dttilde, Dt, K * N);
		for (iterX = 0; iterX < N; ++iterX) {
			LMatrixSqrtEl = SQRT(aDoublePrime[iterX]);
			xtilde[iterX] = x[iterX] - aPrime[iterX] + aDoublePrime[iterX] * Ds[iterX];
			SCAL(&SCALN, &LMatrixSqrtEl, &Dttilde[K * iterX], &incx);
		}

		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, Dttilde, &SYRKLDA, &beta, A, &SYRKLDC);
	
		for (iterK = 0; iterK < K; ++iterK) {
			A[iterK * K + iterK] += 2 * regparam;
		}
	
		GEMV(&trans, &GEMVM2, &GEMVN2, &alpha2, Dt, &GEMVLDA2, xtilde, &incx, &beta2, b, &incy);
		
/*
 		LMatrix = diag(aDoublePrime);
 		xtilde = x - aPrime + LMatrix * Ds;
 		A = double(D' * LMatrix * D + 2 * beta * eye(size(D, 2)));
 		b = - D' * xtilde;
*/
		datacpy(shat, s, K);
		l1_featuresign_sub(shat, A, b, lambda, K, \
								grad, actset, xa, ba, \
								Aa, signa, vect, xnew, idxset, \
								sset, vecti, bi, xnewi, Ai, xmin, d, t, xs, 1);
	
		AXPY(&AXPYN, &alpha2, s, &incx, shat, &incy);
		tf = 1.0;
		preObjVal = objVal - lambda * l1Norm;
		AXPY(&AXPYN2, &alpha2, x, &incx, aPrime, &incy);
		GEMV(&trans, &GEMVM, &GEMVN, &alpha, Dt, &GEMVLDA, aPrime, &incx, &beta, deriv, &incy);
		p = DOT(&DOTN2, deriv, &incx, shat, &incy);
		tAdd = 1.0;
		
		for (iterBT = 0; iterBT < MAXITER; ++iterBT) {
			AXPY(&AXPYN, &tAdd, shat, &incx, s, &incy);
			l1exp_obj_subgrad(&postObjVal, NULL, s, Dt, x, N, K, 0, family, 0, Ds, NULL);
			if (postObjVal < preObjVal + ALPHA * tf * p) {
				break;
			} else {
				tAdd = (BETA - 1) * tf;
				tf = BETA * tf;
			}
		}
		
/*
		GEMV(&trans2, &GEMVM, &GEMVN, &alpha, Dt, &GEMVLDA, s, &incx, &beta, Ds, &incy);
*/
		link_func(&aVal, aPrime, aDoublePrime, Ds, N, 1, family, 0, 1, 1);
		
		objValOld = objVal;
/*
		objVal = - DOT(&DOTN, Ds, &incx, x, &incy);
		objVal += aVal;
*/
		objVal = postObjVal;
		l1Norm = ASUM(&ASUMN, s, &incx);
		objVal += lambda * l1Norm;
		
		if (ABS(objVal - objValOld) < EPS) {
			break;
		}	
		if (++iterOut > MAXITEROUT) {
			break;
		}
	}
	
	if (gradflag == 1) {
		CFREE(grad);
	}
	
	if (actsetflag == 1) {
		CFREE(actset);
	}
	
	if (xaflag == 1) {
		CFREE(xa);
	}
	
	if (baflag == 1) {
		CFREE(ba);
	}
	
	if (Aaflag == 1) {
		CFREE(Aa);
	}
	
	if (signaflag == 1) {
		CFREE(signa);
	}
	
	if (vectflag == 1) {
		CFREE(vect);
	}
	
	if (xnewflag == 1) {
		CFREE(xnew);
	}
	
	if (idxsetflag == 1) {
		CFREE(idxset);
	}
	
	if (ssetflag == 1) {
		CFREE(sset);
	}
	
	if (vectiflag == 1) {
		CFREE(vecti);
	}
	
	if (biflag == 1) {
		CFREE(bi);
	}
	
	if (xnewiflag == 1) {
		CFREE(xnewi);
	}
	
	if (Aiflag == 1) {
		CFREE(Ai);
	}
	
	if (xminflag == 1) {
		CFREE(xmin);
	}
	
	if (dflag == 1) {
		CFREE(d);
	}
	
	if (tflag == 1) {
		CFREE(t);
	}
	
	if (xsflag == 1) {
		CFREE(xs);
	}
	
	if (Dsflag == 1) {
		CFREE(Ds);
	}
	
	if (xtildeflag == 1) {
		CFREE(xtilde);
	}
	
/*
	if (Dtildeflag == 1) {
		CFREE(Dtilde);
	}
*/
	
	if (Dttildeflag == 1) {
		CFREE(Dttilde);
	}
	
	if (Aflag == 1) {
		CFREE(A);
	}
	
	if (bflag == 1) {
		CFREE(b);
	}
	
	if (shatflag == 1) {
		CFREE(shat);
	}
	
	if (aPrimeflag == 1) {
		CFREE(aPrime);
	}
	
	if (aDoublePrimeflag == 1) {
		CFREE(aDoublePrime);
	}
	
	if (derivflag == 1) {
		CFREE(deriv);
	}
	
/*
	if (LMatrixSqrtflag == 1) {
		CFREE(LMatrixSqrt);
	}
*/
}	
		
void l1exp_featuresign(DOUBLE *S, DOUBLE *X, DOUBLE *D, DOUBLE *lambdap, INT N, INT K, \
				INT numSamples, EXPONENTIAL_TYPE family, DOUBLE *regparamp) {
	
	DOUBLE lambda = *lambdap;
	DOUBLE regparam;
	if (regparamp == NULL) {
		regparam = 0.0001;
	} else {
		regparam = *regparamp;
	}
	
	DOUBLE *Dt = (DOUBLE *) MALLOC(K * N * sizeof(DOUBLE));
	transpose(D, Dt, N, K);
	
	INT iterX;
	
	DOUBLE *grad;
	INT *actset;
	DOUBLE *xa;
	DOUBLE *ba;
	DOUBLE *Aa;
	DOUBLE *signa;
	DOUBLE *vect;
	DOUBLE *xnew;
	INT *idxset;
	INT *sset;
	DOUBLE *vecti;
	DOUBLE *bi;
	DOUBLE *xnewi;
	DOUBLE *Ai;
	DOUBLE *xmin;
	DOUBLE *d;
	DOUBLE *t;
	DOUBLE *xs;
	DOUBLE *Ds;
	DOUBLE *xtilde;
	DOUBLE *Dttilde;
	DOUBLE *A;
	DOUBLE *b;
	DOUBLE *shat;
	DOUBLE *aPrime;
	DOUBLE *aDoublePrime;
	DOUBLE *deriv;
	
	#pragma omp parallel private(iterX, grad, actset, xa, ba, Aa, signa, vect, xnew, idxset, sset, vecti, \
			bi, xnewi, Ai, xmin, d, t, xs, Ds, xtilde, Dttilde, A, b, shat, aPrime, aDoublePrime, deriv) \
			shared(S) firstprivate(K, lambda, numSamples)
	{
		
		grad = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		actset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		xa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		ba = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		Aa = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		signa = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		vect = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnew = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		idxset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		sset = (INT *) CMALLOC(K * 1 * sizeof(INT));
		vecti = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		bi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xnewi = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		Ai = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		xmin = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		d = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		t = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		xs = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		Ds = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		xtilde = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		Dttilde = (DOUBLE *) CMALLOC(K * N * sizeof(DOUBLE));
		A = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		b = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		shat = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		aPrime = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		aDoublePrime = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		deriv = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
	
		#pragma omp for
		for (iterX = 0; iterX < numSamples; ++iterX) {
			l1exp_featuresign_sub(&S[K * iterX], &X[N * iterX], Dt, lambda, N, K, family, regparam, \
						grad, actset, xa, ba, \
						Aa, signa, vect, xnew, idxset, \
						sset, vecti, bi, xnewi, Ai, \
						xmin, d, t, xs, Ds, xtilde, \
						Dttilde, A, b, shat, aPrime, aDoublePrime, deriv);
		}
		
		CFREE(grad);
		CFREE(actset);
		CFREE(xa);
		CFREE(ba);
		CFREE(Aa);
		CFREE(signa);
		CFREE(vect);
		CFREE(xnew);
		CFREE(idxset);
		CFREE(sset);
		CFREE(vecti);
		CFREE(bi);
		CFREE(xnewi);
		CFREE(Ai);
		CFREE(xmin);
		CFREE(d);
		CFREE(t);
		CFREE(xs);
		CFREE(Ds);
		CFREE(xtilde);
		CFREE(Dttilde);
		CFREE(A);
		CFREE(b);
		CFREE(shat);
		CFREE(aPrime);
		CFREE(aDoublePrime);
		CFREE(deriv);
	}
	FREE(Dt);
}

void exp_irls_sub(DOUBLE *s, DOUBLE *x, DOUBLE *Dt, INT N, INT K, EXPONENTIAL_TYPE family, DOUBLE regparam, \
						DOUBLE *Ds, DOUBLE *xtilde, DOUBLE *Dttilde, DOUBLE *A, DOUBLE *shat, \
						DOUBLE *aPrime, DOUBLE *aDoublePrime, DOUBLE *deriv) {
	
	INT Dsflag = 0;
	if (Ds == NULL) {
		Ds = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		Dsflag = 1;
	}
	
	INT xtildeflag = 0;
	if (xtilde == NULL) {
		xtilde = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		xtildeflag = 1;
	}
	
/*
	INT Dtildeflag = 0;
	if (Dtilde == NULL) {
		Dtilde = (DOUBLE *) CMALLOC(N * K * sizeof(DOUBLE));
		Dtildeflag = 1;
	}
*/
	
	INT Dttildeflag = 0;
	if (Dttilde == NULL) {
		Dttilde = (DOUBLE *) CMALLOC(K * N * sizeof(DOUBLE));
		Dttildeflag = 1;
	}
	
	INT Aflag = 0;
	if (A == NULL) {
		A = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		Aflag = 1;
	}
	
	INT shatflag = 0;
	if (shat == NULL) {
		shat = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		shatflag = 1;
	}
	
	INT aPrimeflag = 0;
	if (aPrime == NULL) {
		aPrime = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		aPrimeflag = 1;
	}
	
	INT aDoublePrimeflag = 0;
	if (aDoublePrime == NULL) {
		aDoublePrime = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		aDoublePrimeflag = 1;
	}
	
	INT derivflag = 0;
	if (deriv == NULL) {
		deriv = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		derivflag = 1;
	}
	
/*
	INT LMatrixSqrtflag = 0;
	if (LMatrixSqrt == NULL) {
		LMatrixSqrt = (DOUBLE *) CMALLOC(N * N * sizeof(DOUBLE));
		LMatrixSqrtflag = 1;
	}
*/
	
	CHAR trans = 'N';
/*
	CHAR trans2 = 'T';
*/
	
	INT GEMVM = K;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = K;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
/*
	INT DOTN = N;
*/
	INT DOTN2 = K;
	INT SCALN = K;
	INT AXPYN = K;
	INT AXPYN2 = N;
	
	CHAR uplo = 'U';
	INT POTRSN = K;
	INT POTRSNRHS = 1;
	INT POTRSLDA = K;
	INT POTRSLDB = K;
	INT INFO;
/*
	CHAR side = 'L';
	CHAR transa = 'N';
	CHAR diag = 'N';
	INT TRMMM = N;
	INT TRMMN = K;
	INT TRMMLDA = N;
	INT TRMMLDB = N;
*/
	
/*
	INT SYRKN = K;
	INT SYRKK = N;
	INT SYRKLDA = N;
	INT SYRKLDC = K;
*/
	
	INT SYRKN = K;
	INT SYRKK = N;
	INT SYRKLDA = K;
	INT SYRKLDC = K;
	
/*
	INT GEMVM2 = N;
	INT GEMVN2 = K;
	DOUBLE alpha2 = -1;
	DOUBLE beta2 = 0;
	INT GEMVLDA2 = N;
	DOUBLE beta2 = 0;
*/

	INT GEMVM2 = K;
	INT GEMVN2 = N;
	DOUBLE alpha2 = -1;
	DOUBLE beta2 = 0;
	INT GEMVLDA2 = K;
	
	DOUBLE LMatrixSqrtEl;
	
	DOUBLE BETA = 0.9;
	DOUBLE ALPHA = 0.3;
	DOUBLE EPS = POW(10, -12);
	INT MAXITER = 500;
	INT MAXITEROUT = 200;
	DOUBLE tf;
	DOUBLE tAdd;
	DOUBLE p;
	
	DOUBLE objVal;
	DOUBLE objValOld;
	DOUBLE aVal;
	DOUBLE preObjVal;
	DOUBLE postObjVal;
	INT iterX;
	INT iterK;
	INT iterBT;
	INT iterOut;
/*
	DOUBLE LMatrixSqrtInvEl;
*/
	
	memset((void *) s, 0, K * sizeof(DOUBLE));
	memset((void *) Ds, 0, N * 1 * sizeof(DOUBLE));
	link_func(&aVal, aPrime, aDoublePrime, Ds, N, 1, family, 1, 1, 1);
	objVal = aVal;
	iterOut = 0;
	while (1) {
/*
		for (iterX = 0; iterX < N; ++iterX) {
			LMatrixSqrtEl = SQRT(aDoublePrime[iterX]);
			LMatrixSqrtInvEl = 1 / LMatrixSqrtEl;
			if (ISNAN(LMatrixSqrtInvEl)) {
				LMatrixSqrtInvEl = 0;
			}
			LMatrixSqrt[iterX * N + iterX] = LMatrixSqrtEl;
			xtilde[iterX] = LMatrixSqrtInvEl * (x[iterX] - aPrime[iterX]) + LMatrixSqrtEl * Ds[iterX];
		}
		
		datacpy(Dtilde, D, K * N);
		TRMM(&side, &uplo, &transa, &diag, &TRMMM, &TRMMN, &alpha, LMatrixSqrt, &TRMMLDA, Dtilde, &TRMMLDB);

		SYRK(&uplo, &trans2, &SYRKN, &SYRKK, &alpha, Dtilde, &SYRKLDA, &beta, A, &SYRKLDC);
	
		for (iterK = 0; iterK < K; ++iterK) {
			A[iterK * K + iterK] += 2 * regparam;
		}
	
		GEMV(&trans2, &GEMVM2, &GEMVN2, &alpha2, Dtilde, &GEMVLDA, xtilde, &incx, &beta2, b, &incy);
*/
		
		datacpy(Dttilde, Dt, K * N);
		for (iterX = 0; iterX < N; ++iterX) {
			LMatrixSqrtEl = SQRT(aDoublePrime[iterX]);
			xtilde[iterX] = x[iterX] - aPrime[iterX] + aDoublePrime[iterX] * Ds[iterX];
			SCAL(&SCALN, &LMatrixSqrtEl, &Dttilde[K * iterX], &incx);
		}

		SYRK(&uplo, &trans, &SYRKN, &SYRKK, &alpha, Dttilde, &SYRKLDA, &beta, A, &SYRKLDC);
	
		for (iterK = 0; iterK < K; ++iterK) {
			A[iterK * K + iterK] += 2 * regparam;
		}
	
		GEMV(&trans, &GEMVM2, &GEMVN2, &alpha, Dt, &GEMVLDA2, xtilde, &incx, &beta2, shat, &incy);
		
/*
 		LMatrix = diag(aDoublePrime);
 		xtilde = x - aPrime + LMatrix * Ds;
 		A = double(D' * LMatrix * D + 2 * beta * eye(size(D, 2)));
 		b = - D' * xtilde;
*/
		
		POTRF(&uplo, &POTRSN, A, &POTRSLDA, &INFO);
		if (INFO != 0) {
			printf("Error, INFO = %d. LAPACK POTRF error.", INFO);
		}	
		POTRS(&uplo, &POTRSN, &POTRSNRHS, A, &POTRSLDA, shat, &POTRSLDB, &INFO);
		if (INFO != 0) {
			printf("Error, INFO = %d. LAPACK POTRF error.", INFO);
		}
		
/*
		l1qp_featuresign_sub(shat, A, b, lambda, K, \
								grad, actset, xa, ba, \
								Aa, signa, vect, xnew, idxset, \
								sset, vecti, bi, xnewi, Ai, xmin, d, t, xs, 1);
*/
	
		AXPY(&AXPYN, &alpha2, s, &incx, shat, &incy);
		tf = 1.0;
		preObjVal = objVal;
		AXPY(&AXPYN2, &alpha2, x, &incx, aPrime, &incy);
		GEMV(&trans, &GEMVM, &GEMVN, &alpha, Dt, &GEMVLDA, aPrime, &incx, &beta, deriv, &incy);
		p = DOT(&DOTN2, deriv, &incx, shat, &incy);
		tAdd = 1.0;
		
		for (iterBT = 0; iterBT < MAXITER; ++iterBT) {
			AXPY(&AXPYN, &tAdd, shat, &incx, s, &incy);
			l1exp_obj_subgrad(&postObjVal, NULL, s, Dt, x, N, K, 0, family, 0, Ds, NULL);
			if (postObjVal < preObjVal + ALPHA * tf * p) {
				break;
			} else {
				tAdd = (BETA - 1) * tf;
				tf = BETA * tf;
			}
		}
		
/*
		GEMV(&trans2, &GEMVM, &GEMVN, &alpha, Dt, &GEMVLDA, s, &incx, &beta, Ds, &incy);
*/
		link_func(&aVal, aPrime, aDoublePrime, Ds, N, 1, family, 0, 1, 1);
		
		objValOld = objVal;
/*
		objVal = - DOT(&DOTN, Ds, &incx, x, &incy);
		objVal += aVal;
*/
		objVal = postObjVal;
		
		if (ABS(objVal - objValOld) < EPS) {
			break;
		}	
		if (++iterOut > MAXITEROUT) {
			break;
		}
	}

	if (Dsflag == 1) {
		CFREE(Ds);
	}
	
	if (xtildeflag == 1) {
		CFREE(xtilde);
	}
	
/*
	if (Dtildeflag == 1) {
		CFREE(Dtilde);
	}
*/
	
	if (Dttildeflag == 1) {
		CFREE(Dttilde);
	}
	
	if (Aflag == 1) {
		CFREE(A);
	}
	
	if (shatflag == 1) {
		CFREE(shat);
	}
	
	if (aPrimeflag == 1) {
		CFREE(aPrime);
	}
	
	if (aDoublePrimeflag == 1) {
		CFREE(aDoublePrime);
	}
	
	if (derivflag == 1) {
		CFREE(deriv);
	}
	
/*
	if (LMatrixSqrtflag == 1) {
		CFREE(LMatrixSqrt);
	}
*/
}

void exp_irls(DOUBLE *S, DOUBLE *X, DOUBLE *D, INT N, INT K, \
				INT numSamples, EXPONENTIAL_TYPE family, DOUBLE *regparamp) {
	
	DOUBLE regparam;
	if (regparamp == NULL) {
		regparam = 0.0001;
	} else {
		regparam = *regparamp;
	}
	
	DOUBLE *Dt = (DOUBLE *) MALLOC(K * N * sizeof(DOUBLE));
	transpose(D, Dt, N, K);
	
	INT iterX;
	
	DOUBLE *Ds;
	DOUBLE *xtilde;
	DOUBLE *Dttilde;
	DOUBLE *A;
	DOUBLE *shat;
	DOUBLE *aPrime;
	DOUBLE *aDoublePrime;
	DOUBLE *deriv;
	
	#pragma omp parallel private(iterX, Ds, xtilde, Dttilde, A, shat, aPrime, aDoublePrime, deriv) \
			shared(S) firstprivate(K, numSamples)
	{
		
		Ds = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		xtilde = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		Dttilde = (DOUBLE *) CMALLOC(K * N * sizeof(DOUBLE));
		A = (DOUBLE *) CMALLOC(K * K * sizeof(DOUBLE));
		shat = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
		aPrime = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		aDoublePrime = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		deriv = (DOUBLE *) CMALLOC(K * 1 * sizeof(DOUBLE));
	
		#pragma omp for
		for (iterX = 0; iterX < numSamples; ++iterX) {
			exp_irls_sub(&S[K * iterX], &X[N * iterX], Dt, N, K, family, regparam, \
						Ds, xtilde, Dttilde, A, shat, aPrime, aDoublePrime, deriv);
		}
		
		CFREE(Ds);
		CFREE(xtilde);
		CFREE(Dttilde);
		CFREE(A);
		CFREE(shat);
		CFREE(aPrime);
		CFREE(aDoublePrime);
		CFREE(deriv);
	}
	FREE(Dt);
}

void l1exp_obj_subgrad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *s, DOUBLE *Dt, DOUBLE *x, INT N, INT K, \
		DOUBLE lambda, EXPONENTIAL_TYPE family, INT derivFlag, DOUBLE *Ds, DOUBLE *aPrime) {
	
	INT Dsflag = 0;
	if (Ds == NULL) {
		Ds = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		Dsflag = 1;
	}
	
	INT aPrimeflag = 0;
	if ((derivFlag == 1) && (aPrime == NULL)) {
		aPrime = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		aPrimeflag = 1;
	}
	
	CHAR trans = 'T';
	INT GEMVM = K;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = K;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Dt, &GEMVLDA, s, &incx, &beta, Ds, &incy);
	
	INT DOTN = N;
	INT ASUMN = K;
	DOUBLE aVal;
	if (derivFlag == 1) {
		link_func(&aVal, aPrime, NULL, Ds, N, 1, family, 1, 1, 0);
	} else {
		link_func(&aVal, NULL, NULL, Ds, N, 1, family, 1, 0, 0);
	}
	*obj = - DOT(&DOTN, Ds, &incx, x, &incy);
	*obj += aVal;
	if (lambda > 0) {
		*obj += lambda * ASUM(&ASUMN, s, &incx);
	}
	
	if (derivFlag == 1) {
		
		INT AXPYN = N;
		alpha = -1;
		AXPY(&AXPYN, &alpha, x, &incx, aPrime, &incy);

		trans = 'N';
		GEMV(&trans, &GEMVM, &GEMVN, &alpha, Dt, &GEMVLDA, aPrime, &incx, &beta, deriv, &incy);
		
		if (lambda > 0) {
			INT iterK;
			for (iterK = 0; iterK < K; ++iterK) {
				if (s[iterK] > 0) {
					deriv[iterK] += lambda;
				} else if (s[iterK] < 0) {
					deriv[iterK] -= lambda;
				}
			}
		}
	}
	
	if (Dsflag == 1) {
		CFREE(Ds);
	}
	
	if ((derivFlag == 1) && (aPrimeflag == 1)) {
		CFREE(aPrime);
	}
}
