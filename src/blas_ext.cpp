/*
 * blas_ext.cpp
 *
 *  Created on: Nov 13, 2011
 *      Author: igkiou
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "blas_ext.h"

void dimm(CHAR *side, CHAR *trans, INT *m, INT *n, DOUBLE *alpha, DOUBLE *a, DOUBLE *b, DOUBLE *beta, DOUBLE *c) {

	INT M = *m;
	INT N = *n;
	INT incx;
	INT iter;
	DOUBLE alphaval;
	if (*beta == 0) {
		INT SCALN;
		datacpy(c, b, M * N);
		if (*side == 'L') {
			SCALN = N;
			incx = M;
			if (*trans == 'M') {
				for (iter = 0; iter < M; ++iter) {
					alphaval = *alpha * a[iter * M + iter];
					SCAL(&SCALN, &alphaval, &c[iter], &incx);
				}
			} else if (*trans == 'V') {
				for (iter = 0; iter < M; ++iter) {
					alphaval = *alpha * a[iter];
					SCAL(&SCALN, &alphaval, &c[iter], &incx);
				}
			}
		} else if (*side == 'R') {
			SCALN = M;
			incx = 1;
			if (*trans == 'M') {
				for (iter = 0; iter < N; ++iter) {
					alphaval = *alpha * a[iter * M + iter];
					SCAL(&SCALN, &alphaval, &c[iter * M], &incx);
				}
			} else if (*trans == 'V') {
				for (iter = 0; iter < N; ++iter) {
					alphaval = *alpha * a[iter];
					SCAL(&SCALN, &alphaval, &c[iter * M], &incx);
				}
			}
		}
	} else {
		INT AXPYN;
		memset((void *) c, 0, M * N * sizeof(DOUBLE));
		if (*side == 'L') {
			AXPYN = N;
			incx = M;
			if (*trans == 'M') {
				for (iter = 0; iter < M; ++iter) {
					alphaval = *alpha * a[iter * M + iter];
					AXPY(&AXPYN, &alphaval, &b[iter], &incx, &c[iter], &incx);
				}
			} else if (*trans == 'V') {
				for (iter = 0; iter < M; ++iter) {
					alphaval = *alpha * a[iter];
					AXPY(&AXPYN, &alphaval, &b[iter], &incx, &c[iter], &incx);
				}
			}
		} else if (*side == 'R') {
			AXPYN = M;
			incx = 1;
			if (*trans == 'M') {
				for (iter = 0; iter < N; ++iter) {
					alphaval = *alpha * a[iter * M + iter];
					AXPY(&AXPYN, &alphaval, &b[iter * M], &incx, &c[iter * M], &incx);
				}
			} else if (*trans == 'V') {
				for (iter = 0; iter < N; ++iter) {
					alphaval = *alpha * a[iter];
					AXPY(&AXPYN, &alphaval, &b[iter * M], &incx, &c[iter * M], &incx);
				}
			}
		}
	}
}

void diag(CHAR *trans, INT *m, INT *n, DOUBLE *a, DOUBLE *b) {

	INT M = *m;
	INT N = *n;
	INT iter;
	INT MINMN = IMIN(M, N);
	if (*trans == 'V') {
		for (iter = 0; iter < MINMN; ++iter) {
			b[iter] = a[iter * M + iter];
		}
	} else if (*trans == 'M') {
		memset((void *) a, 0, M * N * sizeof(DOUBLE));
		for (iter = 0; iter < MINMN; ++iter) {
			a[iter * M + iter] = b[iter];
		}
	}
}

DOUBLE trac(INT *m, INT *n, DOUBLE *a) {

	INT M = *m;
	INT N = *n;
	INT iter;
	INT MINMN = IMIN(M, N);
	DOUBLE traceValue = 0;
	for (iter = 0; iter < MINMN; ++iter) {
		traceValue += a[iter * M + iter];
	}

	return traceValue;
}

DOUBLE vsum(INT *n, DOUBLE *x, INT *incx) {

	DOUBLE sum = 0;
	INT iter;
	INT N = *n;
	INT INCX = *incx;

	for (iter = 0; iter < N; ++iter) {
		sum += x[iter * INCX];
	}

	return sum;
}

DOUBLE vprd(INT *n, DOUBLE *x, INT *incx) {

	DOUBLE product = 1;
	INT iter;
	INT N = *n;
	INT INCX = *incx;

	for (iter = 0; iter < N; ++iter) {
		product *= x[iter * INCX];
	}

	return product;
}

void geva(CHAR *trans, INT *n, INT *m, DOUBLE *alpha, DOUBLE *a, DOUBLE *b) {

	INT M = *m;
	INT N = *n;
	INT iter;
	INT incx = 1;
	INT incy;
	INT AXPYN;

	if (*trans == 'C') {
		AXPYN = M;
		incy = 1;
		for (iter = 0; iter < N; ++iter) {
			AXPY(&AXPYN, alpha, a, &incx, &b[iter * M], &incy);
		}
	} else if (*trans == 'R') {
		AXPYN = N;
		incy = M;
		for (iter = 0; iter < M; ++iter) {
			AXPY(&AXPYN, alpha, a, &incx, &b[iter], &incy);
		}
	}
}

void gevm(CHAR *trans, INT *m, INT *n, DOUBLE *alpha, DOUBLE *a, DOUBLE *b) {

	INT M = *m;
	INT N = *n;
	INT incx;
	INT iter;
	INT SCALN;
	DOUBLE alphaval;
	if (*trans == 'R') {
		SCALN = N;
		incx = M;
		for (iter = 0; iter < M; ++iter) {
			alphaval = *alpha * a[iter];
			SCAL(&SCALN, &alphaval, &b[iter], &incx);
		}
	} else if (*trans == 'C') {
		SCALN = M;
		incx = 1;
		for (iter = 0; iter < N; ++iter) {
			alphaval = *alpha * a[iter];
			SCAL(&SCALN, &alphaval, &b[iter * M], &incx);
		}
	}
}

void gesu(CHAR *trans, INT *n, INT *m, DOUBLE *alpha, DOUBLE *a, DOUBLE *beta, DOUBLE *b) {

	INT M = *m;
	INT N = *n;
	INT iter;
	INT incx;
	INT incy = 1;
	INT AXPYN;
	INT SCALN;

	if (*trans == 'C') {
		AXPYN = M;
		SCALN = M;
		incx = 1;
		if (*beta == 0) {
			memset((void *) b, 0, SCALN * sizeof(DOUBLE));
		} else {
			SCAL(&SCALN, beta, b, &incy);
		}
		for (iter = 0; iter < N; ++iter) {
			AXPY(&AXPYN, alpha, &a[iter * M], &incx, b, &incy);
		}
	} else if (*trans == 'R') {
		AXPYN = N;
		SCALN = N;
		incx = M;
		if (*beta == 0) {
			memset((void *) b, 0, SCALN * sizeof(DOUBLE));
		} else {
			SCAL(&SCALN, beta, b, &incy);
		}
		for (iter = 0; iter < M; ++iter) {
			AXPY(&AXPYN, alpha, &a[iter], &incx, b, &incy);
		}
	}
}
