#ifndef __USE_BLAS_H__
#define __USE_BLAS_H__

/* Options:
 * #define __DEBUG__: use debugging versions
 * #define USE_CUSTOM_BLAS: use custom versions of some BLAS functions
 * #define __USE_GSL__: use GSL variant of custom versions of BLAS functions
 * 						(no effect if USE_CUSTOM_BLAS is not defined)
 * #define USE_DOUBLE_PRECISION: use double precision versions
 * #define USE_SINGLE_PRECISION: use single precision versions (if both double
 * 								 and single precision flags are defined, double
 * 								 overrides)
 * #define __USE_PARALLEL__: use threaded versions of some functions
 */
#define __USE_PARALLEL__

#include <math.h>
#include <stdio.h>

#include <mkl.h>
#include <omp.h>

/* TODO: Link MKL with ARPACK and PROPACK
 * TODO: Enforce const-ness.
 * TODO: Enforce uniform check of obj = NULL, or grad = NULL, or gradFlag in
 * obj_grad functions.
 * TODO: Enforce uniform check for initialization in functions needing it.
 */

#ifdef USE_DOUBLE_PRECISION
	typedef double DOUBLE;
#elif defined(USE_SINGLE_PRECISION)
	typedef float DOUBLE;
#endif
typedef MKL_INT INT;
typedef char CHAR;

#ifdef USE_DOUBLE_PRECISION
	#define MKLFUNC(NAME) d ## NAME
	#define MKLFUNC2PART(NAME1,NAME2) NAME1 ## d ## NAME2
	#define MATHFUNC(NAME) NAME
#elif defined(USE_SINGLE_PRECISION)
	#define MKLFUNC(NAME) s ## NAME
	#define MKLFUNC2PART(NAME1,NAME2) NAME1 ## s ## NAME2
	#define MATHFUNC(NAME) NAME ## f
#endif

/* Declarations from interfaces. */
extern void ERROR(const char *error_msg);
extern void *MALLOC(size_t n);
extern void *CMALLOC(size_t n);
extern void FREE(void *ptr);
extern void CFREE(void *ptr);
extern bool ISINF(double x);
extern bool ISNAN(double x);

/* MKL functions */
inline DOUBLE ASUM(INT *n, DOUBLE *x, INT *incx) {
	return MKLFUNC(asum)(n, x, incx);
}

inline void AXPBY(INT *n, DOUBLE *alpha, DOUBLE *x, INT *incx, DOUBLE *beta, \
				DOUBLE *y, INT *incy) {
	MKLFUNC(axpby)(n, alpha, x, incx, beta, y, incy);
}

inline void AXPY(INT *n, DOUBLE *alpha, DOUBLE *x, INT *incx, DOUBLE *y, \
				INT *incy) {
	MKLFUNC(axpy)(n, alpha, x, incx, y, incy);
}

inline void COPY(INT *n, DOUBLE *x, INT *incx, DOUBLE *y, INT *incy) {
	MKLFUNC(copy)(n, x, incx, y, incy);
}

inline DOUBLE DOT(INT *n, DOUBLE *x, INT *incx, DOUBLE *y, INT *incy) {
	return MKLFUNC(dot)(n, x, incx, y, incy);
}

inline void GELSY(INT* m, INT* n, INT* nrhs, DOUBLE* a, INT* lda, DOUBLE* b, \
				INT* ldb, INT* jpvt, DOUBLE* rcond, INT* rank, DOUBLE* work, \
				INT* lwork, INT* info) {
	MKLFUNC(gelsy)(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank, work, lwork, \
					info);
	if ((*info) != 0) {
		fprintf(stderr, "Error, INFO = %lld. ", *info);
		ERROR("LAPACK GELSY error.");
	}
}

inline void GEMM(CHAR *transa, CHAR *transb, INT *m, INT *n, INT *k, \
				DOUBLE *alpha, DOUBLE *a, INT *lda, DOUBLE *b, INT *ldb, \
				DOUBLE *beta, DOUBLE *c, INT *ldc) {
	MKLFUNC(gemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void GEMV(CHAR *trans, INT *m, INT *n, DOUBLE *alpha, DOUBLE *a, \
				INT *lda, DOUBLE *x, INT *incx, DOUBLE *beta, DOUBLE *y, \
				INT *incy) {
	MKLFUNC(gemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void GER(INT *m, INT *n, DOUBLE *alpha, DOUBLE *x, INT *incx, DOUBLE *y, \
				INT *incy, DOUBLE *a, INT *lda) {
	MKLFUNC(ger)(m, n, alpha, x, incx, y, incy, a, lda);
}

inline void GESDD(CHAR* jobz, INT* m, INT* n, DOUBLE* a, INT* lda, DOUBLE* s, \
				DOUBLE* u, INT* ldu, DOUBLE* vt, INT* ldvt, DOUBLE* work, \
				INT* lwork, INT* iwork, INT* info) {
	MKLFUNC(gesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, \
				info);
	if ((*info) != 0) {
		fprintf(stderr, "Error, INFO = %lld. ", *info);
		ERROR("LAPACK GESDD error.");
	}
}

inline void GESVD(CHAR* jobu, CHAR* jobvt, INT* m, INT* n, DOUBLE* a, INT* lda, \
				DOUBLE* s, DOUBLE* u, INT* ldu, DOUBLE* vt, INT* ldvt, \
				DOUBLE* work, INT* lwork, INT* info) {
	MKLFUNC(gesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, \
				info);
	if ((*info) != 0) {
		fprintf(stderr, "Error, INFO = %lld. ", *info);
		ERROR("LAPACK GESVD error.");
	}
}

inline INT IAMAX(INT *n, DOUBLE *x, INT *incx) {
	return MKLFUNC2PART(i,amax)(n, x, incx);
}

inline void LACPY(CHAR* uplo, INT* m, INT* n, DOUBLE* a, INT* lda, DOUBLE* b, \
				INT* ldb) {
	MKLFUNC(lacpy)(uplo, m, n, a, lda, b, ldb);
}

inline DOUBLE LAMCH(CHAR* cmach) {
	return MKLFUNC(lamch)(cmach);
}

inline DOUBLE LANGE(CHAR* norm, INT* m, INT* n, DOUBLE* a, INT* lda, \
					DOUBLE* work) {
	return MKLFUNC(lange)(norm, m, n, a, lda, work);
}

inline DOUBLE LANSY(CHAR* norm, CHAR* uplo, INT* n, DOUBLE* a, INT* lda, \
					DOUBLE* work) {
	return MKLFUNC(lansy)(norm, uplo, n, a, lda, work);
}

inline DOUBLE NRM2(INT *n, DOUBLE *x, INT *incx) {
	return MKLFUNC(nrm2)(n, x, incx);
}

inline void POTRF(CHAR* uplo, INT* n, DOUBLE* a, INT* lda, INT* info) {
	MKLFUNC(potrf)(uplo, n, a, lda, info);
	if ((*info) != 0) {
		fprintf(stderr, "Error, INFO = %lld. ", *info);
		ERROR("LAPACK POTRF error.");
	}
}

inline void POTRS(CHAR* uplo, INT* n, INT* nrhs, DOUBLE* a, INT* lda, DOUBLE* b, \
				INT* ldb, INT* info) {
	MKLFUNC(potrs)(uplo, n, nrhs, a, lda, b, ldb, info);
	if ((*info) != 0) {
		fprintf(stderr, "Error, INFO = %lld. ", *info);
		ERROR("LAPACK POTRS error.");
	}
}

inline void SCAL(INT *n, DOUBLE *a, DOUBLE *x, INT *incx) {
	MKLFUNC(scal)(n, a, x, incx);
}

inline void SYEV(CHAR* jobz, CHAR* uplo, INT* n, DOUBLE* a, INT* lda, DOUBLE* w, \
				DOUBLE* work, INT* lwork, INT* info) {
	MKLFUNC(syev)(jobz, uplo, n, a, lda, w, work, lwork, info);
	if ((*info) != 0) {
		fprintf(stderr, "Error, INFO = %lld. ", *info);
		ERROR("LAPACK SYEV error.");
	}
}

inline void SYEVR(CHAR* jobz, CHAR* range, CHAR* uplo, INT* n, DOUBLE* a, \
				INT* lda, DOUBLE* vl, DOUBLE* vu, INT* il, INT* iu, \
				DOUBLE* abstol, INT* m, DOUBLE* w, DOUBLE* z, INT* ldz, \
				INT* isuppz, DOUBLE* work, INT* lwork, INT* iwork, INT* liwork, \
				INT* info) {
	MKLFUNC(syevr)(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, \
					z, ldz, isuppz, work, lwork, iwork, liwork, info);
	if ((*info) != 0) {
		fprintf(stderr, "Error, INFO = %lld. ", *info);
		ERROR("LAPACK SYEVR error.");
	}
}

inline void SYMM(CHAR *side, CHAR *uplo, INT *m, INT *n, DOUBLE *alpha, \
				DOUBLE *a, INT *lda, DOUBLE *b, INT *ldb, DOUBLE *beta, \
				DOUBLE *c, INT *ldc) {
	MKLFUNC(symm)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void SYMV(CHAR *uplo, INT *n, DOUBLE *alpha, DOUBLE *a, INT *lda, \
				DOUBLE *x, INT *incx, DOUBLE *beta, DOUBLE *y, INT *incy) {
	MKLFUNC(symv)(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void SYR(CHAR *uplo, INT *n, DOUBLE *alpha, DOUBLE *x, INT *incx, \
				DOUBLE *a, INT *lda) {
	MKLFUNC(syr)(uplo, n, alpha, x, incx, a, lda);
}

inline void SYR2(CHAR *uplo, INT *n, DOUBLE *alpha, DOUBLE *x, INT *incx, \
				DOUBLE *y, INT *incy, DOUBLE *a, INT *lda) {
	MKLFUNC(syr2)(uplo, n, alpha, x, incx, y, incy, a, lda);
}

inline void SYRK(CHAR *uplo, CHAR *trans, INT *n, INT *k, DOUBLE *alpha, \
				DOUBLE *a, INT *lda, DOUBLE *beta, DOUBLE *c, INT *ldc) {
	MKLFUNC(syrk)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void TRMM(CHAR *side, CHAR *uplo, CHAR *transa, CHAR *diag, INT *m, \
				INT *n, DOUBLE *alpha, DOUBLE *a, INT *lda, DOUBLE *b, \
				INT *ldb) {
	MKLFUNC(trmm)(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void TRSV(CHAR *uplo, CHAR *trans, CHAR *diag, INT *n, DOUBLE *a, \
				INT *lda, DOUBLE *x, INT *incx) {
	MKLFUNC(trsv)(uplo, trans, diag, n, a, lda, x, incx);
}

/* math.h functions */
inline DOUBLE EXP(DOUBLE x) {
	return MATHFUNC(exp)(x);
}

inline DOUBLE LOG(DOUBLE x) {
	return MATHFUNC(log)(x);
}

inline DOUBLE POW(DOUBLE x, DOUBLE e) {
	return MATHFUNC(pow)(x, e);
}

inline DOUBLE SQRT(DOUBLE x) {
	return MATHFUNC(sqrt)(x);
}

inline DOUBLE TANH(DOUBLE x) {
	return MATHFUNC(tanh)(x);
}

/* some math defines */
#define SQR(X)  ((X) * (X))
#define SIGN(X)  ((X) > 0 ? (1) : (((X) < 0 ? (-(1)) : (0))))
#define ABS(X)  ((X) > 0 ? (X) : (-(X)))
#define IMIN(X, Y)  ((X) < (Y) ? (X) : (Y))
#define IMAX(X, Y)  ((X) > (Y) ? (X) : (Y))

/* Various constants */
#define CUSTOM_OMP_NUM_THREADS 16
#define CUSTOM_MKL_NUM_THREADS 16

/**************************************************************************
 * Matrix transpose.
 *
 * Computes Y := X'
 *
 * Parameters:
 *   X - input matrix of size n X m
 *   Y - output matrix of size m X n
 *   n, m - dimensions of X
 *
 **************************************************************************/
void transpose(DOUBLE *X, DOUBLE *Y, INT N, INT M);

/* Note: slow, very naive implementation. */
void transpose_inplace(DOUBLE *X, INT N);

/* Copy a data vector. */
inline void datacpy(DOUBLE *target, DOUBLE *source, INT size) {
	INT incx = 1;
	COPY(&size, source, &incx, target, &incx);
}

/* Copy a matrix. May be faster for symmetric or other structured. */
inline void matcpy(DOUBLE *target, DOUBLE *source, INT M, INT N, CHAR part) {
	INT LDA = M;
	INT LDB = M;
	LACPY(&part, &M, &N, source, &LDA, target, &LDB);
}

#endif /* __USE_BLAS_H__ */
