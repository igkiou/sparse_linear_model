/*
 * usecuda.h
 *
 *  Created on: Nov 7, 2011
 *      Author: igkiou
 */

#ifndef __USE_CUDA_H__
#define __USE_CUDA_H__

#include <stdio.h>

#include "cublas_v2.h"
#include "cula_lapack_device.h"
#include "cuda_runtime.h"
#include "useinterfaces.h"

/* TODO: Change #define's to typedefs and consts, or other alternative (inline)
 * for function names.
 * TODO: Link MKL with ARPACK and PROPACK
 */

#ifdef USE_DOUBLE_PRECISION
typedef double CUDOUBLE;
#elif defined(USE_SINGLE_PRECISION)
typedef float CUDOUBLE;
#endif
typedef char CUCHAR;
typedef int CUINT;
typedef cublasHandle_t CUHANDLE;
typedef cublasStatus_t CUSTATUS;

#ifdef USE_DOUBLE_PRECISION
#define CUDAFUNC2PART(NAME1,NAME2) NAME1 ## D ## NAME2
#elif defined(USE_SINGLE_PRECISION)
#define CUDAFUNC2PART(NAME1,NAME2) NAME1 ## S ## NAME2
#endif

/* Declarations from interfaces. */
extern void ERROR(const char *error_msg);

inline void cublasInitialize(CUHANDLE *handle) {
	CUSTATUS status = cublasCreate(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		ERROR("CUBLAS: initialization error.\n" ) ;
	}
}

inline void cublasShutdown(CUHANDLE handle) {
	cublasDestroy(handle);
}

/* Copy a data vector from device to device memory. */
inline void cud2dcpy(CUHANDLE handle, void *target, void *source, CUINT size) {

	if (cudaMemcpy(target, source, size * sizeof(CUDOUBLE), \
			cudaMemcpyDeviceToDevice) != cudaSuccess) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: device to device memory copy error.\n");
	}
}

/* Copy a data vector from host to device memory. */
inline void cuh2dcpy(CUHANDLE handle, void *target, void *source, CUINT size) {
	CUSTATUS status = cublasSetVector(size, sizeof(CUDOUBLE), source, 1, \
										target, 1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: device write error.\n");
	}
}

/* Copy a data vector from device to host memory. */
inline void cud2hcpy(CUHANDLE handle, void *target, void *source, CUINT size) {
	CUSTATUS status = cublasGetVector(size, sizeof(CUDOUBLE), source, 1, \
										target, 1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: device read error.\n");
	}
}

/* Allocate memory in device. */
inline void cumalloc(CUHANDLE handle, void **target, size_t size) {

	if (cudaMalloc(target, size) != cudaSuccess) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: device memory allocation error.\n");
	}
}

/* Free memory in device. */
inline void cufree(CUHANDLE handle, void *target) {

	if (cudaFree(target) != cudaSuccess) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: device memory free error.\n");
	}
}

/* Set memory to value. */
inline void cumemset(CUHANDLE handle, void *target, CUINT value, size_t size) {

	if (cudaMemset(target, value, size) != cudaSuccess) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: device memory set error.\n");
	}
}

/* Check CULA status. */
inline void culaCheckStatus(CUHANDLE handle, culaStatus status) {
    if(!status)
        return;

    char buf[256];
    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    fprintf(stderr, "%s\n", buf);
	culaShutdown();
	cublasShutdown(handle);
    ERROR("CULA error.\n");
}

inline cublasOperation_t cublasGetOperation(CUCHAR trans) {
	if ((trans == 'N') || (trans == 'n')) {
		return CUBLAS_OP_N;
	} else if ((trans == 'T') || (trans == 't')) {
		return CUBLAS_OP_T;
	} else if ((trans == 'C') || (trans == 'c')) {
		return CUBLAS_OP_C;
	}
	return CUBLAS_OP_N;
}

inline cublasFillMode_t cublasGetFill(CUCHAR uplo) {
	if ((uplo == 'U') || (uplo == 'u')) {
		return CUBLAS_FILL_MODE_UPPER;
	} else if ((uplo == 'L') || (uplo == 'l')) {
		return CUBLAS_FILL_MODE_LOWER;
	}
	return CUBLAS_FILL_MODE_UPPER;
}

inline cublasSideMode_t cublasGetSide(CUCHAR side) {
	if ((side == 'R') || (side == 'r')) {
		return CUBLAS_SIDE_RIGHT;
	} else if ((side == 'L') || (side == 'l')) {
		return CUBLAS_SIDE_LEFT;
	}
	return CUBLAS_SIDE_RIGHT;
}

inline void cublasSetPointerModeDevice(CUHANDLE handle) {
	CUSTATUS status = cublasSetPointerMode(handle, \
											CUBLAS_POINTER_MODE_DEVICE);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: pointer mode error.\n");
	}
}

inline void cublasSetPointerModeHost(CUHANDLE handle) {
	CUSTATUS status = cublasSetPointerMode(handle, \
											CUBLAS_POINTER_MODE_HOST);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: pointer mode error.\n");
	}
}

/* MKL functions */
inline void CUASUM(CUHANDLE handle, CUINT n, CUDOUBLE *x, CUINT incx, \
				CUDOUBLE *result) {
	CUSTATUS status = CUDAFUNC2PART(cublas, asum)(handle, n, x, incx, result);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: CUASUM error.\n");
	}
}

inline void CUAXPBY(CUHANDLE handle, CUINT n, CUDOUBLE *alpha,
			CUDOUBLE *x, CUINT incx, CUDOUBLE *beta, CUDOUBLE *y, CUINT incy) {
	CUDAFUNC2PART(cublas, scal)(handle, n, beta, y, incx);
	CUDAFUNC2PART(cublas, axpy)(handle, n, alpha, x, incx, y, incy);

}

inline void CUAXPY(CUHANDLE handle, CUINT n, CUDOUBLE *alpha, CUDOUBLE *x, \
				CUINT incx, CUDOUBLE *y, CUINT incy) {
	CUSTATUS status = cublasDaxpy(handle, n, alpha, x, incx, y, incy);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: CUAXPY error.\n");
	}
}

inline void CUGEMM(CUHANDLE handle, CUCHAR transa, CUCHAR transb, CUINT m, \
				CUINT n, CUINT k, CUDOUBLE *alpha, CUDOUBLE *A, CUINT lda, \
				CUDOUBLE*B, CUINT ldb, CUDOUBLE *beta, CUDOUBLE *C, CUINT ldc) {
	CUSTATUS status = CUDAFUNC2PART(cublas, gemm)(handle, \
								cublasGetOperation(transa), \
								cublasGetOperation(transb), m, n, k, alpha, A, \
								lda, B, ldb, beta, C, ldc);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: CUGEMM error.\n");
	}
}

inline void CUGESVD(CUHANDLE handle, CUCHAR jobu, CUCHAR jobvt, CUINT m, \
					CUINT n, CUDOUBLE* a, CUINT lda, CUDOUBLE* s, CUDOUBLE* u, \
					CUINT ldu, CUDOUBLE* vt, CUINT ldvt) {
	culaStatus status = CUDAFUNC2PART(culaDevice, gesvd)(jobu, jobvt, m, n, a, \
									lda, s, u, ldu, vt, ldvt);
	culaCheckStatus(handle, status);
}

inline void CUNRM2(CUHANDLE handle, CUINT n, CUDOUBLE *x, CUINT incx, \
				CUDOUBLE *result) {
	CUSTATUS status = CUDAFUNC2PART(cublas, nrm2)(handle, n, x, incx, result);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: CUNRM2 error.\n");
	}
}

inline void CUSCAL(CUHANDLE handle, CUINT n, CUDOUBLE *alpha, CUDOUBLE *x, \
				CUINT incx) {
	CUSTATUS status = CUDAFUNC2PART(cublas, scal)(handle, n, alpha, x, incx);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: CUSCAL error.\n");
	}
}

inline void CUSYMM(CUHANDLE handle, CUCHAR side, CUCHAR uplo, CUINT m, \
				CUINT n, CUDOUBLE *alpha, CUDOUBLE *A, CUINT lda, CUDOUBLE*B, \
				CUINT ldb, CUDOUBLE *beta, CUDOUBLE *C, CUINT ldc) {
	CUSTATUS status = CUDAFUNC2PART(cublas, symm)(handle, cublasGetSide(side), \
												cublasGetFill(uplo), m, n, \
												alpha, A, lda, B, ldb, beta, C, \
												ldc);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: CUSYMM error.\n");
	}
}

inline void CUSYRK(CUHANDLE handle, CUCHAR uplo, CUCHAR trans, CUINT n, CUINT k, \
				CUDOUBLE *alpha, CUDOUBLE *A, CUINT lda, CUDOUBLE *beta, \
				CUDOUBLE *C, CUINT ldc) {
	CUSTATUS status = CUDAFUNC2PART(cublas, syrk)(handle, cublasGetFill(uplo), \
												cublasGetOperation(trans), n, k, \
												alpha, A, lda, beta, C, ldc);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		cublasShutdown(handle);
		ERROR("CUDA: CUSYRK error.\n");
	}
}

#endif /* __USE_CUDA_H__ */
