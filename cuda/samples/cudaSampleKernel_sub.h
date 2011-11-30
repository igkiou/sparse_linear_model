/*
 * cudaSampleKernel_sub.h
 *
 *  Created on: Nov 9, 2011
 *      Author: igkiou
 */

#ifndef CUDASAMPLEKERNEL_SUB_H_
#define CUDASAMPLEKERNEL_SUB_H_

#include <stdio.h>

#include "cublas_v2.h"
#include "cula_lapack_device.h"
#include "cuda_runtime.h"

/* TODO: Change #define's to typedefs and consts, or other alternative (inline)
 * for function names.
 * TODO: Link MKL with ARPACK and PROPACK
 */

typedef float CUDOUBLE;
typedef int CUINT;
typedef char CUCHAR;
#define mxPRECISION_CLASS mxSINGLE_CLASS
#define MXPRECISION_CLASS mxSINGLE_CLASS
#define CUDAFUNC2PART(NAME1,NAME2) NAME1 ## S ## NAME2

/* MKL functions */
#define CUASUM CUDAFUNC2PART(cublas, asum)
#define CUAXPBY CUDAFUNC2PART(cublas, axpby)
#define CUAXPY CUDAFUNC2PART(cublas, axpy)
#define CUCOPY CUDAFUNC2PART(cublas, copy)
#define CUGEMM CUDAFUNC2PART(cublas, gemm)
#define CUGESVD CUDAFUNC2PART(culaDevice, gesvd)
#define CUNRM2 CUDAFUNC2PART(cublas, nrm2)
#define CUSCAL CUDAFUNC2PART(cublas, scal)
#define CUSYMM CUDAFUNC2PART(cublas, symm)

inline void cublasInitialize(cublasHandle_t *handle) {
	cublasStatus_t status = cublasCreate(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		fprintf(stderr, "CUBLAS: initialization error.\n" ) ;
	}
}

/* Copy a data vector from device to device memory. */
inline void cud2dcpy(void *target, void *source, CUINT size) {

	if (cudaMemcpy(target, source, size * sizeof(CUDOUBLE), \
			cudaMemcpyDeviceToDevice) != cudaSuccess);
	culaShutdown();
	fprintf(stderr, "CUDA: device to device memory copy error.\n");
}

/* Copy a data vector from host to device memory. */
inline void cuh2dcpy(void *target, void *source, CUINT size) {
	cublasStatus_t status = cublasSetVector(size, sizeof(CUDOUBLE), source, 1, \
										target, 1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		fprintf(stderr, "CUDA: device write error.\n");
	}
}

/* Copy a data vector from device to host memory. */
inline void cud2hcpy(void *target, void *source, CUINT size) {
	cublasStatus_t status = cublasGetVector(size, sizeof(CUDOUBLE), source, 1, \
										target, 1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		fprintf(stderr, "CUDA: device read error.\n");
	}
}

/* Allocate memory in device. */
inline void cumalloc(void **target, size_t size) {

	if (cudaMalloc(target, size) != cudaSuccess) {
		culaShutdown();
		fprintf(stderr, "CUDA: device memory allocation error.\n");
	}
}

/* Free memory in device. */
inline void cufree(void *target) {

	if (cudaFree(target) != cudaSuccess) {
		culaShutdown();
		fprintf(stderr, "CUDA: device memory free error.\n");
	}
}

/* Set memory to value. */
inline void cumemset(void *target, CUINT value, size_t size) {

	if (cudaMemset(target, value, size) != cudaSuccess) {
		culaShutdown();
		fprintf(stderr, "CUDA: device memory set error.\n");
	}
}

/* Check CULA status. */
inline void checkCulaStatus(culaStatus status) {
    if(!status)
        return;

    char buf[256];
    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);
    culaShutdown();
    fprintf(stderr, "CULA error.\n");
}

inline cublasOperation_t getCublasOperation(CUCHAR trans) {
	if ((trans == 'N') || (trans == 'n')) {
		return CUBLAS_OP_N;
	} else if ((trans == 'T') || (trans == 't')) {
		return CUBLAS_OP_T;
	} else if ((trans == 'C') || (trans == 'c')) {
		return CUBLAS_OP_C;
	}
	return CUBLAS_OP_N;
}

inline cublasFillMode_t getCublasFill(CUCHAR uplo) {
	if ((uplo == 'U') || (uplo == 'u')) {
		return CUBLAS_FILL_MODE_UPPER;
	} else if ((uplo == 'L') || (uplo == 'l')) {
		return CUBLAS_FILL_MODE_LOWER;
	}
	return CUBLAS_FILL_MODE_UPPER;
}

inline cublasSideMode_t getCublasSide(CUCHAR side) {
	if ((side == 'R') || (side == 'r')) {
		return CUBLAS_SIDE_RIGHT;
	} else if ((side == 'L') || (side == 'l')) {
		return CUBLAS_SIDE_LEFT;
	}
	return CUBLAS_SIDE_RIGHT;
}

inline void cublasSetPointerModeDevice(cublasHandle_t handle) {
	cublasStatus_t status = cublasSetPointerMode(handle, \
											CUBLAS_POINTER_MODE_DEVICE);

	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		fprintf(stderr, "CUDA: pointer mode error.\n");
	}
}

inline void cublasSetPointerModeHost(cublasHandle_t handle) {
	cublasStatus_t status = cublasSetPointerMode(handle, \
											CUBLAS_POINTER_MODE_HOST);

	if (status != CUBLAS_STATUS_SUCCESS) {
		culaShutdown();
		fprintf(stderr, "CUDA: pointer mode error.\n");
	}
}

void cuSoftThresholdKernel(CUDOUBLE *sv, CUDOUBLE tau, CUINT N);

#endif /* CUDASAMPLEKERNEL_SUB_H_ */
