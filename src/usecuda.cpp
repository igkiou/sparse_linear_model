/*
 * usecuda.c
 *
 *  Created on: Nov 7, 2011
 *      Author: igkiou
 */

#include "usecuda.h"
#include "useinterfaces.h"

///* Copy a data vector from device to device memory. */
//inline void cud2dcpy(CUDOUBLE *target, CUDOUBLE *source, CUINT size) {
//	CUINT incx = 1;
//	CUCOPY(size, source, incx, target, incx);
//}
//
///* Copy a data vector from host to device memory. */
//inline void cuh2dcpy(CUDOUBLE *target, CUDOUBLE *source, CUINT size) {
//	cublasStatus status = cublasSetVector(size, sizeof(source[0]), source, 1, \
//										target, 1);
//
//	if (status != CUBLAS_STATUS_SUCCESS) {
//		cublasShutdown();
//		culaShutdown();
//		ERROR("CUDA: device write error.\n");
//	}
//}
//
///* Copy a data vector from device to host memory. */
//inline void cud2hcpy(CUDOUBLE *target, CUDOUBLE *source, CUINT size) {
//	cublasStatus status = cublasGetVector(size, sizeof(target[0]), source, 1, \
//										target, 1);
//
//	if (status != CUBLAS_STATUS_SUCCESS) {
//		cublasShutdown();
//		culaShutdown();
//		ERROR("CUDA: device read error.\n");
//	}
//}
//
///* Allocate memory in device. */
//inline void cumalloc(void *target, size_t size) {
//
//	if (cudaMalloc((void**)&target, size) != cudaSuccess) {
//		cublasShutdown();
//		culaShutdown();
//		ERROR("CUDA: device memory allocation error.\n");
//	}
//
//}
//
///* Free memory in device. */
//inline void cufree(void *target) {
//
//	if (cudaFree(target) != cudaSuccess) {
//		cublasShutdown();
//		culaShutdown();
//		ERROR("CUDA: device memory free error.\n");
//	}
//}
//
///* Set memory to value. */
//inline void cumemset(void *target, CUINT value, size_t size) {
//
//	if (cudaMemset(target, value, size) != cudaSuccess) {
//		cublasShutdown();
//		culaShutdown();
//		ERROR("CUDA: device memory set error.\n");
//	}
//
//}
//
///* Check CULA status. */
//inline void checkCulaStatus(culaStatus status) {
//    if(!status)
//        return;
//
//    char buf[256];
//    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
//    printf("%s\n", buf);
//	cublasShutdown();
//    culaShutdown();
//    ERROR("CULA error.\n");
//}
