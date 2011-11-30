/*
 * cudaSampleKernel.cu
 *
 *  Created on: Nov 7, 2011
 *      Author: igkiou
 */

#include "cudaSampleKernel_sub.h"

__global__ void cuSoftThresholdKernel_sub(CUDOUBLE *sv, CUDOUBLE tau, CUINT N) {
	CUINT iterN = blockDim.x * blockIdx.x + threadIdx.x;
	// for (iterN = 0; iterN < N; ++iterN) {
	if (iterN < N) {
		sv[iterN] = sv[iterN] - tau;
		if (sv[iterN] < 0) {
			sv[iterN] = 0;
		}
	}
}

void cuSoftThresholdKernel(CUDOUBLE *sv, CUDOUBLE tau, CUINT N) {
	CUINT threadsPerBlock = 8;
	CUINT blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	cuSoftThresholdKernel_sub<<<blocksPerGrid, threadsPerBlock>>>(sv, tau, N);
}






//#include <stdio.h>
//
//#define CHECK_BANK_CONFLICTS 0
//#if CHECK_BANK_CONFLICTS
//#define AS(i, j) cutilBankChecker(((DOUBLE*)&As[0][0]), (BLOCK_SIZE * i + j))
//#define BS(i, j) cutilBankChecker(((DOUBLE*)&Bs[0][0]), (BLOCK_SIZE * i + j))
//#else
//#define AS(i, j) As[i][j]
//#define BS(i, j) Bs[i][j]
//#endif
//
//typedef float DOUBLE;
//#define BLOCK_SIZE 16
//
//////////////////////////////////////////////////////////////////////////////////
////! Matrix multiplication on the device: C = A * B
////! wA is A's width and wB is B's width
//////////////////////////////////////////////////////////////////////////////////
// __global__ void matrixMul( DOUBLE* C, DOUBLE* A, DOUBLE* B, int wA, int wB) {
//    // Block index
//    int bx = blockIdx.x;
//    int by = blockIdx.y;
//
//    // Thread index
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//
//    // Index of the first sub-matrix of A processed by the block
//    int aBegin = wA * BLOCK_SIZE * by;
//
//    // Index of the last sub-matrix of A processed by the block
//    int aEnd   = aBegin + wA - 1;
//
//    // Step size used to iterate through the sub-matrices of A
//    int aStep  = BLOCK_SIZE;
//
//    // Index of the first sub-matrix of B processed by the block
//    int bBegin = BLOCK_SIZE * bx;
//
//    // Step size used to iterate through the sub-matrices of B
//    int bStep  = BLOCK_SIZE * wB;
//
//    // Csub is used to store the element of the block sub-matrix
//    // that is computed by the thread
//    DOUBLE Csub = 0;
//
//    // Loop over all the sub-matrices of A and B
//    // required to compute the block sub-matrix
//    for (int a = aBegin, b = bBegin;
//             a <= aEnd;
//             a += aStep, b += bStep) {
//
//        // Declaration of the shared memory array As used to
//        // store the sub-matrix of A
//        __shared__ DOUBLE As[BLOCK_SIZE][BLOCK_SIZE];
//
//        // Declaration of the shared memory array Bs used to
//        // store the sub-matrix of B
//        __shared__ DOUBLE Bs[BLOCK_SIZE][BLOCK_SIZE];
//
//        // Load the matrices from device memory
//        // to shared memory; each thread loads
//        // one element of each matrix
//        AS(ty, tx) = A[a + wA * ty + tx];
//        BS(ty, tx) = B[b + wB * ty + tx];
//
//        // Synchronize to make sure the matrices are loaded
//        __syncthreads();
//
//        // Multiply the two matrices together;
//        // each thread computes one element
//        // of the block sub-matrix
//        for (int k = 0; k < BLOCK_SIZE; ++k)
//            Csub += AS(ty, k) * BS(k, tx);
//
//        // Synchronize to make sure that the preceding
//        // computation is done before loading two new
//        // sub-matrices of A and B in the next iteration
//        __syncthreads();
//    }
//
//    // Write the block sub-matrix to device memory;
//    // each thread writes one element
//    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
//    C[c + wB * ty + tx] = Csub;
//}
//
// void matrixMulWrap( DOUBLE* C, DOUBLE* A, DOUBLE* B, int wA, int wB) {
//	 matrixMul<<< 8, 8 >>>( C, A, B, wA, wB);
// }
