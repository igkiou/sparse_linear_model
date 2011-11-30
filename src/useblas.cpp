/*
 * TODO: Parallelize using openMP, critical, careful if inside other
 * parrallel code (another reason to use MKL's implementation.
 */

#include "useblas.h"

void transpose(DOUBLE *X, DOUBLE *Y, INT N, INT M)
{
	INT i, j, iterM, iterN;
  
	if (N < M) {
		for (j = 0; j < M; ++j) {
			iterN = j * N;
			for (i = 0; i < N; ++i) {
				Y[j + i * M] = X[i + iterN];
			}
		}
	}
	else {
		for (i = 0; i < N; ++i) {
			iterM = i * M;
			for (j = 0; j < M; ++j) {
				Y[j + iterM] = X[i + j * N];
			}
		}
	}
}

void transpose_inplace(DOUBLE *X, INT N) {

	INT i, j;
	DOUBLE tmp;
	for (i = 0; i < N; ++i) {
		for (j = i + 1; j < N; ++j) {
			tmp = X[i * N + j];
			X[i * N + j] = X[j * N + i];
			X[j * N + i] = tmp;
		}
	}
}
