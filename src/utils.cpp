#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "utils.h"

void print_matrix(DOUBLE *matrix, INT M, INT N) {
	
	INT iterM, iterN;
	for (iterM = 0; iterM < M; ++iterM) {
		for (iterN = 0; iterN < N; ++iterN) {
			PRINTF("%g ", matrix[iterN * M + iterM]);
		}
		PRINTF("\n");
	}
}

void print_matrix_int(INT *matrix, INT M, INT N) {
	
	INT iterM, iterN;
	for (iterM = 0; iterM < M; ++iterM) {
		for (iterN = 0; iterN < N; ++iterN) {
			PRINTF("%d ", matrix[iterN * M + iterM]);
		}
		PRINTF("\n");
	}
}

void rand_normal_custom(DOUBLE *r1, DOUBLE *r2, DOUBLE std) {
	
	DOUBLE x, y, r, bm;
	
	do {
		/* choose x, y in uniform square (-1,-1) to (+1,+1) */
		x = -1 + 2 * rand() / (DOUBLE) RAND_MAX;
		y = -1 + 2 * rand() / (DOUBLE) RAND_MAX;

		/* see if it is in the unit circle */
		r = x * x + y * y;
	}
	while (r > 1.0 || r == 0);

	/* Box-Muller transform */
	if ((r1 != NULL) || (r2 != NULL)) {
		bm = std * SQRT(((DOUBLE) -2.0) * LOG(r) / r);
		if (r1 != NULL) {
			*r1 = x * bm;
		}
		if (r2 != NULL) {
			*r2 = y * bm;
		}
	}
}

void quicksort(DOUBLE* data, INT *indices, INT N) {
	INT i, j, tint;
	DOUBLE v, t;
 
	if( N <= 1 )
		return;
 
	v = data[0];
	i = 0;
	j = N;
	for(;;)
	{
		while(data[++i] < v && i < N) { }
		while(data[--j] > v) { }
		if(i >= j) {
			break;
		}
		t = data[i];
		data[i] = data[j];
		data[j] = t;
		tint = indices[i];
		indices[i] = indices[j];
		indices[j] = tint;
	}
	t = data[i - 1];
	data[i - 1] = data[0];
	data[0] = t;
	tint = indices[i - 1];
	indices[i - 1] = indices[0];
	indices[0] = tint;
	quicksort(data, indices, i - 1);
	quicksort(data + i, indices + i, N - i);
}

// Old implementation
//void randperm(INT *perm, INT N, DOUBLE *buffer) {
//
//	INT bufferFlag = 0;
//	if (buffer == NULL) {
//		buffer = (DOUBLE *) MALLOC(N * sizeof(DOUBLE));
//		bufferFlag = 1;
//	}
//
//	INT iterX;
//	for (iterX = 0; iterX < N; ++iterX) {
//		buffer[iterX] = rand() / (DOUBLE) RAND_MAX;
//		perm[iterX] = iterX;
//	}
//	quicksort(buffer, perm, N);
//
//	if (bufferFlag == 1) {
//		FREE(buffer);
//	}
//}

// GSL implementation, based on Knuth's shuffle algorithm.
void randperm(INT *perm, INT N) {
	
	INT iterX;
	INT randInt;
	INT temp;
	for (iterX = 0; iterX < N; ++iterX) {
		perm[iterX] = iterX;
	}
	for (iterX = N - 1; iterX > 0; --iterX) {
		// NOTE: Maybe replace this with rand % N?
		randInt = (INT) floor((rand() / (DOUBLE) RAND_MAX) * N);
		temp = perm[iterX];
		perm[iterX] = perm[randInt];
		perm[randInt] = temp;
	}
}

// GSL implementation. NOTE: Not 100% sure whether it does the right thing.
// NOTE2: It's from Knuth's book, so probably correct :) .
// TODO: This needs to be made REALLY faster. It kills PEGASOS.
void randchoose (INT *chosen, INT populationSize, INT sampleSize) {

	INT i, j = 0;
  /* Choose k out of n items, return an array x[] of the k items.
     These items will prevserve the relative order of the original
     input -- you can use shuffle() to randomize the output if you
     wish */

	if (populationSize < sampleSize)	{
	  ERROR ("Sample size is larger than population.");
	}

	for (i = 0; i < populationSize && j < sampleSize; ++i) {
		if ((populationSize - i) * rand() / (DOUBLE) RAND_MAX < sampleSize - j) {
			chosen[j++] = i;
		}
	}
}
