/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useblas.h"
#include "useinterfaces.h"
#include "learn_sensing_svm.h"

void square_obj_grad_for_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, DOUBLE *Ypred, DOUBLE *Ytemp) {
	
	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}
	
	INT YtempFlag = 0;
	if (Ytemp == NULL) {
		Ytemp = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		YtempFlag = 1;
	}

	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Phi, &GEMVLDA, weights, &incx, &beta, Ytemp, &incy);
	
	trans = 'T';
	GEMVM = N;
	GEMVN = numSamples;
	alpha = 1;
	GEMVLDA = N;
	beta = 0;
	incx = 1;
	incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, X, &GEMVLDA, Ytemp, &incx, &beta, Ypred, &incy);
	
	INT iterX;
	DOUBLE YYpred;
	DOUBLE biasterm = *bias;
	DOUBLE objtemp = 0;
	
	INT GERM = M;
	INT GERN = N;
	INT GERLDA = M;
	
	memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	
	for (iterX = 0; iterX < numSamples; ++iterX) {
		
		YYpred = Y[iterX] * (Ypred[iterX] + biasterm);
		
		if (YYpred < 1) {
			
			objtemp += SQR(1 - YYpred);
			
			alpha = 2 * (Ypred[iterX] + biasterm - Y[iterX]) / numSamples;
			GER(&GERM, &GERN, &alpha, weights, &incx, &X[iterX * N], &incy, deriv, &GERLDA);
		}
	}
	
	*obj = objtemp / numSamples;
	
	if (YpredFlag == 1) {
		FREE(Ypred);
	}
	
	if (YtempFlag == 1) {
		FREE(Ytemp);
	}
}

void square_obj_grad_tensor_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					DOUBLE *wXtensor, INT M, INT N, INT numSamples, INT initFlag, DOUBLE *Ypred, DOUBLE *Ytemp) {
		
	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}
	
	INT YtempFlag = 0;
	if (Ytemp == NULL) {
		Ytemp = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		YtempFlag = 1;
	}

	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Phi, &GEMVLDA, weights, &incx, &beta, Ytemp, &incy);
	
	trans = 'T';
	GEMVM = N;
	GEMVN = numSamples;
	alpha = 1;
	GEMVLDA = N;
	beta = 0;
	incx = 1;
	incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, X, &GEMVLDA, Ytemp, &incx, &beta, Ypred, &incy);
	
	INT iterX;
	DOUBLE YYpred;
	DOUBLE biasterm = *bias;
	DOUBLE objtemp = 0;
	
	INT AXPYN = M * N;
	
	if (initFlag == 1) {
		memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	}
	
	for (iterX = 0; iterX < numSamples; ++iterX) {
		
		YYpred = Y[iterX] * (Ypred[iterX] + biasterm);

		if (YYpred < 1) {
			objtemp += SQR(1 - YYpred);
			
			alpha = 2 * (Ypred[iterX] + biasterm - Y[iterX]) / numSamples;
			AXPY(&AXPYN, &alpha, &wXtensor[iterX * M * N], &incx, deriv, &incy);
		}
	}
	
	if (initFlag == 1) {
		*obj = objtemp / numSamples;
	} else {
		*obj += objtemp / numSamples;
	}
	
	if (YpredFlag == 1) {
		FREE(Ypred);
	}
	
	if (YtempFlag == 1) {
		FREE(Ytemp);
	}
}

void square_obj_grad_multitask_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT numTasks, DOUBLE *Ypred, DOUBLE *Ytemp, DOUBLE *derivtemp) {
	
	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}
	
	INT YtempFlag = 0;
	if (Ytemp == NULL) {
		Ytemp = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		YtempFlag = 1;
	}
	
	INT derivtempFlag = 0;
	if (derivtemp == NULL) {
		derivtemp = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
		derivtempFlag = 1;
	}
	
	DOUBLE objtemp;
	DOUBLE objp = 0;
	
	INT AXPYN = M * N;
	DOUBLE alpha = 1;
	INT incx = 1;
	INT incy = 1;
	INT iterT;
	
	memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	for (iterT = 0; iterT < numTasks; ++iterT) {
		square_obj_grad_for_serial(&objtemp, derivtemp, Phi, X, &Y[numSamples * iterT], &weights[M * iterT], &bias[iterT], \
					M, N, numSamples, Ypred, Ytemp);
		AXPY(&AXPYN, &alpha, derivtemp, &incx, deriv, &incy);
		objp += objtemp;
	}
	
	*obj = objp;
	
	if (YpredFlag == 1) {
		FREE(Ypred);
	}
	
	if (YtempFlag == 1) {
		FREE(Ytemp);
	}
	
	if (derivtempFlag == 1) {
		FREE(derivtemp);
	}
}

void square_obj_grad_for(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT derivFlag, DOUBLE *Ypred, DOUBLE *Ytemp) {
	
	INT YpredFlag = 0;
	if (Ypred == NULL) { 
		Ypred = (DOUBLE *) CMALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}
	
	INT YtempFlag = 0;
	if (Ytemp == NULL) {
		Ytemp = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		YtempFlag = 1;
	}
	
	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Phi, &GEMVLDA, weights, &incx, &beta, Ytemp, &incy);
	
	trans = 'T';
	GEMVM = N;
	GEMVN = numSamples;
	alpha = 1;
	GEMVLDA = N;
	beta = 0;
	incx = 1;
	incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, X, &GEMVLDA, Ytemp, &incx, &beta, Ypred, &incy);
	
	INT iterX;
	DOUBLE YYpred;
	DOUBLE biasterm = *bias;
	DOUBLE objtemp = 0;
	
	INT GERM = M;
	INT GERN = N;
	INT GERLDA = M;
	
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	}
	
	for (iterX = 0; iterX < numSamples; ++iterX) {
		YYpred = Y[iterX] * (Ypred[iterX] + biasterm);
		
		if (YYpred < 1) {
			
			objtemp += (1 - YYpred)*(1-YYpred);
			
			if (derivFlag == 1) {
				alpha = 2 * (Ypred[iterX] + biasterm - Y[iterX]) / numSamples;
				GER(&GERM, &GERN, &alpha, weights, &incx, &X[iterX * N], &incy, deriv, &GERLDA);
			}
		}
	}
	
	*obj = objtemp / numSamples;
	
	if (YpredFlag == 1) {
		CFREE(Ypred);
	}
	
	if (YtempFlag == 1) {
		CFREE(Ytemp);
	}
}

void square_obj_grad_multitask(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT numTasks, INT derivFlag) {
	
	DOUBLE *Ypred;
	DOUBLE *Ytemp;
	DOUBLE *derivtemp;
	DOUBLE objtemp;
	DOUBLE objp = 0;
	
	INT AXPYN = M * N;
	DOUBLE alpha = 1;
	INT incx = 1;
	INT incy = 1;
	INT iterT;
	
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	}
	
	#pragma omp parallel private(iterT, derivtemp, Ypred, Ytemp, objtemp) shared(weights, X, Y, bias, Phi) \
			reduction(+: objp) firstprivate(M, N, numSamples, AXPYN, incx, incy, alpha) 
	{
				
		
		if (derivFlag == 1) {
			derivtemp = (DOUBLE *) CMALLOC(M * N * sizeof(DOUBLE));
		} else {
			derivtemp = NULL;
		}
		Ypred = (DOUBLE *) CMALLOC(numTasks * numSamples * sizeof(DOUBLE));
		Ytemp = (DOUBLE *) CMALLOC(N * numTasks * sizeof(DOUBLE));
		
		#pragma omp for
		for (iterT = 0; iterT < numTasks; ++iterT) {
			square_obj_grad_for(&objtemp, derivtemp, Phi, X, &Y[numSamples * iterT], &weights[M * iterT], &bias[iterT], \
						M, N, numSamples, derivFlag, Ypred, Ytemp);
			if (derivFlag == 1) {
				#pragma omp critical
					AXPY(&AXPYN, &alpha, derivtemp, &incx, deriv, &incy);
			}
			objp += objtemp;
		}
		if (derivFlag == 1) {
			CFREE(derivtemp);
		}
		CFREE(Ypred);
		CFREE(Ytemp);
	}
	
	*obj = objp;
}


void huber_obj_grad_for_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, DOUBLE *Ypred, DOUBLE *Ytemp) {
	
	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}
	
	INT YtempFlag = 0;
	if (Ytemp == NULL) {
		Ytemp = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		YtempFlag = 1;
	}

	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Phi, &GEMVLDA, weights, &incx, &beta, Ytemp, &incy);
	
	trans = 'T';
	GEMVM = N;
	GEMVN = numSamples;
	alpha = 1;
	GEMVLDA = N;
	beta = 0;
	incx = 1;
	incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, X, &GEMVLDA, Ytemp, &incx, &beta, Ypred, &incy);
	
	INT iterX;
	DOUBLE YYpred;
	DOUBLE biasterm = *bias;
	DOUBLE objtemp = 0;
	
	INT GERM = M;
	INT GERN = N;
	INT GERLDA = M;
	
	memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	
	for (iterX = 0; iterX < numSamples; ++iterX) {
		
		YYpred = Y[iterX] * (Ypred[iterX] + biasterm);
		
		if (YYpred < -1) {
			
			objtemp += -4 * YYpred;
			
			alpha = - 4 * Y[iterX] / numSamples;
			GER(&GERM, &GERN, &alpha, weights, &incx, &X[iterX * N], &incy, deriv, &GERLDA);
		} else if ((YYpred < 1) && (YYpred >= -1)) {
			
			objtemp += SQR(1 - YYpred);
			
			alpha = 2 * (Ypred[iterX] + biasterm - Y[iterX]) / numSamples;
			GER(&GERM, &GERN, &alpha, weights, &incx, &X[iterX * N], &incy, deriv, &GERLDA);
		}
	}
	
	*obj = objtemp / numSamples;
	
	if (YpredFlag == 1) {
		FREE(Ypred);
	}
	
	if (YtempFlag == 1) {
		FREE(Ytemp);
	}
}

void huber_obj_grad_tensor_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					DOUBLE *wXtensor, INT M, INT N, INT numSamples, INT initFlag, DOUBLE *Ypred, DOUBLE *Ytemp) {
		
	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}
	
	INT YtempFlag = 0;
	if (Ytemp == NULL) {
		Ytemp = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		YtempFlag = 1;
	}

	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Phi, &GEMVLDA, weights, &incx, &beta, Ytemp, &incy);
	
	trans = 'T';
	GEMVM = N;
	GEMVN = numSamples;
	alpha = 1;
	GEMVLDA = N;
	beta = 0;
	incx = 1;
	incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, X, &GEMVLDA, Ytemp, &incx, &beta, Ypred, &incy);
	
	INT iterX;
	DOUBLE YYpred;
	DOUBLE biasterm = *bias;
	DOUBLE objtemp = 0;
	
	INT AXPYN = M * N;
	
	if (initFlag == 1) {
		memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	}
	
	for (iterX = 0; iterX < numSamples; ++iterX) {
		
		YYpred = Y[iterX] * (Ypred[iterX] + biasterm);

		if (YYpred < -1) {
			objtemp += -4 * YYpred;
			
			alpha = - 4 * Y[iterX] / numSamples;
			AXPY(&AXPYN, &alpha, &wXtensor[iterX * M * N], &incx, deriv, &incy);
			
		} else if ((YYpred < 1) && (YYpred >= -1)) {
			objtemp += SQR(1 - YYpred);
			
			alpha = 2 * (Ypred[iterX] + biasterm - Y[iterX]) / numSamples;
			AXPY(&AXPYN, &alpha, &wXtensor[iterX * M * N], &incx, deriv, &incy);
		}
	}
	
	if (initFlag == 1) {
		*obj = objtemp / numSamples;
	} else {
		*obj += objtemp / numSamples;
	}
	
	if (YpredFlag == 1) {
		FREE(Ypred);
	}
	
	if (YtempFlag == 1) {
		FREE(Ytemp);
	}
}

void huber_obj_grad_multitask_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT numTasks, DOUBLE *Ypred, DOUBLE *Ytemp, DOUBLE *derivtemp) {
	
	INT YpredFlag = 0;
	if (Ypred == NULL) {
		Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}
	
	INT YtempFlag = 0;
	if (Ytemp == NULL) {
		Ytemp = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
		YtempFlag = 1;
	}
	
	INT derivtempFlag = 0;
	if (derivtemp == NULL) {
		derivtemp = (DOUBLE *) MALLOC(M * N * sizeof(DOUBLE));
		derivtempFlag = 1;
	}
	
	DOUBLE objtemp;
	DOUBLE objp = 0;
	
	INT AXPYN = M * N;
	DOUBLE alpha = 1;
	INT incx = 1;
	INT incy = 1;
	INT iterT;
	
	memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	for (iterT = 0; iterT < numTasks; ++iterT) {
		huber_obj_grad_for_serial(&objtemp, derivtemp, Phi, X, &Y[numSamples * iterT], &weights[M * iterT], &bias[iterT], \
					M, N, numSamples, Ypred, Ytemp);
		AXPY(&AXPYN, &alpha, derivtemp, &incx, deriv, &incy);
		objp += objtemp;
	}
	
	*obj = objp;
	
	if (YpredFlag == 1) {
		FREE(Ypred);
	}
	
	if (YtempFlag == 1) {
		FREE(Ytemp);
	}
	
	if (derivtempFlag == 1) {
		FREE(derivtemp);
	}
}

void huber_obj_grad_for(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT derivFlag, DOUBLE *Ypred, DOUBLE *Ytemp) {
	
	INT YpredFlag = 0;
	if (Ypred == NULL) { 
		Ypred = (DOUBLE *) CMALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}
	
	INT YtempFlag = 0;
	if (Ytemp == NULL) {
		Ytemp = (DOUBLE *) CMALLOC(N * 1 * sizeof(DOUBLE));
		YtempFlag = 1;
	}
	
	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, Phi, &GEMVLDA, weights, &incx, &beta, Ytemp, &incy);
	
	trans = 'T';
	GEMVM = N;
	GEMVN = numSamples;
	alpha = 1;
	GEMVLDA = N;
	beta = 0;
	incx = 1;
	incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, X, &GEMVLDA, Ytemp, &incx, &beta, Ypred, &incy);
	
	INT iterX;
	DOUBLE YYpred;
	DOUBLE biasterm = *bias;
	DOUBLE objtemp = 0;
	
	INT GERM = M;
	INT GERN = N;
	INT GERLDA = M;
	
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	}
	
	for (iterX = 0; iterX < numSamples; ++iterX) {
		YYpred = Y[iterX] * (Ypred[iterX] + biasterm);
		
		if (YYpred < -1) {
			
			objtemp += -4 * YYpred;
			
			if (derivFlag == 1) {			
				alpha = - 4 * Y[iterX] / numSamples;
				GER(&GERM, &GERN, &alpha, weights, &incx, &X[iterX * N], &incy, deriv, &GERLDA);
			}
		} else if ((YYpred < 1) && (YYpred >= -1)) {
			
			objtemp += (1 - YYpred) * (1 - YYpred);
			
			if (derivFlag == 1) {
				alpha = 2 * (Ypred[iterX] + biasterm - Y[iterX]) / numSamples;
				GER(&GERM, &GERN, &alpha, weights, &incx, &X[iterX * N], &incy, deriv, &GERLDA);
			}
		}
	}
	
	*obj = objtemp / numSamples;
	
	if (YpredFlag == 1) {
		CFREE(Ypred);
	}
	
	if (YtempFlag == 1) {
		CFREE(Ytemp);
	}
}

void huber_obj_grad_multitask(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT numTasks, INT derivFlag) {
	
	DOUBLE *Ypred;
	DOUBLE *Ytemp;
	DOUBLE *derivtemp;
	DOUBLE objtemp;
	DOUBLE objp = 0;
	
	INT AXPYN = M * N;
	DOUBLE alpha = 1;
	INT incx = 1;
	INT incy = 1;
	INT iterT;
	
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	}
	
	#pragma omp parallel private(iterT, derivtemp, Ypred, Ytemp, objtemp) shared(weights, X, Y, bias, Phi) \
			reduction(+: objp) firstprivate(M, N, numSamples, AXPYN, incx, incy, alpha) 
	{
		if (derivFlag == 1) {
			derivtemp = (DOUBLE *) CMALLOC(M * N * sizeof(DOUBLE));
		} else {
			derivtemp = NULL;
		}
		Ypred = (DOUBLE *) CMALLOC(numTasks * numSamples * sizeof(DOUBLE));
		Ytemp = (DOUBLE *) CMALLOC(N * numTasks * sizeof(DOUBLE));
		
		#pragma omp for
		for (iterT = 0; iterT < numTasks; ++iterT) {
			huber_obj_grad_for(&objtemp, derivtemp, Phi, X, &Y[numSamples * iterT], &weights[M * iterT], &bias[iterT], \
						M, N, numSamples, derivFlag, Ypred, Ytemp);
			if (derivFlag == 1) {
				#pragma omp critical
					AXPY(&AXPYN, &alpha, derivtemp, &incx, deriv, &incy);
			}
			objp += objtemp;
		}
		if (derivFlag == 1) {
			CFREE(derivtemp);
		}
		CFREE(Ypred);
		CFREE(Ytemp);
	}
	
	*obj = objp;
}

void huber_average_obj_grad_for(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT numPool, INT derivFlag, DOUBLE *Ypred, DOUBLE *Ytemp) {
	
	INT YpredFlag = 0;
	if (Ypred == NULL) { 
		Ypred = (DOUBLE *) CMALLOC(1 * numSamples * sizeof(DOUBLE));
		YpredFlag = 1;
	}
	
	INT YtempFlag = 0;
	if (Ytemp == NULL) {
		Ytemp = (DOUBLE *) CMALLOC(N * numPool * 1 * sizeof(DOUBLE));
		YtempFlag = 1;
	}
	
	CHAR trans = 'T';
	INT GEMVM = M;
	INT GEMVN = N;
	DOUBLE alpha = 1;
	INT GEMVLDA = M;
	DOUBLE beta = 0;
	INT incx = 1;
	INT incy = 1;
	
	INT iterP;
	
	for (iterP = 0; iterP < numPool; ++iterP) {
		GEMV(&trans, &GEMVM, &GEMVN, &alpha, Phi, &GEMVLDA, &weights[iterP * M], &incx, &beta, &Ytemp[iterP * N], &incy);
	}
	
	trans = 'T';
	GEMVM = N * numPool;
	GEMVN = numSamples;
	alpha = 1;
	GEMVLDA = N * numPool;
	beta = 0;
	incx = 1;
	incy = 1;
	
	GEMV(&trans, &GEMVM, &GEMVN, &alpha, X, &GEMVLDA, Ytemp, &incx, &beta, Ypred, &incy);
	
	INT iterX;
	DOUBLE YYpred;
	DOUBLE biasterm = *bias;
	DOUBLE objtemp = 0;
	
	INT GERM = M;
	INT GERN = N;
	INT GERLDA = M;
	
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	}
	
	for (iterX = 0; iterX < numSamples; ++iterX) {
		YYpred = Y[iterX] * (Ypred[iterX] + biasterm);
		
		if (YYpred < -1) {
			
			objtemp += -4 * YYpred;
			
			if (derivFlag == 1) {			
				alpha = - 4 * Y[iterX] / numSamples;
				for (iterP = 0; iterP < numPool; ++iterP) {
					GER(&GERM, &GERN, &alpha, &weights[iterP * M], &incx, &X[iterX * N * numPool + iterP * N], &incy, deriv, &GERLDA);
				}
			}
		} else if ((YYpred < 1) && (YYpred >= -1)) {
			
			objtemp += (1 - YYpred)*(1-YYpred);
			
			if (derivFlag == 1) {
				alpha = 2 * (Ypred[iterX] + biasterm - Y[iterX]) / numSamples;
				for (iterP = 0; iterP < numPool; ++iterP) {
					GER(&GERM, &GERN, &alpha, &weights[iterP * M], &incx, &X[iterX * N * numPool + iterP * N], &incy, deriv, &GERLDA);
				}
			}
		}
	}
	
	*obj = objtemp / numSamples;
	
	if (YpredFlag == 1) {
		CFREE(Ypred);
	}
	
	if (YtempFlag == 1) {
		CFREE(Ytemp);
	}
}

void huber_average_obj_grad_multitask(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT numPool, INT numTasks, INT derivFlag) {
	
	DOUBLE *Ypred;
	DOUBLE *Ytemp;
	DOUBLE *derivtemp;
	DOUBLE objtemp;
	DOUBLE objp = 0;
	
	INT AXPYN = M * N;
	DOUBLE alpha = 1;
	INT incx = 1;
	INT incy = 1;
	INT iterT;
	
	if (derivFlag == 1) {
		memset((void *) deriv, 0, M * N * sizeof(DOUBLE));
	}
	
	#pragma omp parallel private(iterT, derivtemp, Ypred, Ytemp, objtemp) shared(weights, X, Y, bias, Phi) \
			reduction(+: objp) firstprivate(M, N, numSamples, AXPYN, incx, incy, alpha) 
	{
		if (derivFlag == 1) {
			derivtemp = (DOUBLE *) CMALLOC(M * N * sizeof(DOUBLE));
		} else {
			derivtemp = NULL;
		}
		Ypred = (DOUBLE *) CMALLOC(numTasks * numSamples * sizeof(DOUBLE));
		Ytemp = (DOUBLE *) CMALLOC(N * numPool * numTasks * sizeof(DOUBLE));
		
		#pragma omp for
		for (iterT = 0; iterT < numTasks; ++iterT) {
			huber_average_obj_grad_for(&objtemp, derivtemp, Phi, X, &Y[numSamples * iterT], &weights[M * iterT], &bias[iterT], \
						M, N, numSamples, numPool, derivFlag, Ypred, Ytemp);
			if (derivFlag == 1) {
				#pragma omp critical
					AXPY(&AXPYN, &alpha, derivtemp, &incx, deriv, &incy);
			}
			objp += objtemp;
		}
		if (derivFlag == 1) {
			CFREE(derivtemp);
		}
		CFREE(Ypred);
		CFREE(Ytemp);
	}
	
	*obj = objp;
}


void minimize_huber_average(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *Xtrain, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
				INT M, INT N, INT numSamples, INT numPool, INT numTasks) {

	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
	INT MN = M * N;
	
	CHAR lamch_opt = 'U';
	DOUBLE realmin = LAMCH(&lamch_opt);

	DOUBLE red = 1;

	INT i = 0;
	INT ls_failed = 0;
	DOUBLE f0;
	DOUBLE *df0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *dftemp = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *df3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *s = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE d0;
	INT derivFlag = 1;

	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	
	huber_average_obj_grad_multitask(&f0, df0, X, Xtrain, Y, weights, bias, \
					M, N, numSamples, numPool, numTasks, derivFlag);
	
	INT incx = 1;
	INT incy = 1;
	INT iter;

	datacpy(s, df0, MN);
	DOUBLE alpha = -1;
	SCAL(&MN, &alpha, s, &incx);
	
	d0 = - DOT(&MN, s, &incx, s, &incy);
	
	DOUBLE x1;
	DOUBLE x2;
	DOUBLE x3;
	DOUBLE x4;
	DOUBLE *X0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *X3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE F0;
	DOUBLE *dF0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	INT Mmin;
	DOUBLE f1;
	DOUBLE f2;
	DOUBLE f3;
	DOUBLE f4;
	DOUBLE d1;
	DOUBLE d2;
	DOUBLE d3;
	DOUBLE d4;
	INT success;
	DOUBLE A;
	DOUBLE B;
	DOUBLE sqrtquantity;
	DOUBLE tempnorm;
	DOUBLE tempinprod1;
	DOUBLE tempinprod2;
	DOUBLE tempscalefactor;

	x3 = red / (1 - d0);            

	while (i++ < length) {
		datacpy(X0, X, MN);
		datacpy(dF0, df0, MN);
		F0 = f0;
		Mmin = MAX;
		
		while (1) {
			x2 = 0;
			f2 = f0;
			d2 = d0;
			f3 = f0;

			datacpy(df3, df0, MN);
			
			success = 0;
			while ((!success) && (Mmin > 0)) {
				Mmin = Mmin - 1;

				datacpy(X3, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X3, &incy);
				
				huber_average_obj_grad_multitask(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numPool, numTasks, derivFlag);
				
				if (ISNAN(f3) || ISINF(f3)) {  /* any(isnan(df3)+isinf(df3)) */
					x3 = (x2 + x3) * 0.5;
				} else {
					success = 1;
				}
			}
			
			if (f3 < F0) {
				datacpy(X0, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X0, &incy);
				datacpy(dF0, df3, MN);
				F0 = f3;
			}	
			
			d3 = DOT(&MN, df3, &incx, s, &incy);

			if ((d3 > SIG * d0) || (f3 > f0 + x3 * RHO * d0) || (Mmin == 0)) {
				break;
			}
			
			x1 = x2; 
			f1 = f2; 
			d1 = d2;
			x2 = x3; 
			f2 = f3; 
			d2 = d3;
			A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1);
			B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1);
			sqrtquantity = B * B - A * d1 * (x2 - x1);

			if (sqrtquantity < 0) {
				x3 = x2 * EXT;
			} else {
				x3 = x1 - d1 * SQR(x2 - x1) / (B + SQRT(sqrtquantity));
				if (ISNAN(x3) || ISINF(x3) || (x3 < 0)) {
					x3 = x2 * EXT;
				} else if (x3 > x2 * EXT) {
					x3 = x2 * EXT;
				} else if (x3 < x2 + INTERV * (x2 - x1)) {
					x3 = x2 + INTERV * (x2 - x1);
				}
			}		
		}                
	
		while (((ABS(d3) > - SIG * d0) || (f3 > f0 + x3 * RHO * d0)) && (Mmin > 0)) {
			if ((d3 > 0) || (f3 > f0 + x3 * RHO * d0)) {
				x4 = x3;
				f4 = f3;
				d4 = d3;
			} else {
				x2 = x3;
				f2 = f3;
				d2 = d3;
			}

			if (f4 > f0) {
				x3 = x2 - (0.5 * d2 * SQR(x4 - x2)) / (f4 - f2 - d2 * (x4 - x2));
			} else {
				A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2);
				B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2);
				x3 = x2 + (SQRT(B * B - A * d2 * SQR(x4 - x2)) - B) / A;
			}

			if (ISNAN(x3) || ISINF(x3)) {
				x3 = (x2 + x4) * 0.5;
			}
			x3 = IMAX(IMIN(x3, x4 - INTERV * (x4 - x2)), x2 + INTERV * (x4 - x2));

			datacpy(X3, X, MN);
			alpha = x3;
			AXPY(&MN, &alpha, s, &incx, X3, &incy);			

			huber_average_obj_grad_multitask(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numPool, numTasks, derivFlag);		

			if (f3 < F0) {
				datacpy(X0, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X0, &incy);
				datacpy(dF0, df3, MN);
				F0 = f3;
			}

			Mmin = Mmin - 1;
			d3 = DOT(&MN, df3, &incx, s, &incy);
			
		}
		
		if ((ABS(d3) < - SIG * d0) && (f3 < f0 + x3 * RHO * d0)) {
			alpha = x3;
			AXPY(&MN, &alpha, s, &incx, X, &incy);
			f0 = f3;
			
			datacpy(dftemp, df3, MN);
			alpha = -1;
			AXPY(&MN, &alpha, df0, &incx, dftemp, &incy);
			tempinprod1 = DOT(&MN, dftemp, &incx, df3, &incy);
			tempnorm = NRM2(&MN, df0, &incx);
			tempinprod2 = SQR(tempnorm);
			tempscalefactor = tempinprod1 / tempinprod2;

			alpha = tempscalefactor;
			SCAL(&MN, &alpha, s, &incx);
			alpha = -1;
			AXPY(&MN, &alpha, df3, &incx, s, &incy);
			datacpy(df0, df3, MN);
			d3 = d0;
			d0 = DOT(&MN, df0, &incx, s, &incy);

			if (d0 > 0) {
				datacpy(s, df0, MN);
				alpha = -1;
				SCAL(&MN, &alpha, s, &incx);
				tempnorm = NRM2(&MN, s, &incx);
				d0 = - SQR(tempnorm);
			}
			x3 = x3 * IMIN(RATIO, d3 / (d0 - realmin));
			ls_failed = 0;
		} else {
			datacpy(X, X0, MN);
			datacpy(df0, dF0, MN);
			f0 = F0;
			
			if ((ls_failed == 1) || (i > length)) {
				break;
			}
			
			datacpy(s, df0, MN);
			alpha = -1;
			SCAL(&MN, &alpha, s, &incx);
			tempnorm = NRM2(&MN, s, &incx);
			d0 = - SQR(tempnorm);
			x3 = 1 / (1 - d0);
			
			ls_failed = 1;
		}
	}

	datacpy(Xopt, X, MN);
	
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}


void minimize_huber(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *Xtrain, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
				INT M, INT N, INT numSamples, INT numTasks) {

	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
	INT MN = M * N;
	
	CHAR lamch_opt = 'U';
	DOUBLE realmin = LAMCH(&lamch_opt);

	DOUBLE red = 1;

	INT i = 0;
	INT ls_failed = 0;
	DOUBLE f0;
	DOUBLE *df0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *dftemp = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *df3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *s = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE d0;
	INT derivFlag = 1;

	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	
#ifdef __USE_PARALLEL__
	huber_obj_grad_multitask(&f0, df0, X, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, derivFlag);
#else
	DOUBLE *Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
	DOUBLE *Ytemp = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *derivtemp = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	huber_obj_grad_multitask_serial(&f0, df0, X, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, Ypred, Ytemp, derivtemp);
#endif
	
	INT incx = 1;
	INT incy = 1;
	INT iter;

	datacpy(s, df0, MN);
	DOUBLE alpha = -1;
	SCAL(&MN, &alpha, s, &incx);
	
	d0 = - DOT(&MN, s, &incx, s, &incy);
	
	DOUBLE x1;
	DOUBLE x2;
	DOUBLE x3;
	DOUBLE x4;
	DOUBLE *X0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *X3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE F0;
	DOUBLE *dF0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	INT Mmin;
	DOUBLE f1;
	DOUBLE f2;
	DOUBLE f3;
	DOUBLE f4;
	DOUBLE d1;
	DOUBLE d2;
	DOUBLE d3;
	DOUBLE d4;
	INT success;
	DOUBLE A;
	DOUBLE B;
	DOUBLE sqrtquantity;
	DOUBLE tempnorm;
	DOUBLE tempinprod1;
	DOUBLE tempinprod2;
	DOUBLE tempscalefactor;

	x3 = red / (1 - d0);            

	while (i++ < length) {
		datacpy(X0, X, MN);
		datacpy(dF0, df0, MN);
		F0 = f0;
		Mmin = MAX;
		
		while (1) {
			x2 = 0;
			f2 = f0;
			d2 = d0;
			f3 = f0;
			
			datacpy(df3, df0, MN);
			
			success = 0;
			while ((!success) && (Mmin > 0)) {
				Mmin = Mmin - 1;

				datacpy(X3, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X3, &incy);
				
#ifdef __USE_PARALLEL__
				huber_obj_grad_multitask(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, derivFlag);
#else
				huber_obj_grad_multitask_serial(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, Ypred, Ytemp, derivtemp);
#endif				

				if (ISNAN(f3) || ISINF(f3)) {  /* any(isnan(df3)+isinf(df3)) */
					x3 = (x2 + x3) * 0.5;
				} else {
					success = 1;
				}
			}
			
			if (f3 < F0) {
				datacpy(X0, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X0, &incy);
				datacpy(dF0, df3, MN);
				F0 = f3;
			}	
			
			d3 = DOT(&MN, df3, &incx, s, &incy);

			if ((d3 > SIG * d0) || (f3 > f0 + x3 * RHO * d0) || (Mmin == 0)) {
				break;
			}
			
			x1 = x2; 
			f1 = f2; 
			d1 = d2;
			x2 = x3; 
			f2 = f3; 
			d2 = d3;
			A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1);
			B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1);
			sqrtquantity = B * B - A * d1 * (x2 - x1);

			if (sqrtquantity < 0) {
				x3 = x2 * EXT;
			} else {
				x3 = x1 - d1 * SQR(x2 - x1) / (B + SQRT(sqrtquantity));
				if (ISNAN(x3) || ISINF(x3) || (x3 < 0)) {
					x3 = x2 * EXT;
				} else if (x3 > x2 * EXT) {
					x3 = x2 * EXT;
				} else if (x3 < x2 + INTERV * (x2 - x1)) {
					x3 = x2 + INTERV * (x2 - x1);
				}
			}		
		}                
	
		while (((ABS(d3) > - SIG * d0) || (f3 > f0 + x3 * RHO * d0)) && (Mmin > 0)) {
			if ((d3 > 0) || (f3 > f0 + x3 * RHO * d0)) {
				x4 = x3;
				f4 = f3;
				d4 = d3;
			} else {
				x2 = x3;
				f2 = f3;
				d2 = d3;
			}

			if (f4 > f0) {
				x3 = x2 - (0.5 * d2 * SQR(x4 - x2)) / (f4 - f2 - d2 * (x4 - x2));
			} else {
				A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2);
				B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2);
				x3 = x2 + (SQRT(B * B - A * d2 * SQR(x4 - x2)) - B) / A;
			}

			if (ISNAN(x3) || ISINF(x3)) {
				x3 = (x2 + x4) * 0.5;
			}
			x3 = IMAX(IMIN(x3, x4 - INTERV * (x4 - x2)), x2 + INTERV * (x4 - x2));

			datacpy(X3, X, MN);
			alpha = x3;
			AXPY(&MN, &alpha, s, &incx, X3, &incy);			

#ifdef __USE_PARALLEL__
			huber_obj_grad_multitask(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, derivFlag);
#else
			huber_obj_grad_multitask_serial(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, Ypred, Ytemp, derivtemp);
#endif				

			if (f3 < F0) {
				datacpy(X0, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X0, &incy);
				datacpy(dF0, df3, MN);
				F0 = f3;
			}

			Mmin = Mmin - 1;
			d3 = DOT(&MN, df3, &incx, s, &incy);
		}
		
		if ((ABS(d3) < - SIG * d0) && (f3 < f0 + x3 * RHO * d0)) {
			alpha = x3;
			AXPY(&MN, &alpha, s, &incx, X, &incy);
			f0 = f3;
			
			datacpy(dftemp, df3, MN);
			alpha = -1;
			AXPY(&MN, &alpha, df0, &incx, dftemp, &incy);
			tempinprod1 = DOT(&MN, dftemp, &incx, df3, &incy);
			tempnorm = NRM2(&MN, df0, &incx);
			tempinprod2 = SQR(tempnorm);
			tempscalefactor = tempinprod1 / tempinprod2;
			
			alpha = tempscalefactor;
			SCAL(&MN, &alpha, s, &incx);
			alpha = -1;
			AXPY(&MN, &alpha, df3, &incx, s, &incy);
			datacpy(df0, df3, MN);
			d3 = d0;
			d0 = DOT(&MN, df0, &incx, s, &incy);

			if (d0 > 0) {
				datacpy(s, df0, MN);
				alpha = -1;
				SCAL(&MN, &alpha, s, &incx);
				tempnorm = NRM2(&MN, s, &incx);
				d0 = - SQR(tempnorm);
			}
			x3 = x3 * IMIN(RATIO, d3 / (d0 - realmin));
			ls_failed = 0;
		} else {
			datacpy(X, X0, MN);
			datacpy(df0, dF0, MN);
			f0 = F0;
			
			if ((ls_failed == 1) || (i > length)) {
				break;
			}
			
			datacpy(s, df0, MN);
			alpha = -1;
			SCAL(&MN, &alpha, s, &incx);
			tempnorm = NRM2(&MN, s, &incx);
			d0 = - SQR(tempnorm);
			x3 = 1 / (1 - d0);
			
			ls_failed = 1;
		}
	}

	datacpy(Xopt, X, MN);
	
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}

void minimize_square(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *Xtrain, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
				INT M, INT N, INT numSamples, INT numTasks) {

	DOUBLE INTERV = 0.1;
	DOUBLE EXT = 3.0;   
	INT MAX = 20;       
	DOUBLE RATIO = (DOUBLE) 10;  
	DOUBLE SIG = 0.1; 
	DOUBLE RHO = SIG / (DOUBLE) 2;
	INT MN = M * N;
	
	CHAR lamch_opt = 'U';
	DOUBLE realmin = LAMCH(&lamch_opt);

	DOUBLE red = 1;

	INT i = 0;
	INT ls_failed = 0;
	DOUBLE f0;
	DOUBLE *df0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *dftemp = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *df3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *s = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE d0;
	INT derivFlag = 1;
	
	DOUBLE *X = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	datacpy(X, Xorig, MN);
	
#ifdef __USE_PARALLEL__
	square_obj_grad_multitask(&f0, df0, X, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, derivFlag);
#else
	DOUBLE *Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
	DOUBLE *Ytemp = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	DOUBLE *derivtemp = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	square_obj_grad_multitask_serial(&f0, df0, X, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, Ypred, Ytemp, derivtemp);
#endif
	
	INT incx = 1;
	INT incy = 1;
	INT iter;
		
	datacpy(s, df0, MN);
	DOUBLE alpha = -1;
	SCAL(&MN, &alpha, s, &incx);
	
	d0 = - DOT(&MN, s, &incx, s, &incy);
	
	DOUBLE x1;
	DOUBLE x2;
	DOUBLE x3;
	DOUBLE x4;
	DOUBLE *X0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE *X3 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	DOUBLE F0;
	DOUBLE *dF0 = (DOUBLE *) MALLOC(MN * sizeof(DOUBLE));
	INT Mmin;
	DOUBLE f1;
	DOUBLE f2;
	DOUBLE f3;
	DOUBLE f4;
	DOUBLE d1;
	DOUBLE d2;
	DOUBLE d3;
	DOUBLE d4;
	INT success;
	DOUBLE A;
	DOUBLE B;
	DOUBLE sqrtquantity;
	DOUBLE tempnorm;
	DOUBLE tempinprod1;
	DOUBLE tempinprod2;
	DOUBLE tempscalefactor;
	
	x3 = red / (1 - d0);            

	while (i++ < length) {
		datacpy(X0, X, MN);
		datacpy(dF0, df0, MN);
		F0 = f0;
		Mmin = MAX;
		
		while (1) {
			x2 = 0;
			f2 = f0;
			d2 = d0;
			f3 = f0;

			datacpy(df3, df0, MN);
			
			success = 0;
			while ((!success) && (Mmin > 0)) {
				Mmin = Mmin - 1;
				datacpy(X3, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X3, &incy);
				
#ifdef __USE_PARALLEL__
				square_obj_grad_multitask(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, derivFlag);
#else
				square_obj_grad_multitask_serial(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, Ypred, Ytemp, derivtemp);
#endif

				if (ISNAN(f3) || ISINF(f3)) {  /* any(isnan(df3)+isinf(df3)) */
					x3 = (x2 + x3) * 0.5;
				} else {
					success = 1;
				}
			}
			
			if (f3 < F0) {
				datacpy(X0, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X0, &incy);
				datacpy(dF0, df3, MN);
				F0 = f3;
			}	
			
			d3 = DOT(&MN, df3, &incx, s, &incy);

			if ((d3 > SIG * d0) || (f3 > f0 + x3 * RHO * d0) || (Mmin == 0)) {
				break;
			}
			
			x1 = x2; 
			f1 = f2; 
			d1 = d2;
			x2 = x3; 
			f2 = f3; 
			d2 = d3;
			A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1);
			B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1);
			sqrtquantity = B * B - A * d1 * (x2 - x1);

			if (sqrtquantity < 0) {
				x3 = x2 * EXT;
			} else {
				x3 = x1 - d1 * SQR(x2 - x1) / (B + SQRT(sqrtquantity));
				if (ISNAN(x3) || ISINF(x3) || (x3 < 0)) {
					x3 = x2 * EXT;
				} else if (x3 > x2 * EXT) {
					x3 = x2 * EXT;
				} else if (x3 < x2 + INTERV * (x2 - x1)) {
					x3 = x2 + INTERV * (x2 - x1);
				}
			}		
		}                
	
		while (((ABS(d3) > - SIG * d0) || (f3 > f0 + x3 * RHO * d0)) && (Mmin > 0)) {
			if ((d3 > 0) || (f3 > f0 + x3 * RHO * d0)) {
				x4 = x3;
				f4 = f3;
				d4 = d3;
			} else {
				x2 = x3;
				f2 = f3;
				d2 = d3;
			}

			if (f4 > f0) {
				x3 = x2 - (0.5 * d2 * SQR(x4 - x2)) / (f4 - f2 - d2 * (x4 - x2));
			} else {
				A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2);
				B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2);
				x3 = x2 + (SQRT(B * B - A * d2 * SQR(x4 - x2)) - B) / A;
			}

			if (ISNAN(x3) || ISINF(x3)) {
				x3 = (x2 + x4) * 0.5;
			}
			x3 = IMAX(IMIN(x3, x4 - INTERV * (x4 - x2)), x2 + INTERV * (x4 - x2));

			datacpy(X3, X, MN);
			alpha = x3;
			AXPY(&MN, &alpha, s, &incx, X3, &incy);
							
#ifdef __USE_PARALLEL__
			square_obj_grad_multitask(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, derivFlag);
#else
			square_obj_grad_multitask_serial(&f3, df3, X3, Xtrain, Y, weights, bias, \
					M, N, numSamples, numTasks, Ypred, Ytemp, derivtemp);
#endif

			if (f3 < F0) {
				datacpy(X0, X, MN);
				alpha = x3;
				AXPY(&MN, &alpha, s, &incx, X0, &incy);
				datacpy(dF0, df3, MN);
				F0 = f3;
			}

			Mmin = Mmin - 1;
			d3 = DOT(&MN, df3, &incx, s, &incy);
		}
		
		if ((ABS(d3) < - SIG * d0) && (f3 < f0 + x3 * RHO * d0)) {
			alpha = x3;
			AXPY(&MN, &alpha, s, &incx, X, &incy);
			f0 = f3;
			
			datacpy(dftemp, df3, MN);
			alpha = -1;
			AXPY(&MN, &alpha, df0, &incx, dftemp, &incy);
			tempinprod1 = DOT(&MN, dftemp, &incx, df3, &incy);
			tempnorm = NRM2(&MN, df0, &incx);
			tempinprod2 = SQR(tempnorm);
			tempscalefactor = tempinprod1 / tempinprod2;

			alpha = tempscalefactor;
			SCAL(&MN, &alpha, s, &incx);
			alpha = -1;
			AXPY(&MN, &alpha, df3, &incx, s, &incy);
			datacpy(df0, df3, MN);
			d3 = d0;
			d0 = DOT(&MN, df0, &incx, s, &incy);

			if (d0 > 0) {
				datacpy(s, df0, MN);
				alpha = -1;
				SCAL(&MN, &alpha, s, &incx);
				tempnorm = NRM2(&MN, s, &incx);
				d0 = - SQR(tempnorm);
			}
			x3 = x3 * IMIN(RATIO, d3 / (d0 - realmin));
			ls_failed = 0;
		} else {
			datacpy(X, X0, MN);
			datacpy(df0, dF0, MN);
			f0 = F0;
			
			if ((ls_failed == 1) || (i > length)) {
				break;
			}
			
			datacpy(s, df0, MN);
			alpha = -1;
			SCAL(&MN, &alpha, s, &incx);
			tempnorm = NRM2(&MN, s, &incx);
			d0 = - SQR(tempnorm);
			x3 = 1 / (1 - d0);
			
			ls_failed = 1;
		}
	}

	datacpy(Xopt, X, MN);
	
#ifdef __USE_PARALLEL_
	FREE(Ypred);
	FREE(Ytemp);
	FREE(derivtemp);
#endif
	FREE(df0);
	FREE(dftemp);
	FREE(df3);
	FREE(s);
	FREE(X);
	FREE(X0);
	FREE(X3);
	FREE(dF0);
}
