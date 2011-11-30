/*
#define __DEBUG__
*/

#ifdef __DEBUG__
#include "mat.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

/* TODO: Merge computational part with pca.c, convert the rest to regular format
 * for _mex.c.
 */


#ifdef __DEBUG__

void putVariable(DOUBLE *varData, INT varSize1, INT varSize2, char *varName, MATFile *matFile) {

	int status; 
	mxArray *tempMat = mxCreateNumericMatrix(varSize1, varSize2, MXPRECISION_CLASS, mxREAL);
	DOUBLE *tempPtr = (DOUBLE*) mxGetData(tempMat);
	datacpy(tempPtr, varData, varSize1 * varSize2);
	status = matPutVariable(matFile, varName, tempMat);
	if (status != 0) {
	  PRINTF("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
	  exit(EXIT_FAILURE);
	}  
	mxDestroyArray(tempMat);
}
#endif



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Check number of input arguments */
	if (nrhs < 3) {
		ERROR("Three input arguments are required.");
    } else if (nrhs > 6) {
		ERROR("At most six input arguments may be provided.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }
	
	DOUBLE *X1 = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X2 = (DOUBLE*) mxGetData(prhs[1]);
	INT numDims = (INT) * (DOUBLE *) mxGetData(prhs[2]);
	CHAR kernelName;
	if (nrhs >= 4) {
		if (!mxIsChar(prhs[3])) {
			ERROR("Third argument must be of type CHAR.");
		}
		kernelName = (CHAR)*(CHAR*) mxGetData(prhs[3]);
	} else {
		kernelName = 'G';
	}
	KERNEL_TYPE kernelType = convertKernelName(kernelName);
	
	DOUBLE *param1;
	DOUBLE *param2;
	if (nrhs >= 5) {
		param1 = (DOUBLE*) mxGetData(prhs[4]);
	} else {
		param1 = NULL;
	}
	if (nrhs >= 6) {
		param2 = (DOUBLE*) mxGetData(prhs[5]);
	} else {
		param2 = NULL;
	}
	
	INT N = (INT) mxGetM(prhs[0]);
	INT numSamples1 = (INT) mxGetN(prhs[0]);
	INT numSamples2 = (INT) mxGetN(prhs[1]);
	
	if ((mxGetM(prhs[1]) != N) && (numSamples2 != 0)) {
		ERROR("The signal dimension (first dimension) of the second sample matrix does not match the signal dimension (first dimension) of the first sample matrix.");
	}
	
	DOUBLE *normMat1 = NULL;
	if (kernelType == KERNEL_GAUSSIAN) {
		normMat1 = (DOUBLE *) MALLOC(numSamples1 * 1 * sizeof(DOUBLE));
	}
	DOUBLE *oneVec = (DOUBLE *) MALLOC(numSamples1 * 1 * sizeof(DOUBLE));
	
	DOUBLE *gramTrainTrain = (DOUBLE *) MALLOC(numSamples1 * numSamples1 * sizeof(DOUBLE));
	kernel_gram(gramTrainTrain, X1, NULL, N, numSamples1, 0, kernelType, param1, param2, \
				normMat1, oneVec);
	
#ifdef __DEBUG__
	MATFile *matFile;
	const char *fileName = "kernel_pca_mex_debug.mat";

	PRINTF("Creating file %s...\n", fileName);
	matFile = matOpen(fileName, "w");
	if (matFile == NULL) {
		PRINTF("Error creating file %s\n", fileName);
		exit(EXIT_FAILURE);
	}
	
	PRINTF("Storing gramTrainTrainPreprocessed...\n");
	putVariable(gramTrainTrain, numSamples1, numSamples1, "gramTrainTrainDbg", matFile);
#endif	

	DOUBLE *column_sums = (DOUBLE *) MALLOC(numSamples1 * sizeof(DOUBLE));
	DOUBLE total_sum;
	
	total_sum = 0;
	DOUBLE temp_sum;
	INT iterM;
	INT iterN;
	for (iterN = 0; iterN < numSamples1; ++iterN) {
		temp_sum = 0;
		for (iterM = 0; iterM < numSamples1; ++iterM) {
			temp_sum += gramTrainTrain[iterN * numSamples1 + iterM];
		}
		column_sums[iterN] = temp_sum / numSamples1;
		total_sum += column_sums[iterN] / numSamples1;
		oneVec[iterN] = 1;
	}
	
	INT AXPYN = numSamples1;
	DOUBLE alpha = -1;
	DOUBLE alpha2 = total_sum;
	DOUBLE alpha3;
	INT INCX = 1;
	INT INCY = 1;
	for (iterN = 0; iterN < numSamples1; ++iterN) {
		AXPY(&AXPYN, &alpha, column_sums, &INCX, &gramTrainTrain[iterN * numSamples1], &INCY);
		alpha3 = - column_sums[iterN] + alpha2;
		AXPY(&AXPYN, &alpha3, oneVec, &INCX, &gramTrainTrain[iterN * numSamples1], &INCY);
	}

#ifdef __DEBUG__
	PRINTF("Storing gramTrainTrainPostprocessed...\n");
	putVariable(gramTrainTrain, numSamples1, numSamples1, "gramTrainTrainPostDbg", matFile);
	putVariable(column_sums, numSamples1, 1, "columnSumsDbg", matFile);
	putVariable(&total_sum, 1, 1, "totalSumDbg", matFile);
#endif
	
	CHAR lamch_opt = 'S';
	DOUBLE sfmin = LAMCH(&lamch_opt);
	
	/* Setup SYEVR parameters */
	CHAR jobz = 'V';
	CHAR range = 'I';
	CHAR uplo = 'U';
	INT SYEVRN = numSamples1;
	INT SYEVLDA = numSamples1;
	DOUBLE VL, VU;
	INT IL = numSamples1 - numDims + 1;
	INT IU = numSamples1;
	DOUBLE abstol = sfmin;
	INT SYEVLDZ = numSamples1;
	INT lwork = -1;
	INT liwork = -1;
	DOUBLE *work;
	INT *iwork;
	DOUBLE work_temp;
	INT iwork_temp;
	INT SYEVRM;
	INT INFO;
	const INT SYEVRM_expected = IU - IL + 1;
	
	DOUBLE *lvec = (DOUBLE *) MALLOC(numSamples1 * 1 * sizeof(DOUBLE));
	DOUBLE *Vr = (DOUBLE *) MALLOC(numSamples1 * numDims * sizeof(DOUBLE));
	INT *ISUPPZ;
	if (SYEVRM_expected == SYEVRN) {
		ISUPPZ = (INT *) MALLOC(2 * SYEVRM_expected * sizeof(INT));
	} else {
		ISUPPZ = NULL;
	}
	
	SYEVR(&jobz, &range, &uplo, &SYEVRN, gramTrainTrain, &SYEVLDA, &VL, &VU, &IL, &IU, &abstol, &SYEVRM, \
			lvec, Vr, &SYEVLDZ, ISUPPZ, &work_temp, &lwork, &iwork_temp, &liwork, &INFO);
	
	lwork = (INT) work_temp;
	work = (DOUBLE*) MALLOC(lwork * sizeof(DOUBLE));
	liwork = (INT) iwork_temp;
	iwork = (INT*) MALLOC(liwork * sizeof(INT));
	
	SYEVR(&jobz, &range, &uplo, &SYEVRN, gramTrainTrain, &SYEVLDA, &VL, &VU, &IL, &IU, &abstol, &SYEVRM, \
			lvec, Vr, &SYEVLDZ, ISUPPZ, work, &lwork, iwork, &liwork, &INFO);	
	if (SYEVRM != SYEVRM_expected) {
		PRINTF("Error, only %d eigenvalues were found, when %d were expected. ", SYEVRM, SYEVRM_expected);
		ERROR("LAPACK execution error.");
	}
	FREE(gramTrainTrain);
	if (SYEVRM_expected == SYEVRN) {
		FREE(ISUPPZ);
	}
	FREE(work);
	FREE(iwork);

#ifdef __DEBUG__
	PRINTF("Storing eigendecomposition results...\n");
	putVariable(Vr, numSamples1, numDims, "VDbg", matFile);
	putVariable(lvec, numSamples1, 1, "lvecDbg", matFile);
#endif
	
	DOUBLE *Vrtemp = (DOUBLE *) MALLOC(numSamples1 * numDims * sizeof(DOUBLE));
	datacpy(oneVec, lvec, numDims);
	datacpy(Vrtemp, Vr, numSamples1 * numDims);
	for (iterN = 0; iterN < numDims; ++iterN) {
		lvec[iterN] = SQRT(oneVec[numDims - iterN - 1]);
		datacpy(&Vr[iterN * numSamples1], &Vrtemp[(numDims - iterN - 1) * numSamples1], numSamples1);
	}
	FREE(Vrtemp);
	plhs[0] = mxCreateNumericMatrix(numDims, numSamples1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *X1reduced = (DOUBLE*) mxGetData(plhs[0]);
	memset((void *) X1reduced, 0, numDims * numSamples1 * sizeof(DOUBLE));

#ifdef __DEBUG__
	PRINTF("Storing reordeding results...\n");
	putVariable(Vr, numSamples1, numDims, "VOrderedDbg", matFile);
	putVariable(lvec, numSamples1, 1, "sqrtLvecOrderedDbg", matFile);
#endif
	
	AXPYN = numSamples1;
	INCX = 1;
	INCY = numDims;
	for(iterM = 0; iterM < numDims; ++iterM) {
		alpha = lvec[iterM];
		AXPY(&AXPYN, &alpha, &Vr[iterM * numSamples1], &INCX, &X1reduced[iterM], &INCY);
	}
	
	for (iterN = 0; iterN < numDims; ++iterN) {
		lvec[iterN] = 1 / lvec[iterN];
	}	
	
	if (numSamples2 != 0) {
		
		DOUBLE *gramTrainTest = (DOUBLE *) MALLOC(numSamples1 * numSamples2 * sizeof(DOUBLE));
		kernel_gram(gramTrainTest, X1, X2, N, numSamples1, numSamples2, kernelType, param1, param2, \
					normMat1, oneVec);

#ifdef __DEBUG__
		PRINTF("Storing gramTrainTestPreprocessed...\n");
		putVariable(lvec, numSamples1, 1, "invSqrtLvecOrderedDbg", matFile);
		putVariable(gramTrainTest, numSamples1, numSamples2, "gramTrainTestDbg", matFile);
#endif

		FREE(normMat1);
		DOUBLE *column_sums_test = (DOUBLE *) MALLOC(numSamples2 * 1 * sizeof(DOUBLE));

		for (iterN = 0; iterN < numSamples2; ++iterN) {
			temp_sum = 0;
			for (iterM = 0; iterM < numSamples1; ++iterM) {
				temp_sum += gramTrainTest[iterN * numSamples1 + iterM];
			}
			column_sums_test[iterN] = temp_sum;
		}

		for (iterN = 0; iterN < numSamples1; ++iterN) {
			oneVec[iterN] = 1;
		}

		AXPYN = numSamples1;
		alpha = -1;
		alpha2 = total_sum;
		INCX = 1;
		INCY = 1;
		for (iterN = 0; iterN < numSamples2; ++iterN) {
			AXPY(&AXPYN, &alpha, column_sums, &INCX, &gramTrainTest[iterN * numSamples1], &INCY);
			alpha3 = - column_sums_test[iterN] + alpha2;
			AXPY(&AXPYN, &alpha3, oneVec, &INCX, &gramTrainTest[iterN * numSamples1], &INCY);
		}

#ifdef __DEBUG__
		PRINTF("Storing gramTrainTestPostprocessed...\n");
		putVariable(column_sums_test, numSamples2, 1, "columnSumsTestDbg", matFile);
		putVariable(gramTrainTest, numSamples1, numSamples2, "gramTrainTestPostDbg", matFile);
#endif

		FREE(column_sums);
		FREE(column_sums_test);
		FREE(oneVec);

		plhs[1] = mxCreateNumericMatrix(numDims, numSamples2, MXPRECISION_CLASS, mxREAL);
		DOUBLE *X2reduced = (DOUBLE*) mxGetData(plhs[1]);

		INT SCALN = numSamples1;
		INCX = 1;
		for(iterM = 0; iterM < numDims; ++iterM) {
			alpha = lvec[iterM];
			SCAL(&SCALN, &alpha, &Vr[iterM * numSamples1], &INCX);
		}

		FREE(lvec);

#ifdef __DEBUG__
		PRINTF("Storing intermediate result for test mapping...\n");
		putVariable(Vr, numSamples1, numDims, "invSqrtLVOrderedDbg", matFile);
#endif

		CHAR transa = 'T';
		CHAR transb = 'N';
		INT GEMMM = numDims;
		INT GEMMN = numSamples2;
		INT GEMMK = numSamples1;
		alpha = 1;
		INT GEMMLDA = numSamples1;
		DOUBLE beta = 0;
		INT GEMMLDB = numSamples1;
		INT GEMMLDC = numDims;
		GEMM(&transa, &transb, &GEMMM, &GEMMN, &GEMMK, &alpha, Vr, &GEMMLDA, gramTrainTest, &GEMMLDB, &beta, X2reduced, &GEMMLDC);

		FREE(gramTrainTest);
		FREE(Vr);

#ifdef __DEBUG__
		if (matClose(matFile) != 0) {
			PRINTF("Error closing file %s\n", fileName);
			exit(EXIT_FAILURE);
		}
#endif
	} else {
		plhs[1] = mxCreateNumericMatrix(numDims, numSamples2, MXPRECISION_CLASS, mxREAL);
		
		FREE(normMat1);
		FREE(column_sums);
		FREE(oneVec);
		FREE(lvec);
		FREE(Vr);
	}
}	
