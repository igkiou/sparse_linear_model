/*
 * useinterface.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __USE_INTERFACES_H__
#define __USE_INTERFACES_H__

#ifdef USE_MATLAB_INTERFACE

#include "mex.h"
#include "matrix.h"
#ifdef USE_DOUBLE_PRECISION
	#define MXPRECISION_CLASS mxDOUBLE_CLASS
#elif defined(USE_SINGLE_PRECISION)
	#define MXPRECISION_CLASS mxSINGLE_CLASS
#endif

inline void ERROR(const char *error_msg) {
	mexErrMsgTxt(error_msg);
}

#define PRINTF mexPrintf

/*
 * TODO: Consider replacing with mkl_malloc, or good old malloc.
 */
inline void *MALLOC(size_t n) {
	void *buffer = mxMalloc(n);
	if (buffer == NULL) {
		ERROR("MALLOC error.");
	}
	return buffer;
}

inline void *CMALLOC(size_t n) {
	void *buffer = malloc(n);
	if (buffer == NULL) {
		ERROR("MALLOC error.");
	}
	return buffer;
}

inline void FREE(void *ptr) {
	mxFree(ptr);
}

inline void CFREE(void *ptr) {
	free(ptr);
}

inline bool ISINF(double x) {
	return mxIsInf(x);
}

inline bool ISNAN(double x) {
	return mxIsNaN(x);
}

#elif defined(USE_STANDALONE_INTERFACE)

#ifdef USE_DOUBLE_PRECISION
#define MXPRECISION_CLASS mxDOUBLE_CLASS
#elif defined(USE_SINGLE_PRECISION)
#define MXPRECISION_CLASS mxSINGLE_CLASS
#endif

inline void ERROR(const char *error_msg) {
	fprintf(stderr, error_msg);
}

#define PRINTF printf

inline void *MALLOC(size_t n) {
	void *buffer = malloc(n);
	if (buffer == NULL) {
		ERROR("MALLOC error.");
	}
	return buffer;
}

inline void *CMALLOC(size_t n) {
	void *buffer = malloc(n);
	if (buffer == NULL) {
		ERROR("MALLOC error.");
	}
	return buffer;
}

inline void FREE(void *ptr) {
	free(ptr);
}

inline void CFREE(void *ptr) {
	free(ptr);
}

inline bool ISINF(double x) {
	return isinf(x);
}

inline bool ISNAN(double x) {
	return isnan(x);
}

#endif

#endif /* __USE_INTERFACES_H__ */
