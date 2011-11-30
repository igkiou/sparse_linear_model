#ifndef __USE_MATLAB_H__
#define __USE_MATLAB_H__

#include "mex.h"
#include "matrix.h"

/* MATLAB declarations */
#ifdef USE_DOUBLE_PRECISION
#define MXPRECISION_CLASS mxDOUBLE_CLASS
#elif defined(USE_SINGLE_PRECISION)
#define MXPRECISION_CLASS mxSINGLE_CLASS
#endif
#define MALLOC mxMalloc
#define FREE mxFree
#define ISINF mxIsInf
#define ISNAN mxIsNaN
#define PRINTF mexPrintf
#define ERROR mexErrMsgTxt

#endif /* __USE_MATLAB_H__ */
