/*
 * usestandalone.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __USE_STANDALONE_H__
#define __USE_STANDALONE_H__

/* Standalone declarations */
#ifdef USE_DOUBLE_PRECISION
#define MXPRECISION_CLASS mxDOUBLE_CLASS
#elif defined(USE_SINGLE_PRECISION)
#define MXPRECISION_CLASS mxSINGLE_CLASS
#endif
#define MALLOC malloc
#define FREE free
#define ISINF isinf
#define ISNAN isnan
#define PRINTF printf
#define ERROR(msg) fprintf(stderr, msg)

#endif /* __USE_STANDALONE_H__ */
