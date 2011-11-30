/*
 * cudakernels.h
 *
 *  Created on: Nov 7, 2011
 *      Author: igkiou
 */

#ifndef __CUDAKERNELS_H__
#define __CUDAKERNELS_H__

#include "usecuda.h"

void cuSoftThreshold(CUDOUBLE *X, CUDOUBLE tau, CUINT N);

#endif /* __CUDAKERNELS_H__ */
