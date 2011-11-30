#ifndef __SPARSE_CLASSIFICATION_H__
#define __SPARSE_CLASSIFICATION_H__

#include "useinterfaces.h"
#include "useblas.h"
#include "utils.h"
#ifdef USE_CUDA
#include "usecuda.h"
#endif

/*
 * TODO: Recompile with profiling information.
 * TODO: Create matlab independent C wrappers that support handling of mat
 * files, and profile with gprof/VMTune.
 * TODO: Add documentation with .m files for _mex files, and with doxygen
 * comments for C files.
 * TODO: In all _mex files, change prhs and plhs references to #define-d
 * variable names, both when getting PrData and when getting dimensions.
 * TODO: Make sure than, in all non-mex functions, no data is copied if not
 * necessary, and instead the result is overwritten on the original memory.
 * Adjust calls by _mex files appropriately.
 * TODO: Remove all derivFlags, and change obj_grad files to be doing checks of
 * whether obj or deriv are NULL instead.
 * TODO: Remove all inclusions from mexfiles, except for
 * sparse_classification.h.
 * TODO: Rename this file and project into something different.
 */

#include "projection.h"
#ifdef USE_CUDA
#include "cudakernels.h"
#endif
#include "distance.h"
#include "group_lasso_proximal.h"
#include "kernel_gram.h"
#include "kernel_learning.h"
#include "ksvd.h"
#include "l1_featuresign.h"
#include "l1_proximal.h"
#include "l2exp_learn_basis.h"
#include "l2kernel_learn_basis.h"
#include "l2ls_learn_basis.h"
#include "learn_mahalanobis.h"
#include "learn_sensing.h"
#include "learn_sensing_svm.h"
#include "exponential.h"
#include "matrix_completion.h"
#include "matrix_dictionary.h"
#include "metric_learning.h"
#include "matrix_proximal.h"
#include "pca.h"
#include "svm_optimization.h"

#endif /* __SPARSE_CLASSIFICATION_H__ */
