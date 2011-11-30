# PRODUCE_REPORTS = 1 
# USE_GUIDE = 1
# GENERATE_PROFILE = 1 
# USE_PROFILE = 1
# DEBUG_MODE = 1
USE_CUDA = 1

# USE_GCC = 1
ifdef USE_GCC
	include gcc.mk
else
	include icc.mk
endif

ROOTDIR = $(shell pwd)
INCLUDEDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src
ifdef USE_CUDA
	CUDASRCDIR = $(ROOTDIR)/cuda
endif
MEXDIR = $(ROOTDIR)/mexfiles
BINDIR = $(ROOTDIR)/bin
INCLUDES += -I$(INCLUDEDIR)

all: basis epca group_lasso_proximal kernel_learning mahalanobis matrix \
	matrix_completion matrix_dictionary metric_learning pca sparse_featuresign \
	sparse_proximal svm tools

basis: \
	$(MEXDIR)/basis_exp_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/basis_kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/dual_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/l2exp_learn_basis_gradient_projection_backtracking_mex.$(MEXEXT) \
	$(MEXDIR)/l2kernel_learn_basis_conj_mex.$(MEXEXT) \
	$(MEXDIR)/l2ls_learn_basis_dual_mex.$(MEXEXT)

epca: \
	$(MEXDIR)/exp_irls_mex.$(MEXEXT) \
	$(MEXDIR)/l2exp_learn_basis_gradient_backtracking_mex.$(MEXEXT)

exponential: \
	$(MEXDIR)/basis_exp_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/l1exp_featuresign_mex.$(MEXEXT) \
	$(MEXDIR)/l1exp_obj_subgrad_mex.$(MEXEXT) \
	$(MEXDIR)/l2exp_learn_basis_gradient_backtracking_mex.$(MEXEXT) \
	$(MEXDIR)/l2exp_learn_basis_gradient_projection_backtracking_mex.$(MEXEXT) \
	$(MEXDIR)/link_func_dual_mex.$(MEXEXT) \
	$(MEXDIR)/link_func_mex.$(MEXEXT)

gaussian: \
	$(MEXDIR)/dual_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/l1qp_featuresign_mex.$(MEXEXT) \
	$(MEXDIR)/l2ls_learn_basis_dual_mex.$(MEXEXT)
	
group_lasso_proximal: \
	$(MEXDIR)/group_lasso_fista_mex.$(MEXEXT) \
	$(MEXDIR)/group_lasso_ista_mex.$(MEXEXT) \
	$(MEXDIR)/group_lasso_proximal_mex.$(MEXEXT)

kernel: \
	$(MEXDIR)/basis_kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/kernel_ball_projection_mex.$(MEXEXT) \
	$(MEXDIR)/kernel_distance_mex.$(MEXEXT) \
	$(MEXDIR)/kernel_gram_mex.$(MEXEXT) \
	$(MEXDIR)/kernel_pca_mex.$(MEXEXT) \
	$(MEXDIR)/kernel_sphere_projection_mex.$(MEXEXT) \
	$(MEXDIR)/l1kernel_featuresign_mex.$(MEXEXT) \
	$(MEXDIR)/l2kernel_learn_basis_conj_mex.$(MEXEXT)
	
kernel_learning: \
	$(MEXDIR)/nrkl_apg_mex.$(MEXEXT) \
	$(MEXDIR)/nrkl_fpc_mex.$(MEXEXT) \
	$(MEXDIR)/nrkl_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/nrkl_svp_mex.$(MEXEXT) \
	$(MEXDIR)/nrkl_svt_mex.$(MEXEXT) 

mahalanobis: \
	$(MEXDIR)/mahalanobis_ml_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/mahalanobis_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/mahalanobis_unweighted_obj_grad_mex.$(MEXEXT)

ifdef USE_CUDA
matrix: \
	$(MEXDIR)/matrix_psd_projection_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_approx_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_hard_thresholding_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_proximal_cuda_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_proximal_gesdd_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_proximal_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_psd_hard_thresholding_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_psd_proximal_mex.$(MEXEXT)
else
matrix: \
	$(MEXDIR)/matrix_psd_projection_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_approx_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_hard_thresholding_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_proximal_gesdd_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_proximal_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_psd_hard_thresholding_mex.$(MEXEXT) \
	$(MEXDIR)/nuclear_psd_proximal_mex.$(MEXEXT)
endif

matrix_completion: \
	$(MEXDIR)/matrix_completion_apg_mex.$(MEXEXT) \
	$(MEXDIR)/operator_completion_apg_mex.$(MEXEXT)

ifdef USE_CUDA	
matrix_dictionary: \
	$(MEXDIR)/matrix_dictionary_hard_thresholding_mex.$(MEXEXT) \
	$(MEXDIR)/matrix_dictionary_intermediate_mex.$(MEXEXT) \
	$(MEXDIR)/matrix_dictionary_kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/matrix_dictionary_learning_lowrank_apg_cuda_mex.$(MEXEXT) \
	$(MEXDIR)/matrix_dictionary_learning_lowrank_apg_mex.$(MEXEXT) \
	$(MEXDIR)/matrix_dictionary_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/operator_dictionary_learning_lowrank_weighted_apg_mex.$(MEXEXT)
else
matrix_dictionary: \
	$(MEXDIR)/matrix_dictionary_hard_thresholding_mex.$(MEXEXT) \
	$(MEXDIR)/matrix_dictionary_intermediate_mex.$(MEXEXT) \
	$(MEXDIR)/matrix_dictionary_kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/matrix_dictionary_learning_lowrank_apg_mex.$(MEXEXT) \
	$(MEXDIR)/matrix_dictionary_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/operator_dictionary_learning_lowrank_weighted_apg_mex.$(MEXEXT)
endif
	
metric_learning: \
	$(MEXDIR)/itml_mex.$(MEXEXT) \
	$(MEXDIR)/lego_mex.$(MEXEXT) \
	$(MEXDIR)/nrml_apg_mex.$(MEXEXT) \
	$(MEXDIR)/nrml_fpc_mex.$(MEXEXT)

ifdef USE_CUDA
pca: \
	$(MEXDIR)/kernel_pca_mex.$(MEXEXT) \
	$(MEXDIR)/robust_operator_pca_apg_mex.$(MEXEXT) \
	$(MEXDIR)/robust_pca_apg_cuda_mex.$(MEXEXT) \
	$(MEXDIR)/robust_pca_apg_gesdd_mex.$(MEXEXT) \
	$(MEXDIR)/robust_pca_apg_mex.$(MEXEXT) \
	$(MEXDIR)/robust_weighted_operator_pca_apg_cuda_mex.$(MEXEXT) \
	$(MEXDIR)/robust_weighted_operator_pca_apg_gesdd_mex.$(MEXEXT) \
	$(MEXDIR)/robust_weighted_operator_pca_apg_mex.$(MEXEXT)
else
pca: \
	$(MEXDIR)/kernel_pca_mex.$(MEXEXT) \
	$(MEXDIR)/robust_operator_pca_apg_mex.$(MEXEXT) \
	$(MEXDIR)/robust_pca_apg_gesdd_mex.$(MEXEXT) \
	$(MEXDIR)/robust_pca_apg_mex.$(MEXEXT) \
	$(MEXDIR)/robust_weighted_operator_pca_apg_gesdd_mex.$(MEXEXT) \
	$(MEXDIR)/robust_weighted_operator_pca_apg_mex.$(MEXEXT)
endif

sparse_featuresign: \
	$(MEXDIR)/l1exp_featuresign_mex.$(MEXEXT) \
	$(MEXDIR)/l1exp_obj_subgrad_mex.$(MEXEXT) \
	$(MEXDIR)/l1kernel_featuresign_mex.$(MEXEXT) \
	$(MEXDIR)/l1qp_featuresign_mex.$(MEXEXT)

ifdef USE_CUDA
sparse_proximal: \
	$(MEXDIR)/kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/l1_proximal_cuda_mex.$(MEXEXT) \
	$(MEXDIR)/l1_proximal_mex.$(MEXEXT) \
	$(MEXDIR)/l1qp_fista_mex.$(MEXEXT) \
	$(MEXDIR)/l1qp_ista_mex.$(MEXEXT) \
	$(MEXDIR)/qp_lipschitz_mex.$(MEXEXT) \
	$(MEXDIR)/qp_obj_grad_mex.$(MEXEXT)
else
sparse_proximal: \
	$(MEXDIR)/kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/l1_proximal_mex.$(MEXEXT) \
	$(MEXDIR)/l1qp_fista_mex.$(MEXEXT) \
	$(MEXDIR)/l1qp_ista_mex.$(MEXEXT) \
	$(MEXDIR)/qp_lipschitz_mex.$(MEXEXT) \
	$(MEXDIR)/qp_obj_grad_mex.$(MEXEXT)
endif
	
svm: \
	$(MEXDIR)/abs_smooth_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/cramersinger_approx_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/cramersinger_nuclear_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/cramersinger_frobenius_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/huberhinge_kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/huberhinge_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/multihuberhinge_kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/multisquaredhinge_kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/pegasos_binary_svm_mex.$(MEXEXT) \
	$(MEXDIR)/pegasos_multiclass_svm_mex.$(MEXEXT) \
	$(MEXDIR)/squaredhinge_kernel_obj_grad_mex.$(MEXEXT) \
	$(MEXDIR)/squaredhinge_obj_grad_mex.$(MEXEXT)
	
tools: \
	$(MEXDIR)/kernel_ball_projection_mex.$(MEXEXT) \
	$(MEXDIR)/kernel_distance_mex.$(MEXEXT) \
	$(MEXDIR)/kernel_gram_mex.$(MEXEXT) \
	$(MEXDIR)/kernel_sphere_projection_mex.$(MEXEXT) \
	$(MEXDIR)/l1_distance_mex.$(MEXEXT) \
	$(MEXDIR)/l2_ball_projection_mex.$(MEXEXT) \
	$(MEXDIR)/l2_distance_mex.$(MEXEXT) \
	$(MEXDIR)/l2_sphere_projection_mex.$(MEXEXT) \
	$(MEXDIR)/link_func_dual_mex.$(MEXEXT) \
	$(MEXDIR)/link_func_mex.$(MEXEXT) \
	$(MEXDIR)/mahalanobis_distance_mex.$(MEXEXT) 
	
temp: \
	$(MEXDIR)/temp.$(MEXEXT) \
	$(MEXDIR)/temp1.$(MEXEXT) \
	$(MEXDIR)/temp2.$(MEXEXT) \
	$(MEXDIR)/temp3.$(MEXEXT)

# mex executable files

ifdef USE_CUDA
$(MEXDIR)/abs_smooth_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/abs_smooth_obj_grad_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/abs_smooth_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/abs_smooth_obj_grad_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

$(MEXDIR)/basis_exp_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/basis_exp_obj_grad_mex.o \
	$(SRCDIR)/l2exp_learn_basis.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/projection.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/basis_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/basis_kernel_obj_grad_mex.o \
	$(SRCDIR)/l2kernel_learn_basis.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_gram.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

ifdef USE_CUDA
$(MEXDIR)/cramersinger_approx_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/cramersinger_approx_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/cramersinger_approx_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/cramersinger_approx_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/cramersinger_frobenius_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/cramersinger_frobenius_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/cramersinger_frobenius_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/cramersinger_frobenius_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
		
ifdef USE_CUDA
$(MEXDIR)/cramersinger_nuclear_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/cramersinger_nuclear_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/cramersinger_nuclear_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/cramersinger_nuclear_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

$(MEXDIR)/dual_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/dual_obj_grad_mex.o \
	$(SRCDIR)/l2ls_learn_basis.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/exp_irls_mex.$(MEXEXT): \
	$(MEXDIR)/exp_irls_mex.o \
	$(SRCDIR)/l1_featuresign.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

ifdef USE_CUDA
$(MEXDIR)/group_lasso_fista_mex.$(MEXEXT): \
	$(MEXDIR)/group_lasso_fista_mex.o \
	$(SRCDIR)/group_lasso_proximal.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/group_lasso_fista_mex.$(MEXEXT): \
	$(MEXDIR)/group_lasso_fista_mex.o \
	$(SRCDIR)/group_lasso_proximal.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/group_lasso_ista_mex.$(MEXEXT): \
	$(MEXDIR)/group_lasso_ista_mex.o \
	$(SRCDIR)/group_lasso_proximal.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/group_lasso_ista_mex.$(MEXEXT): \
	$(MEXDIR)/group_lasso_ista_mex.o \
	$(SRCDIR)/group_lasso_proximal.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/group_lasso_proximal_mex.$(MEXEXT): \
	$(MEXDIR)/group_lasso_proximal_mex.o \
	$(SRCDIR)/group_lasso_proximal.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/group_lasso_proximal_mex.$(MEXEXT): \
	$(MEXDIR)/group_lasso_proximal_mex.o \
	$(SRCDIR)/group_lasso_proximal.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/huberhinge_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/huberhinge_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/huberhinge_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/huberhinge_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/huberhinge_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/huberhinge_kernel_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/huberhinge_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/huberhinge_kernel_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/itml_mex.$(MEXEXT): \
	$(MEXDIR)/itml_mex.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/itml_mex.$(MEXEXT): \
	$(MEXDIR)/itml_mex.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

$(MEXDIR)/kernel_ball_projection_mex.$(MEXEXT): \
	$(MEXDIR)/kernel_ball_projection_mex.o \
	$(SRCDIR)/projection.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/kernel_gram_mex.$(MEXEXT): \
	$(MEXDIR)/kernel_gram_mex.o \
	$(SRCDIR)/kernel_gram.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/kernel_distance_mex.$(MEXEXT): \
	$(MEXDIR)/kernel_distance_mex.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

ifdef USE_CUDA
$(MEXDIR)/kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/kernel_obj_grad_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/kernel_obj_grad_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

$(MEXDIR)/kernel_pca_mex.$(MEXEXT): \
	$(MEXDIR)/kernel_pca_mex.o \
	$(SRCDIR)/kernel_gram.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/kernel_sphere_projection_mex.$(MEXEXT): \
	$(MEXDIR)/kernel_sphere_projection_mex.o \
	$(SRCDIR)/projection.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l1_distance_mex.$(MEXEXT): \
	$(MEXDIR)/l1_distance_mex.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

ifdef USE_CUDA
$(MEXDIR)/l1_proximal_cuda_mex.$(MEXEXT): \
	$(MEXDIR)/l1_proximal_cuda_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/l1_proximal_mex.$(MEXEXT): \
	$(MEXDIR)/l1_proximal_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/l1_proximal_mex.$(MEXEXT): \
	$(MEXDIR)/l1_proximal_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

$(MEXDIR)/l1exp_featuresign_mex.$(MEXEXT): \
	$(MEXDIR)/l1exp_featuresign_mex.o \
	$(SRCDIR)/l1_featuresign.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l1exp_obj_subgrad_mex.$(MEXEXT): \
	$(MEXDIR)/l1exp_obj_subgrad_mex.o \
	$(SRCDIR)/l1_featuresign.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l1kernel_featuresign_mex.$(MEXEXT): \
	$(MEXDIR)/l1kernel_featuresign_mex.o \
	$(SRCDIR)/l1_featuresign.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l1qp_featuresign_mex.$(MEXEXT): \
	$(MEXDIR)/l1qp_featuresign_mex.o \
	$(SRCDIR)/l1_featuresign.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
	
ifdef USE_CUDA
$(MEXDIR)/l1qp_fista_mex.$(MEXEXT): \
	$(MEXDIR)/l1qp_fista_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/l1qp_fista_mex.$(MEXEXT): \
	$(MEXDIR)/l1qp_fista_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/l1qp_ista_mex.$(MEXEXT): \
	$(MEXDIR)/l1qp_ista_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/l1qp_ista_mex.$(MEXEXT): \
	$(MEXDIR)/l1qp_ista_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

$(MEXDIR)/l2_ball_projection_mex.$(MEXEXT): \
	$(MEXDIR)/l2_ball_projection_mex.o \
	$(SRCDIR)/projection.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l2_distance_mex.$(MEXEXT): \
	$(MEXDIR)/l2_distance_mex.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l2exp_learn_basis_gradient_backtracking_mex.$(MEXEXT): \
	$(MEXDIR)/l2exp_learn_basis_gradient_backtracking_mex.o \
	$(SRCDIR)/l2exp_learn_basis.o \
	$(SRCDIR)/projection.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/utils.o \
	$(SRCDIR)/useblas.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l2exp_learn_basis_gradient_projection_backtracking_mex.$(MEXEXT): \
	$(MEXDIR)/l2exp_learn_basis_gradient_projection_backtracking_mex.o \
	$(SRCDIR)/l2exp_learn_basis.o \
	$(SRCDIR)/projection.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/utils.o \
	$(SRCDIR)/useblas.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l2kernel_learn_basis_conj_mex.$(MEXEXT): \
	$(MEXDIR)/l2kernel_learn_basis_conj_mex.o \
	$(SRCDIR)/l2kernel_learn_basis.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_gram.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l2ls_learn_basis_dual_mex.$(MEXEXT): \
	$(MEXDIR)/l2ls_learn_basis_dual_mex.o \
	$(SRCDIR)/l2ls_learn_basis.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/l2_sphere_projection_mex.$(MEXEXT): \
	$(MEXDIR)/l2_sphere_projection_mex.o \
	$(SRCDIR)/projection.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
	
ifdef USE_CUDA
$(MEXDIR)/lego_mex.$(MEXEXT): \
	$(MEXDIR)/lego_mex.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/lego_mex.$(MEXEXT): \
	$(MEXDIR)/lego_mex.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

$(MEXDIR)/link_func_dual_mex.$(MEXEXT): \
	$(MEXDIR)/link_func_dual_mex.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/link_func_mex.$(MEXEXT): \
	$(MEXDIR)/link_func_mex.o \
	$(SRCDIR)/exponential.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/mahalanobis_distance_mex.$(MEXEXT): \
	$(MEXDIR)/mahalanobis_distance_mex.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/mahalanobis_ml_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/mahalanobis_ml_obj_grad_mex.o \
	$(SRCDIR)/learn_mahalanobis.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/mahalanobis_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/mahalanobis_obj_grad_mex.o \
	$(SRCDIR)/learn_mahalanobis.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/mahalanobis_unweighted_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/mahalanobis_unweighted_obj_grad_mex.o \
	$(SRCDIR)/learn_mahalanobis.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

ifdef USE_CUDA
$(MEXDIR)/matrix_completion_apg_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_completion_apg_mex.o \
	$(SRCDIR)/matrix_completion.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/matrix_completion_apg_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_completion_apg_mex.o \
	$(SRCDIR)/matrix_completion.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/matrix_dictionary_hard_thresholding_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_hard_thresholding_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/matrix_dictionary_hard_thresholding_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_hard_thresholding_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

$(MEXDIR)/matrix_dictionary_intermediate_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_intermediate_mex.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

ifdef USE_CUDA
$(MEXDIR)/matrix_dictionary_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_kernel_obj_grad_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/matrix_dictionary_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_kernel_obj_grad_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/matrix_dictionary_learning_lowrank_apg_cuda_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_learning_lowrank_apg_cuda_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/matrix_dictionary_learning_lowrank_apg_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_learning_lowrank_apg_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/matrix_dictionary_learning_lowrank_apg_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_learning_lowrank_apg_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/matrix_dictionary_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_obj_grad_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/matrix_dictionary_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_dictionary_obj_grad_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/matrix_psd_projection_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_psd_projection_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/matrix_psd_projection_mex.$(MEXEXT): \
	$(MEXDIR)/matrix_psd_projection_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/multihuberhinge_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/multihuberhinge_kernel_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/multihuberhinge_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/multihuberhinge_kernel_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/multisquaredhinge_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/multisquaredhinge_kernel_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/multisquaredhinge_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/multisquaredhinge_kernel_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nrkl_apg_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_apg_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nrkl_apg_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_apg_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nrkl_fpc_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_fpc_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nrkl_fpc_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_fpc_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nrkl_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_obj_grad_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nrkl_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_obj_grad_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nrkl_svp_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_svp_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nrkl_svp_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_svp_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nrkl_svt_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_svt_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nrkl_svt_mex.$(MEXEXT): \
	$(MEXDIR)/nrkl_svt_mex.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/nrml_apg_mex.$(MEXEXT): \
	$(MEXDIR)/nrml_apg_mex.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nrml_apg_mex.$(MEXEXT): \
	$(MEXDIR)/nrml_apg_mex.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nrml_fpc_mex.$(MEXEXT): \
	$(MEXDIR)/nrml_fpc_mex.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nrml_fpc_mex.$(MEXEXT): \
	$(MEXDIR)/nrml_fpc_mex.o \
	$(SRCDIR)/metric_learning.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/kernel_learning.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nuclear_approx_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_approx_obj_grad_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nuclear_approx_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_approx_obj_grad_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nuclear_hard_thresholding_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_hard_thresholding_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nuclear_hard_thresholding_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_hard_thresholding_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nuclear_proximal_cuda_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_proximal_cuda_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nuclear_proximal_gesdd_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_proximal_gesdd_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nuclear_proximal_gesdd_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_proximal_gesdd_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/nuclear_proximal_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_proximal_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nuclear_proximal_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_proximal_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/nuclear_psd_hard_thresholding_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_psd_hard_thresholding_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nuclear_psd_hard_thresholding_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_psd_hard_thresholding_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/nuclear_psd_proximal_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_psd_proximal_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/nuclear_psd_proximal_mex.$(MEXEXT): \
	$(MEXDIR)/nuclear_psd_proximal_mex.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/operator_completion_apg_mex.$(MEXEXT): \
	$(MEXDIR)/operator_completion_apg_mex.o \
	$(SRCDIR)/matrix_completion.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/operator_completion_apg_mex.$(MEXEXT): \
	$(MEXDIR)/operator_completion_apg_mex.o \
	$(SRCDIR)/matrix_completion.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/operator_dictionary_learning_lowrank_weighted_apg_mex.$(MEXEXT): \
	$(MEXDIR)/operator_dictionary_learning_lowrank_weighted_apg_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/operator_dictionary_learning_lowrank_weighted_apg_mex.$(MEXEXT): \
	$(MEXDIR)/operator_dictionary_learning_lowrank_weighted_apg_mex.o \
	$(SRCDIR)/matrix_dictionary.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/pegasos_binary_svm_mex.$(MEXEXT): \
	$(MEXDIR)/pegasos_binary_svm_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/pegasos_binary_svm_mex.$(MEXEXT): \
	$(MEXDIR)/pegasos_binary_svm_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/pegasos_multiclass_svm_mex.$(MEXEXT): \
	$(MEXDIR)/pegasos_multiclass_svm_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/pegasos_multiclass_svm_mex.$(MEXEXT): \
	$(MEXDIR)/pegasos_multiclass_svm_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/qp_lipschitz_mex.$(MEXEXT): \
	$(MEXDIR)/qp_lipschitz_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/qp_lipschitz_mex.$(MEXEXT): \
	$(MEXDIR)/qp_lipschitz_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/qp_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/qp_obj_grad_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/qp_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/qp_obj_grad_mex.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/robust_operator_pca_apg_mex.$(MEXEXT): \
	$(MEXDIR)/robust_operator_pca_apg_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/robust_operator_pca_apg_mex.$(MEXEXT): \
	$(MEXDIR)/robust_operator_pca_apg_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/robust_pca_apg_cuda_mex.$(MEXEXT): \
	$(MEXDIR)/robust_pca_apg_cuda_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/robust_pca_apg_mex.$(MEXEXT): \
	$(MEXDIR)/robust_pca_apg_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/robust_pca_apg_mex.$(MEXEXT): \
	$(MEXDIR)/robust_pca_apg_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/robust_pca_apg_gesdd_mex.$(MEXEXT): \
	$(MEXDIR)/robust_pca_apg_gesdd_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/robust_pca_apg_gesdd_mex.$(MEXEXT): \
	$(MEXDIR)/robust_pca_apg_gesdd_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/robust_weighted_operator_pca_apg_cuda_mex.$(MEXEXT): \
	$(MEXDIR)/robust_weighted_operator_pca_apg_cuda_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/robust_weighted_operator_pca_apg_gesdd_mex.$(MEXEXT): \
	$(MEXDIR)/robust_weighted_operator_pca_apg_gesdd_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/robust_weighted_operator_pca_apg_gesdd_mex.$(MEXEXT): \
	$(MEXDIR)/robust_weighted_operator_pca_apg_gesdd_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/robust_weighted_operator_pca_apg_mex.$(MEXEXT): \
	$(MEXDIR)/robust_weighted_operator_pca_apg_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/robust_weighted_operator_pca_apg_mex.$(MEXEXT): \
	$(MEXDIR)/robust_weighted_operator_pca_apg_mex.o \
	$(SRCDIR)/pca.o \
	$(SRCDIR)/l1_proximal.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

ifdef USE_CUDA
$(MEXDIR)/squaredhinge_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/squaredhinge_kernel_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/squaredhinge_kernel_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/squaredhinge_kernel_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif
	
ifdef USE_CUDA
$(MEXDIR)/squaredhinge_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/squaredhinge_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/usecuda.o \
	$(SRCDIR)/utils.o \
	$(CUDASRCDIR)/cudakernels.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
else
$(MEXDIR)/squaredhinge_obj_grad_mex.$(MEXEXT): \
	$(MEXDIR)/squaredhinge_obj_grad_mex.o \
	$(SRCDIR)/svm_optimization.o \
	$(SRCDIR)/matrix_proximal.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^
endif

# temp mex executable 
$(MEXDIR)/temp.$(MEXEXT): \
	$(MEXDIR)/temp.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/temp1.$(MEXEXT): \
	$(MEXDIR)/temp1.o \
	$(SRCDIR)/distance.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/temp2.$(MEXEXT): \
	$(MEXDIR)/temp2.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/temp3.$(MEXEXT): \
	$(MEXDIR)/temp3.o \
	$(SRCDIR)/useblas.o \
	$(SRCDIR)/utils.o
	$(CC) $(LDFLAGS) $(LIBS) $(CFLAGS) -o $@ $^

# mex object files
$(MEXDIR)/%.o: $(MEXDIR)/%.cpp $(INCLUDEDIR)/sparse_classification.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(MEXDIR)/$*.cpp

# src object files
$(SRCDIR)/group_lasso_proximal.o: \
	$(SRCDIR)/group_lasso_proximal.cpp \
	$(INCLUDEDIR)/group_lasso_proximal.h \
	$(INCLUDEDIR)/l1_proximal.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
	
$(SRCDIR)/kernel_gram.o: \
	$(SRCDIR)/kernel_gram.cpp \
	$(INCLUDEDIR)/kernel_gram.h \
	$(INCLUDEDIR)/distance.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
	
$(SRCDIR)/kernel_learning.o: \
	$(SRCDIR)/kernel_learning.cpp \
	$(INCLUDEDIR)/kernel_learning.h \
	$(INCLUDEDIR)/matrix_proximal.h \
	$(INCLUDEDIR)/metric_learning.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
	
$(SRCDIR)/ksvd.o: \
	$(SRCDIR)/ksvd.cpp \
	$(INCLUDEDIR)/ksvd.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
	
$(SRCDIR)/l1_featuresign.o: \
	$(SRCDIR)/l1_featuresign.cpp \
	$(INCLUDEDIR)/l1_featuresign.h \
	$(INCLUDEDIR)/exponential.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp

ifdef USE_CUDA
$(SRCDIR)/l1_proximal.o: \
	$(SRCDIR)/l1_proximal.cpp \
	$(INCLUDEDIR)/l1_proximal.h \
	$(INCLUDEDIR)/cudakernels.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/usecuda.h \
	$(INCLUDEDIR)/useinterfaces.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp	
else
$(SRCDIR)/l1_proximal.o: \
	$(SRCDIR)/l1_proximal.cpp \
	$(INCLUDEDIR)/l1_proximal.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
endif
	
$(SRCDIR)/l2exp_learn_basis.o: \
	$(SRCDIR)/l2exp_learn_basis.cpp \
	$(INCLUDEDIR)/l2exp_learn_basis.h \
	$(INCLUDEDIR)/projection.h \
	$(INCLUDEDIR)/exponential.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
		
$(SRCDIR)/l2kernel_learn_basis.o: \
	$(SRCDIR)/l2kernel_learn_basis.cpp \
	$(INCLUDEDIR)/l2kernel_learn_basis.h \
	$(INCLUDEDIR)/kernel_gram.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp

$(SRCDIR)/learn_sensing.o: \
	$(SRCDIR)/learn_sensing.cpp \
	$(INCLUDEDIR)/learn_sensing.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp

$(SRCDIR)/matrix_dictionary.o: \
	$(SRCDIR)/matrix_dictionary.cpp \
	$(INCLUDEDIR)/matrix_dictionary.h \
	$(INCLUDEDIR)/l1_proximal.h \
	$(INCLUDEDIR)/matrix_proximal.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp

ifdef USE_CUDA
$(SRCDIR)/matrix_proximal.o: \
	$(SRCDIR)/matrix_proximal.cpp \
	$(INCLUDEDIR)/matrix_proximal.h \
	$(INCLUDEDIR)/cudakernels.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/usecuda.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
else
$(SRCDIR)/matrix_proximal.o: \
	$(SRCDIR)/matrix_proximal.cpp \
	$(INCLUDEDIR)/matrix_proximal.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
endif

$(SRCDIR)/metric_learning.o: \
	$(SRCDIR)/metric_learning.cpp \
	$(INCLUDEDIR)/metric_learning.h \
	$(INCLUDEDIR)/distance.h \
	$(INCLUDEDIR)/kernel_learning.h \
	$(INCLUDEDIR)/matrix_proximal.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp

ifdef USE_CUDA
$(SRCDIR)/pca.o: \
	$(SRCDIR)/pca.cpp \
	$(INCLUDEDIR)/pca.h \
	$(INCLUDEDIR)/cudakernels.h \
	$(INCLUDEDIR)/l1_proximal.h \
	$(INCLUDEDIR)/matrix_proximal.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/usecuda.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
else
$(SRCDIR)/pca.o: \
	$(SRCDIR)/pca.cpp \
	$(INCLUDEDIR)/pca.h \
	$(INCLUDEDIR)/l1_proximal.h \
	$(INCLUDEDIR)/matrix_proximal.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp
endif

$(SRCDIR)/svm_optimization.o: \
	$(SRCDIR)/svm_optimization.cpp \
	$(INCLUDEDIR)/svm_optimization.h \
	$(INCLUDEDIR)/matrix_proximal.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $*.cpp

$(SRCDIR)/%.o: \
	$(SRCDIR)/%.cpp \
	$(INCLUDEDIR)/%.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(SRCDIR)/$*.cpp
	
$(CUDASRCDIR)/cudakernels.o: \
	$(CUDASRCDIR)/cudakernels.cu \
	$(INCLUDEDIR)/cudakernels.h \
	$(INCLUDEDIR)/useblas.h \
	$(INCLUDEDIR)/usecuda.h \
	$(INCLUDEDIR)/useinterfaces.h \
	$(INCLUDEDIR)/utils.h
	$(NVCC) $(NVCFLAGS) $(INCLUDES) -c -o $@ $*.cu

clean:	
	rm -rf *.o *~
	rm -rf $(MEXDIR)/*.o $(MEXDIR)/*~
	rm -rf $(SRCDIR)/*.o $(SRCDIR)/*~

distclean:	
	rm -rf *.o *~
	rm -rf $(MEXDIR)/*.o $(MEXDIR)/*.$(MEXEXT) $(MEXDIR)/*~
	rm -rf $(SRCDIR)/*.o $(SRCDIR)/*~
ifdef USE_CUDA
	rm -rf $(CUDASRCDIR)/*.o $(CUDASRCDIR)/*~
endif
