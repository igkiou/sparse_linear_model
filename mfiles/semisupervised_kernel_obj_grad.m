function [obj deriv] = semisupervised_kernel_obj_grad(kernelMatrix, W, X, Y, lambda, classLabels,...
		D, DDt, mu1, mu2, regularizationFlag)
	
if (nargout <= 1),
	obj_sup = multihuberhinge_kernel_obj_grad(kernelMatrix, W, X, Y,...
		lambda, classLabels, regularizationFlag);
	obj_unsup = mahalanobis_unweighted_obj_grad(kernelMatrix, D, DDt);
	obj = mu1 * obj_sup + mu2 * obj_unsup;
else
	[obj_sup, deriv_sup] = multihuberhinge_kernel_obj_grad(kernelMatrix, W, X, Y,...
			lambda, classLabels, regularizationFlag);
	[obj_unsup deriv_unsup] = mahalanobis_unweighted_obj_grad(kernelMatrix, D, DDt);

	obj = mu1 * obj_sup + mu2 * obj_unsup;
	deriv = mu1 * deriv_sup + mu2 * deriv_unsup;
end;
