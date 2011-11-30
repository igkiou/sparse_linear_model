function kernelMatrix = kernel_agp(W, X, Y, D, DDt, lambda,...
		mu1, mu2, classLabels, regularizationFlag, lambdatrace, Kin, varargin)

N = size(X, 1);

if ((nargin < 12) || (isempty(Kin))),
	Kin = zeros(N, N);
end;

proxfunc = @(s, L) nuclear_psd_proximal(s, L);
objgradfunc = @(s) semisupervised_kernel_obj_grad_wrapper(s, W, X, Y, lambda, classLabels,...
		D, DDt, mu1, mu2, regularizationFlag);
    kernelMatrix = apg(Kin, lambdatrace, objgradfunc, proxfunc, varargin{:});

end

function [obj deriv] = semisupervised_kernel_obj_grad_wrapper(kernelMatrix, W, X, Y, lambda, classLabels,...
		D, DDt, mu1, mu2, regularizationFlag)

if (nargout <= 1),
	[obj deriv] = semisupervised_kernel_obj_grad(kernelMatrix, W, X, Y, lambda, classLabels,...
		D, DDt, mu1, mu2, regularizationFlag);
	deriv = reshape(deriv, [size(W, 1) size(W, 1)]);
else
	obj = semisupervised_kernel_obj_grad(kernelMatrix, W, X, Y, lambda, classLabels,...
		D, DDt, mu1, mu2, regularizationFlag);
end;

end
