function D = l2kernel_learn_basis_conj(X, A, D, numSearches, kernelType, varargin)
% 		fact2=A(1,:)*transpose(A)*Kd1;
% 		fact4 = A(1,:)*Xd1;
% 		fact6 = D*(repmat(Kd1,[1 2]).*A)*transpose(A(1,:));
% 		fact8 = repmat(transpose(Xd1),[2 1]).*X*transpose(A(1,:));

if (nargin < 5),
	kernelType = 'G';
elseif ((kernelType ~= 'G') && (kernelType ~= 'g')),
	error('Kernel dictionary learning only implemented for Gaussian kernel.');
end;

D = minimize(D(:), @basis_kernel_obj_grad, numSearches, 0, X, A, kernelType, varargin{:});
