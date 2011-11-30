function [obj deriv] = basis_kernel_obj_grad(D, X, A, kernelType, varargin)

if (nargin < 4),
	kernelType = 'G';
end;

if (nargin < 5),
	s = 1;
else
	s = varargin{1};
end;

[N numSamples] = size(X);
K = size(A, 1);

D = reshape(D, [N K]);

KDD = kernel_gram(D, [], kernelType, varargin{:});
KXD = kernel_gram(X, D, kernelType, varargin{:});

obj = trace(- 2 * KXD * A + A' * KDD * A);

if (nargout > 1)
	deriv = zeros(N, K);
	for iterK = 1:K,
		k = iterK;
		dk = D(:, k);
		KDd = KDD(:, k);
		KXd = KXD(:, k);
		ak = A(k, :);
		mult = ak * (A' * KDd - KXd);
		rvec = (D * bsxfun(@times, KDd, A) - bsxfun(@times, KXd', X)) * ak';
		deriv(:, k) = - 2 / s ^ 2 * (mult * dk - rvec);
	end;
	deriv = deriv(:);
end;

% 		fact2=A(1,:)*transpose(A)*Kd1;
% 		fact4 = A(1,:)*Xd1;
% 		fact6 = D*(repmat(Kd1,[1 2]).*A)*transpose(A(1,:));
% 		fact8 = repmat(transpose(Xd1),[2 1]).*X*transpose(A(1,:));
%		deriv_col = - 2 / s ^ 2 * (mult * dk - rvec);
