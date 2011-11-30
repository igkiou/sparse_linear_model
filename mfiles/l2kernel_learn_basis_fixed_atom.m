function D = l2kernel_learn_basis_fixed_atom(X, A, D, numIters, kernelType, varargin)
% 		fact2=A(1,:)*transpose(A)*Kd1;
% 		fact4 = A(1,:)*Xd1;
% 		fact6 = D*(repmat(Kd1,[1 2]).*A)*transpose(A(1,:));
% 		fact8 = repmat(transpose(Xd1),[2 1]).*X*transpose(A(1,:));

if (nargin < 5),
	kernelType = 'G';
elseif ((kernelType ~= 'G') && (kernelType ~= 'g')),
	error('Kernel dictionary learning only implemented for Gaussian kernel.');
end;

[N numSamples] = size(X);
K = size(D, 2);

for iter = 1:numIters,
	atomInds = randperm(K); 
	for iterK = 1:K,
		k = atomInds(iterK);
		dk = D(:, k);
		KDd = kernel_gram(D, dk, kernelType, varargin{:});
		KXd = kernel_gram(X, dk, kernelType, varargin{:});
		ak = A(k, :);
		mult = ak * (A' * KDd - KXd);
		if (abs(mult) > 0), 
			rvec = (D * bsxfun(@times, KDd, A) - bsxfun(@times, KXd', X)) * ak';
			D(:, k) = rvec / mult;
		end;
	end;
end;
