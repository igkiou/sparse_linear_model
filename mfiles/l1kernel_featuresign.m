function S = l1kernel_featuresign(KDX, KDD, lambda, beta)

if (nargin < 4),
	kernelType = 'G';
end;

if ((nargin < 5) || (isempty(beta))),
	beta = 1e-4;
end;

[K numSamples] = size(KDX);

% sparse codes of the features
S = sparse(K, numSamples);

A = double(KDD + 2*beta*eye(K));
bAll = - KDX;
for ii = 1:numSamples,
    b = bAll(:, ii);
	S(:, ii) = l1qp_featuresign_sub(A, b, lambda);
end
