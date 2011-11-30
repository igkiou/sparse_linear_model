function S = l1qp_featuresign(X, D, lambda, beta)

if (nargin < 4),
	beta = 1e-4;
end;

[N, numSamples] = size(X);
K = size(D, 2);

% sparse codes of the features
S = sparse(K, numSamples);

A = double(D'*D + 2*beta*eye(size(D, 2)));
bAll = -D' * X;
for ii = 1:numSamples,
    b = bAll(:, ii);
    S(:, ii) = l1qp_featuresign_sub(A, b, lambda);
end
