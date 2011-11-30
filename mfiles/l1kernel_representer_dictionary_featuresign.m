function S = l1kernel_representer_dictionary_featuresign(KXX, D, lambda, beta)

if ((nargin < 4) || (isempty(beta))),
	beta = 1e-4;
end;

[numSamples K] = size(D);

% sparse codes of the features
S = sparse(K, numSamples);

KDD = D' * KXX * D;
% KDD = kernel_gram(D, [], kernelType, varargin{:});
A = double(KDD + 2*beta*eye(size(D, 2)));

KDX = D' * KXX; 
% KDX = kernel_gram(D, X, kernelType, varargin{:});
bAll = - KDX;
for ii = 1:numSamples,
    b = bAll(:, ii);
	S(:, ii) = l1qp_featuresign_sub(A, b, lambda);
end
