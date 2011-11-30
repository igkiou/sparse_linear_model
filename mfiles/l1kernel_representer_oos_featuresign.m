function S = l1kernel_representer_oos_featuresign(KXY, KXX, D, lambda, beta)

if ((nargin < 5) || (isempty(beta))),
	beta = 1e-4;
end;

[numSamplesTrain K] = size(D);
numSamples = size(KXY, 2);

% sparse codes of the features
S = sparse(K, numSamples);

KDD = D' * KXX * D;
% KDD = kernel_gram(D, [], kernelType, varargin{:});
A = double(KDD + 2*beta*eye(size(D, 2)));

KDY = D' * KXY;
% KDX = kernel_gram(D, X, kernelType, varargin{:});
bAll = - KDY;
for ii = 1:numSamples,
    b = bAll(:, ii);
	S(:, ii) = l1qp_featuresign_sub(A, b, lambda);
end
