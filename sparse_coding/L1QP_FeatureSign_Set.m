function [S] = L1QP_FeatureSign_Set(X, B, gamma, beta)

if (nargin < 4),
	beta = 1e-4;
end;

[dFea, nSmp] = size(X);
nBases = size(B, 2);

% sparse codes of the features
S = sparse(nBases, nSmp);

A = double(B'*B + 2*beta*eye(size(B, 2)));
bAll = -B' * X;
% disp(sprintf('NumSamples: %d\n', nSmp));
for ii = 1:nSmp,
% 	disp(sprintf('Now runnig sample: %d\n', ii));
    b = bAll(:, ii);
%     [net] = L1QP_FeatureSign(gamma, A, b);
    S(:, ii) = L1QP_FeatureSign_yang(gamma, A, b);
end
