function K = kernel_distance(K, sqrtFlag)
% Kernel norm distance matrix
%   distanceMatrix = kernel_distance(K, sqrtFlag)
%   K:				numSamples x numSamples Grammian matrix.
%	sqrtFlag:		if 0, calculate squared Euclidean distance (default 0).
%   distanceMatrix: numSamples x numSamples.

if (nargin < 1),
	error('Not enough input arguments'); %#ok
end;

if (nargin < 2),
	sqrtFlag = 0;
end

% TODO: Change this so that only upper triangular part is calculated, and
% the rest is reflected.
% TODO: Optimize to as much as possible avoid matrix replication due to
% calls by value.
% TODO: Compare with dist_euclid.m
	
Dsq = diag(K);
K = bsxfun(@plus, Dsq, bsxfun(@minus, Dsq', 2 * K));
if (sqrtFlag == 1),
	K = sqrt(K);
	K = real(K);
end;
K = max(K, K');
K = K - diag(diag(K));
