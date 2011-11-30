function distanceMatrix = mahalanobis_distance(X1, X2, M, sqrtFlag)
% Euclidean distance matrix
%   distanceMatrix = l2_distance(X1, X2, sqrtFlag)
%   X1:				N x numSamples1.
%   X2:				N x numSamples2 (optional).
%	M:				N x N weight matrix.
%	sqrtFlag:		if 0, calculate squared Euclidean distance (default 0).
%   distanceMatrix: numSamples1 x numSamples2 or numSamples1 x numSamples1.

% TODO: Also write a mex version of this script.

if (nargin < 1),
	error('Not enough input arguments'); %#ok
end;

oneArgFlag = 0;
if ((nargin < 2) || isempty(X2)),
	oneArgFlag = 1;
	X2 = [];
elseif (size(X2, 1) ~= size(X1, 1)),
	error('The signal dimension (first dimension) of the second sample matrix does not match the signal dimension (first dimension) of the first sample matrix.'); %#ok
end;

if (nargin < 4),
	sqrtFlag = 0;
end

% TODO: Change this so that only upper triangular part is calculated, and
% the rest is reflected.
% TODO: Optimize to as much as possible avoid matrix replication due to
% calls by value.
% TODO: Compare with dist_euclid.m
% TODO: Implement without the need for chol.
U = chol(M);
if (oneArgFlag == 1),
	X1 = U * X1;
	distanceMatrix = l2_distance(X1, [], sqrtFlag);
else
	X1 = U * X1;
	X2 = U * X2;
	distanceMatrix = l2_distance(X1, X2, sqrtFlag);
end;
