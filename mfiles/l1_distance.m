function distanceMatrix = l1_distance(X1, X2)
% Euclidean distance matrix
%   distanceMatrix = l2_distance(X1, X2, sqrtFlag)
%   X1:				N x numSamples1.
%   X2:				N x numSamples2 (optional).
%	sqrtFlag:		if 0, calculate squared Euclidean distance (default 0).
%   distanceMatrix: numSamples1 x numSamples2 or numSamples1 x numSamples1.

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

if (oneArgFlag == 1),
	[N num1] = size(X1);
	distanceMatrix = zeros(num1);
	for iter1 = 1:num1,
		distanceMatrix(iter1, iter1 + 1:end) = sum(abs(bsxfun(@minus, X1(:, iter1), X1(:, iter1 + 1:end))), 1);
	end;
	distanceMatrix = distanceMatrix + distanceMatrix';
else
	[N num1] = size(X1);
	distanceMatrix = zeros(num1, size(X2, 2));
	for iter1 = 1:num1,
		distanceMatrix(iter1, :) = sum(abs(bsxfun(@minus, X1(:, iter1), X2)), 1);
	end;
end;		
