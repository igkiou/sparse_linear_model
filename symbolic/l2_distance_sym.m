function distanceMatrix = l2_distance_sym(X1, X2, sqrtFlag)
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
elseif (size(X2, 1) ~= size(X1, 1)),
	error('The signal dimension (first dimension) of the second sample matrix does not match the signal dimension (first dimension) of the first sample matrix.'); %#ok
end;

if (nargin < 3),
	sqrtFlag = 0;
end

% TODO: Change this so that only upper triangular part is calculated, and
% the rest is reflected.
if (oneArgFlag == 1),
	if (size(X1, 1) == 1),
		X1 = [X1; zeros(1, size(X1, 2))]; 
	end;
	X1sq = sum(X1 .* X1, 1).';
% 	distanceMatrix = bsxfun(@plus, X1sq, bsxfun(@minus, X1sq', 2 * (X1' * X1)));
	numSamples1 = size(X1, 2);
	distanceMatrix = repmat(X1sq, [1 numSamples1]) + repmat(X1sq.', [numSamples1 1]) - 2 * (X1.' * X1);
	if (sqrtFlag == 1),
		distanceMatrix = sqrt(distanceMatrix);
		distanceMatrix = real(distanceMatrix);
	end;
	distanceMatrix = max(distanceMatrix, distanceMatrix.');
	distanceMatrix = distanceMatrix - diag(diag(distanceMatrix));

else
	if (size(X1, 1) == 1),
		X1 = [X1; zeros(1, size(X1, 2))]; 
		X2 = [X2; zeros(1, size(X2, 2))]; 
	end;
% 	distanceMatrix = bsxfun(@plus, sum(X1 .* X1, 1)', bsxfun(@minus, sum(X2.* X2, 1), 2 * X1' * X2));
	numSamples1 = size(X1, 2);
	numSamples2 = size(X2, 2);
	distanceMatrix = repmat(sum(X1 .* X1, 1).', [1 numSamples2]) + repmat(sum(X2 .* X2, 1), [numSamples1 1]) - 2 * X1.' * X2;
	if (sqrtFlag == 1),
		distanceMatrix = sqrt(distanceMatrix);
		distanceMatrix = real(distanceMatrix);
	end;
end;
