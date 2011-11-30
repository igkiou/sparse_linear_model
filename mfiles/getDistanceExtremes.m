function [l, u, j1, j2] = getDistanceExtremes(X, a, b, M, numTrials)
% [l, u, j1, j2] = getDistanceExtremes(X, a, b, M)
%
% Computes sample histogram of the distances between columns of X and returns
% the value of these distances at the a^th and b^th percentiles.  This
% method is used to determine the upper and lower bounds for
% similarity / dissimilarity constraints.  
%
% X: (n x s) data matrix, n signal dimension and s number of samples
% a: lower bound percentile between 1 and 100
% b: upper bound percentile between 1 and 100
% M: Mahalanobis matrix to compute distances (optional, default identity
% matrix)
%
% Returns l: distance corresponding to a^th percentile
% u: distance corresponding the b^th percentile
% j1, j2: indices of sampled pairs used to calculate distance extremes

if ((nargin < 4) || isempty(M)),
	M = eye(size(X, 1));
end;

if (a < 1 || a > 100),
    error('a must be between 1 and 100')
end
if (b < 1 || b > 100),
    error('b must be between 1 and 100')
end

n = size(X, 2);

if (nargin < 5),
	numTrials = max(size(X, 2), 1000);
	numTrials = min(numTrials, n * (n - 1) / 2);
end;

% we will sample with replacement
j1 = ceil(rand(1, numTrials) * n);
j2 = ceil(rand(1, numTrials) * n);
deltaX = X(:, j1) - X(:, j2);
% dists = diag(deltaX' * M * deltaX)';
dists = zeros(numTrials, 1);
for (i=1:numTrials),
    dists(i) = (X(:, j1(i)) - X(:, j2(i)))' * M *(X(:, j1(i)) - X(:, j2(i)));
end

[f, c] = hist(dists, 100);
l = c(floor(a));
u = c(floor(b));
