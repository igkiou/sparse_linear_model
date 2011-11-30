function [Xr normValue reducedRank] = nuclear_proximal(X, tau, rankIncrement, rankEstimate)

if (nargin < 3),
	rankIncrement = -1;
end;

if (nargin < 4),
	rankEstimate = 10;
end;

[M N] = size(X);
minMN = min(M, N);
if (rankIncrement < 0),
	[U, S, V] = svd(X, 'econ');
else
	foundRank = false;
	while (~foundRank),
		[U, S] = lansvd(C, rankEstimate, 'L');
		foundRank = (S(end) <= tau) || (rankEstimate == minMN);
		rankEstimate = min(rankEstimate + rankIncrement, minMN);
	end;
end;

if (~isreal(S)),
	warning('Matrix has complex singular values with maximum imaginary part %g. Clipping imaginary parts.', max(abs(imag(S(:)))));
	S = real(S);
end;

threshSVD = max(diag(S) - tau, 0);
nonzeroInds = threshSVD > 0;

reducedRank = sum(nonzeroInds);
Xr = U(:, nonzeroInds) * diag(threshSVD(nonzeroInds)) * V(:, nonzeroInds)';
normValue = sum(threshSVD(nonzeroInds));

% function [Xr normValue] = nuclear_proximal(X, tau)
% 
% [U S V] = svd(X, 'econ');
% threshSVD = diag(S) - tau;
% inds = threshSVD > 0;
% Xr = U(:, inds) * diag(threshSVD(inds)) * V(:, inds)';
% normValue = sum(threshSVD(inds));
