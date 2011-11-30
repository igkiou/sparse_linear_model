function [Xr normValue reducedRank] = nuclear_psd_proximal(X, tau, rankIncrement, rankEstimate)

if (nargin < 3),
	rankIncrement = -1;
end;

if (nargin < 4),
	rankEstimate = 10;
end;

M = size(X, 1);
X = (X + X') / 2;
if (rankIncrement < 0),
	[U, S] = safeEig(X);
else
	foundRank = false;
	while (~foundRank),
		[U, S] = laneig(C, rankEstimate, 'LA');
		foundRank = (S(end) <= tau) || (rankEstimate == M);
		rankEstimate = min(rankEstimate + rankIncrement, M);
	end;
end;

if (~isreal(S)),
	warning('Matrix has complex eigenvalues with maximum imaginary part %g. Clipping imaginary parts.', max(abs(imag(S(:)))));
	S = real(S);
end;

threshEig = max(diag(S) - tau, 0);
nonzeroInds = threshEig > 0;
reducedRank = sum(nonzeroInds);
% printf('rank %d', reducedRank);
Xr = U(:, nonzeroInds) * diag(threshEig(nonzeroInds)) * U(:, nonzeroInds)';
normValue = sum(threshEig(nonzeroInds));
