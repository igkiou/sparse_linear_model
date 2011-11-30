function [Xr normValue reducedRank] = nuclear_psd_proximal_truncate(X, tau, rankIncrement, rankEstimate, truncateFlag, gap)

if (nargin < 3),
	rankIncrement = -1;
end;

if (nargin < 4),
	rankEstimate = 10;
end;

if (nargin < 5),
	truncateFlag = 0;
end;

if (nargin < 6),
	gap = 5;
end;

M = size(X, 1);
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

if (truncateFlag == 1),
	q = sum(nonzeroInds);
	xj = zeros(q - 1, 1);
	xjsgap = q;
	for j = (q - 1):(- 1):1,
		xj(j) = mean(threshEig(nonzeroInds(1:j))) / mean(threshEig(nonzeroInds((j + 1):q)));
		if (xj >= gap),
			xjsgap = j;
		end;
	end;
	nonzeroInds((xjsgap + 1):q) = false;
end;
reducedRank = sum(nonzeroInds);
Xr = U(:, nonzeroInds) * diag(threshEig(nonzeroInds)) * U(:, nonzeroInds)';
normValue = sum(threshEig(nonzeroInds));
