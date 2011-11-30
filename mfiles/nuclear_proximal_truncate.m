function [Xr normValue reducedRank] = nuclear_proximal_truncate(X, tau, rankIncrement, rankEstimate, truncateFlag, gap)

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

[M N] = size(X, 1);
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

if (truncateFlag == 1),
	q = sum(nonzeroInds);
	xj = zeros(q - 1, 1);
	xjsgap = q;
	for j = (q - 1):(- 1):1,
		xj(j) = mean(threshSVD(nonzeroInds(1:j))) / mean(threshSVD(nonzeroInds((j + 1):q)));
		if (xj >= gap),
			xjsgap = j;
		end;
	end;
	nonzeroInds((xjsgap + 1):q) = false;
end;
reducedRank = sum(nonzeroInds);
Xr = U(:, nonzeroInds) * diag(threshSVD(nonzeroInds)) * V(:, nonzeroInds)';
normValue = sum(threshSVD(nonzeroInds));
