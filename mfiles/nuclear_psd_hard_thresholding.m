function [Xr normValue] = nuclear_psd_hard_thresholding(X, rank, partial)

if (nargin < 3),
	partial = 0;
end;

if (partial == 1),
	X = (X + X')  / 2;
	opts.disp = 0;
	opts.isreal = 1;
	opts.issym = 1;
	[U S] = eigs(X, rank, 'LA', opts);
	if (~isreal(S)),
		warning('Matrix has complex eigenvalues with maximum imaginary part %g. Clipping imaginary parts.', max(abs(imag(S(:)))));
		S = real(S);
	end;
	threshEig = diag(S);
	inds = threshEig > 0;
	Xr = U(:, inds) * diag(threshEig(inds)) * U(:, inds)';
	normValue = sum(threshEig(inds));
else
	X = (X + X')  / 2;
	[U S] = safeEig(X);
	if (~isreal(S)),
		warning('Matrix has complex eigenvalues with maximum imaginary part %g. Clipping imaginary parts.', max(abs(imag(S(:)))));
		S = real(S);
	end;
	threshEig = diag(S);
	threshEig((rank + 1):end) = 0;
	inds = 1:rank;
	Xr = U(:, inds) * diag(max(threshEig(inds), 0)) * U(:, inds)';
	normValue = sum(threshEig(inds));
end;


