function Xr = matrix_psd_projection(X)

X = (X + X') / 2;
[U, S] = safeEig(X);

if (~isreal(S)),
	warning('Matrix has complex eigenvalues with maximum imaginary part %g. Clipping imaginary parts.', max(abs(imag(S(:)))));
	S = real(S);
end;

threshEig = max(diag(S), 0);
nonzeroInds = threshEig > 0;
% printf('rank %d', reducedRank);
Xr = U(:, nonzeroInds) * diag(threshEig(nonzeroInds)) * U(:, nonzeroInds)';
