function Phi = low_rank_approximation(M, m)

option = struct('disp', 0);
[eigvector, eigvalue] = eigs(M, m, 'la', option);
eigvector = eigvector * sqrt(eigvalue);
eigvalue = diag(eigvalue);
maxEigValue = max(abs(eigvalue));
eigIdx = abs(eigvalue) / maxEigValue < 1e-12;
eigvalue(eigIdx) = [];
if (length(eigvalue) < m),
	warning('Mahalanobis matrix has rank less than the number of measurements.');
end;
Phi = eigvector';
