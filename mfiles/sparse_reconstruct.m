function Y = sparse_reconstruct(X, Phi, D, method, param)

if (nargin < 3),
	method = 'omp';
end;

if (nargin < 4),
	if (strcmp(method, 'omp') || strcmp(method, 'komp')),
		param.L = 6;
		param.eps = 0.0001;
	elseif (strcmp(method, 'lasso')),
		param.lambda = 0.0001;
		param.mode = 1;
	end;
end;

vec=sqrt(sum((Phi*D).^2));
W = Phi * X;
if (strcmp(method, 'omp')),
	alphaCoeffs = mexOMP(W, normcols(Phi*D), param);
elseif (strcmp(method, 'lasso')),
	alphaCoeffs = mexLasso(W, normcols(Phi*D), param);
elseif (strcmp(method, 'komp')),
	Deq = normcols(Phi*D);
	G = Deq'*Deq;
	alphaCoeffs = omp(Deq'*W, G, param.L);
end;
Y = D * diag(1./vec) * alphaCoeffs;
