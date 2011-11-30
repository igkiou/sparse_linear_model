function [B A iter] = robust_pca_apg(D, mu, lambda, gamma, tolerance, delta,...
								numIters, eta)

if ((nargin < 2) || isempty(mu)),
	l = safeEig(D' * D);
	mu = 0.99 * 0.00001 * sqrt(l(1));
end;

if ((nargin < 3) || isempty(lambda)),
	lambda = 1 / sqrt(max(size(D)));
end;

if ((nargin < 4) || isempty(gamma)),
	gamma = 0;
end;

if ((nargin < 5) || isempty(tolerance)),
	tolerance = 10 ^ - 6;
end;

if ((nargin < 6) || isempty(delta)),
	delta = 10 ^ 5;
end;

if ((nargin < 7) || isempty(numIters)),
	numIters = 50000;
end;

if ((nargin < 8) || isempty(eta)),
	eta = 0.9;
end;

kappa = gamma * mu;

[M N] = size(D);
B = zeros(M, N);
Bkm1 = zeros(M, N);
YB = zeros(M, N);
A = zeros(M, N);
Akm1 = zeros(M, N);
YA = zeros(M, N);
BAmD = zeros(M, N);
S = zeros(M, N);

Dt = [eye(M), sqrt(kappa) * eye(M); eye(M), zeros(M)];
v = eig(Dt' * Dt);
Lf = maxv(v(:));
invLf = 1 / Lf;

tk = 1;
tkm1 = 1;
muk = mu * delta;
iter = 0;

while (1),
	YB = (tk + tkm1 - 1) / tk * B - (tkm1 - 1) / tk * Bkm1;
	YA = (tk + tkm1 - 1) / tk * A - (tkm1 - 1) / tk * Akm1;
	BAmD = YB + YA - D;
	
	Akm1 = A;
	A = YA - invLf * BAmD;
	S = Lf * A;
	A = l1_proximal(A, lambda * muk * invLf);
	S = S - Lf * A;
	normsum = norm(S, 'fro');
	
	Bkm1 = B;
	B = (1 - kappa * invLf) * YB - invLf * BAmD;
	S = Lf * B;
	B = nuclear_proximal(B, muk * invLf);
	S = S - Lf * B;
	normsum = normsum + norm(S, 'fro');
	
	tkm1 = tk;
	tk = (1.0 + sqrt(4.0 * tkm1 ^ 2 + 1)) / 2.0;
	muk = max(eta * muk, mu);
	
	if (normsum < tolerance),
		break;
	end;

	iter = iter + 1;

	if (iter == numIters),
		break;
	end;
end;
