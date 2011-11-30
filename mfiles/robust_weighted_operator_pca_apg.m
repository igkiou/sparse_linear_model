function [B A iter] = robust_weighted_operator_pca_apg(D, Y, W, mu, lambda,...
							gamma, tolerance, delta, numIters, eta)

if ((nargin < 4) || isempty(mu)),
	l = safeEig(D' * D);
	mu = 0.99 * 0.00001 * sqrt(l(1));
end;

if ((nargin < 5) || isempty(lambda)),
	lambda = 1 / sqrt(max(size(D)));
end;

if ((nargin < 6) || isempty(gamma)),
	gamma = 0;
end;

if ((nargin < 7) || isempty(tolerance)),
	tolerance = 10 ^ - 6;
end;

if ((nargin < 8) || isempty(delta)),
	delta = 100000;
end;

if ((nargin < 9) || isempty(numIters)),
	numIters = 50000;
end;

if ((nargin < 10) || isempty(eta)),
	eta = 0.9;
end;

kappa = gamma * mu;

[M N] = size(D);
K = size(Y, 2);
B = zeros(M, K);
Bkm1 = zeros(M, K);
YB = zeros(M, K);
A = zeros(M, N);
Akm1 = zeros(M, N);
YA = zeros(M, N);
BAmD = zeros(M, N);
S = zeros(M, N);

Dt = [Y'*W sqrt(kappa) * eye(N); W zeros(N)];
v = eig(Dt' * Dt);
Lf = maxv(v(:));
invLf = 1 / Lf;

Wsq = W' * W;
WsqY = Wsq * Y;

tk = 1;
tkm1 = 1;
muk = mu * delta;
iter = 0;

while (1),
	YB = (tk + tkm1 - 1) / tk * B - (tkm1 - 1) / tk * Bkm1;
	YA = (tk + tkm1 - 1) / tk * A - (tkm1 - 1) / tk * Akm1;
	BAmD = YB * Y' + YA - D;
	
	Akm1 = A;
	A = YA - invLf * BAmD * Wsq;
	S = Lf * A;
	A = l1_proximal(A, invLf * lambda * muk);
	S = S - Lf * A;
	normsum = norm(S, 'fro');
	
	Bkm1 = B;
	B = (1.0 - kappa * invLf) * YB - invLf * BAmD * WsqY;
	S = Lf * B;
	B = nuclear_proximal_mex(B, invLf * muk);
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
