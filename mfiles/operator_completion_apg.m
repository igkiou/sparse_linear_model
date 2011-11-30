function [B iter] = operator_completion_apg(M, N, observationMat, Y, mu,...
								gamma, tolerance, delta, numIters, eta)

if ((nargin < 5) || isempty(mu)),
	dummy = zeros(M, N);
	inds = sub2ind([M N], observationMat(:, 1), observationMat(:, 2));
	dummy(inds) = observationMat(:, 3);
	l = safeEig(dummy' * dummy);
	mu = 0.99 * 0.00001 * sqrt(l(1));
end;

if ((nargin < 6) || isempty(gamma)),
	gamma = 0;
end;

if ((nargin < 7) || isempty(tolerance)),
	tolerance = 10 ^ - 6;
end;

if ((nargin < 8) || isempty(delta)),
	delta = 10 ^ 5;
end;

if ((nargin < 9) || isempty(numIters)),
	numIters = 50000;
end;

if ((nargin < 10) || isempty(eta)),
	eta = 0.9;
end;

kappa = gamma * mu;

inds = sub2ind([M N], observationMat(:, 1), observationMat(:, 2));
vals = observationMat(:, 3);

B = zeros(M, N);
Bkm1 = zeros(M, N);
YB = zeros(M, N);
YBYt = zeros(M, N);
G = zeros(M, N);
S = zeros(M, N);

v = eig(Y' * Y);
Lf = maxv(v(:));
invLf = 1 / Lf;

tk = 1;
tkm1 = 1;
muk = mu * delta;
iter = 0;

while (1),
	YB = (tk + tkm1 - 1) / tk * B - (tkm1 - 1) / tk * Bkm1;
	YBYt = YB * Y';
	
	G(:) = 0;
	G(inds) = YBYt(inds) - vals;
		
	Bkm1 = B;
	B = (1 - kappa * invLf) * YB - invLf * G * Y;
	S = Lf * B;
	B = nuclear_proximal(B, muk * invLf);
	S = S - Lf * B;
	normsum = norm(S, 'fro');
	
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
