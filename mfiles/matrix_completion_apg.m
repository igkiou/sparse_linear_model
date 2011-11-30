function [B iter] = matrix_completion_apg(M, N, observationMat, mu, gamma,...
								tolerance, delta, numIters, eta)

if ((nargin < 4) || isempty(mu)),
	dummy = zeros(M, N);
	inds = sub2ind([M N], observationMat(:, 1), observationMat(:, 2));
	dummy(inds) = observationMat(:, 3);
	l = safeEig(dummy' * dummy);
	mu = 0.99 * 0.00001 * sqrt(l(1));
end;

if ((nargin < 5) || isempty(gamma)),
	gamma = 0;
end;

if ((nargin < 6) || isempty(tolerance)),
	tolerance = 10 ^ - 6;
end;

if ((nargin < 7) || isempty(delta)),
	delta = 10 ^ 5;
end;

if ((nargin < 8) || isempty(numIters)),
	numIters = 50000;
end;

if ((nargin < 9) || isempty(eta)),
	eta = 0.9;
end;

kappa = gamma * mu;

dummy = zeros(M, N);
inds = sub2ind([M N], observationMat(:, 1), observationMat(:, 2));
vals = observationMat(:, 3);
dummy(inds) = observationMat(:,3);

B = zeros(M, N);
Bkm1 = zeros(M, N);
YB = zeros(M, N);
G = zeros(M, N);
S = zeros(M, N);

Lf = 1.0;
invLf = 1.0 / Lf;

tk = 1;
tkm1 = 1;
muk = mu * delta;
iter = 0;

while (1),
	YB = (tk + tkm1 - 1) / tk * B - (tkm1 - 1) / tk * Bkm1;

	G(:) = 0;
	G(inds) = YB(inds) - vals;
		
	Bkm1 = B;
	B = (1 - kappa * invLf) * YB - invLf * G;
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
% 	if (mod(iter, 100) == 0),
% 		disp(iter);
% 	end;

	if (iter == numIters),
		break;
	end;
end;
