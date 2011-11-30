function S = l1exp_featuresign(X, D, lambda, family, beta)

if (nargin < 4),
	family = 'P';
end;

if (nargin < 5),
	beta = 1e-4;
end;

[N, numSamples] = size(X);
K = size(D, 2);

BETA = 0.9;
ALPHA = 0.3;
EPS = 10 ^ - 12;
MAXITER = 500;
MAXITEROUT = 200;

% sparse codes of the features
S = zeros(K, numSamples);
% s = zeros(K, 1);
for ii = 1:numSamples,
	s = zeros(K, 1);
	x = X(:, ii);
	Ds = zeros(N, 1);
	[aVal aPrime aDoublePrime] = link_func(Ds, family);
	objVal = aVal;
	iterOut = 0;
	% TODO: Extract this part and create l1exp_featuresign_sub.m.
	while (1)
% 		LMatrixSqrt = diag(sqrt(aDoublePrime));
% 		LMatrixSqrtInv = diag(safeReciprocal(sqrt(aDoublePrime), 0));
% 		xtilde = LMatrixSqrtInv * (x - aPrime) + LMatrixSqrt * Ds;
% 		Dtilde = LMatrixSqrt * D;
% 		A = double(Dtilde' * Dtilde + 2 * beta * eye(size(D, 2)));
% 		b = - Dtilde' * xtilde;
		
		LMatrix = diag(aDoublePrime);
		xtilde = x - aPrime + LMatrix * Ds;
		A = double(D' * LMatrix * D + 2 * beta * eye(size(D, 2)));
		b = - D' * xtilde;
		
		shat = l1qp_featuresign_sub(A, b, lambda, s);

		preObjVal = objVal - lambda * sum(abs(s));
		stp = shat - s;
		t = 1;
		p = stp' * D' * (aPrime - x);
		for iterBT = 1:MAXITER,
			snew = s + t * stp;
			Ds = D * snew;
			aVal = link_func(Ds, family);
			postObjVal = - Ds' * x + aVal;
			if (postObjVal < preObjVal + ALPHA * t * p),
				break;
			else  
				t = BETA * t;
			end;
		end;

		s = snew;
		[aVal aPrime aDoublePrime] = link_func(Ds, family);
		objValOld = objVal;
		objVal = postObjVal + lambda * sum(abs(s));
		if (abs(objValOld - objVal) < EPS),
			break;
		end;
		iterOut = iterOut + 1;
		if (iterOut > MAXITEROUT),
			break;
		end;
	end;
	S(:, ii) = s;
end;
