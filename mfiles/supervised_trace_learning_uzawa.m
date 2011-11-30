function [M z] = supervised_trace_learning_uzawa(deltaX, b, y, tau, zinit, trainX, testX, trainL, testL)

DELTA = 1.2 * 1000;
MAXITER = 1000;
delta(1:MAXITER) = DELTA;
numPairs = size(deltaX, 2);
if ((nargin >= 5) && ~isempty(zinit)),
	if ((size(zinit, 1) == size(deltaX, 1))...
			&& (size(zinit, 2) == size(deltaX, 1))),
		distances = diag(deltaX' * zinit * deltaX)';
		violations = max((distances - b) .* y, 0);
		z = - DELTA * violations;
	elseif ((size(zinit, 2) == numPairs) && (size(zinit, 1) == 1)),
		z = zinit;
		violations = max(-b .* y, 0);
	else
		error('Improper dimensions for initialization.');
	end
else
	violations = max(-b .* y, 0);
	z = - DELTA * violations;
end;
activeInds = find(z < 0);
normViolations = norm(violations);
M = zeros(size(deltaX, 1));
iter = 0;
while iter < MAXITER,
	iter = iter + 1;
	disp(sprintf('iter = %d, obj = %g, violations %g', iter, 0.5 * norm(M, 'fro') ^ 2 + tau * trace(M), normViolations));
	sumZiCi = deltaX(:, activeInds) * diag(z(activeInds) .* y(activeInds)) * deltaX(:, activeInds)' / numPairs;
	sumZiCi = (sumZiCi + sumZiCi') / 2;
	[V D] = eig(sumZiCi);
	eigVec = diag(D);
	M = V * diag(max(eigVec - tau, 0)) * V';
	distances = diag(deltaX' * M * deltaX)';
	violations = max((distances - b) .* y, 0);
	z = z - delta(iter) * violations;
	activeInds = find(z < 0);
	normViolations = norm(violations);
	if ((normViolations < eps)),
		break;
	end;
	if ((~mod(iter, 20)) && (nargin == 9)),
		sqM = sqrtm(M);
		sqM = real(sqM);
		[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(sqM', trainX', testX');
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testL', trainL');
		disp(sprintf('rank = %d, accuracy = %g', rank(M), accuracy));
	end;
end;
