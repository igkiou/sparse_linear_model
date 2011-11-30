function [M Mtilde z] = semisupervised_trace_learning_uzawa_eig(deltaX, b, y, D, tau, zinit, trainX, testX, trainL, testL)

DELTA = 1.2 * size(D, 2) ^ 2;
MAXITER = 1000;

numAtoms = size(D, 2);
numAtomsSq = numAtoms ^ 2;
numPairs = size(deltaX, 2);
DDt = D * D';
[V L] = safeEig(DDt);
VL = V * L;
Linv = diag(diag(L) .^ -1);
VLinv = V * Linv;
LinvVtX = VLinv' * deltaX;

delta(1:MAXITER) = DELTA;
if ((nargin >= 6) && ~isempty(zinit)),
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
	z = - DELTA * violations * numAtomsSq;
end;
activeInds = find(z < 0);
normViolations = norm(violations);
Mtilde = zeros(size(D, 1));
iter = 0;
while iter < MAXITER,
	iter = iter + 1;
	disp(sprintf('iter %d, obj = %g, violations %g', iter,...
		0.5 * norm(L - Mtilde, 'fro') ^ 2 / numAtomsSq + tau * trace(Mtilde),...
		normViolations));
	sumZiCiDDt = LinvVtX(:, activeInds) * diag(z(activeInds) .* y(activeInds)) * LinvVtX(:, activeInds)' / numPairs + L / numAtomsSq; 
	sumZiCiDDt = (sumZiCiDDt + sumZiCiDDt') / 2;
	[Vtemp Dtemp] = eig(sumZiCiDDt);
	eigVec = diag(Dtemp);
	Mtilde = Vtemp * diag(max(eigVec - tau, 0)) * Vtemp';
	distances = diag(LinvVtX' * Mtilde * LinvVtX)';
	violations = max((distances - b) .* y, 0);
	z = z - delta(iter) * violations;
	activeInds = find(z < 0);
	normViolations = norm(violations);
	if ((normViolations < eps)),
		break;
	end;
	if ((~mod(iter, 20)) && (nargin == 10)),
		M = VLinv * Mtilde * VLinv';
		sqM = sqrtm(M);
		sqM = real(sqM);
		[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(sqM', trainX', testX');
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testL', trainL');
		disp(sprintf('rank = %d, accuracy = %g', rank(M), accuracy));
	end;
end;

M = VLinv * Mtilde * VLinv';
