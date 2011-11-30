function [M Mtilde z] = semisupervised_trace_learning_uzawa(deltaX, b, y, D, tau, zinit)

DELTA = 1.2;
MAXITER = 1000;

numAtoms = size(D, 2);
numAtomsSq = numAtoms ^ 2;
numPairs = size(deltaX, 2);
DDt = D * D';
I = eye(numAtoms);
DDtInv = eye(size(D, 1)) / DDt;
DDtInv = (DDtInv + DDtInv') / 2;
DDtInvD = DDtInv * D;
XtDDtInvD = deltaX' * DDtInvD;

delta(1:MAXITER) = DELTA;
if (nargin >= 6),
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
Mtilde = zeros(size(D, 2));
iter = 0;
while iter < MAXITER,
	iter = iter + 1;
	disp(sprintf('iter %d, obj = %g, violations %g', iter,...
		0.5 * norm(I - Mtilde, 'fro') ^ 2 / numAtomsSq + tau * trace(Mtilde),...
		normViolations));
	sumZiCiDDt = XtDDtInvD(:, activeInds)' * diag(z(activeInds) .* y(activeInds)) * XtDDtInvD(:, activeInds) + I; 
	sumZiCiDDt = (sumZiCiDDt + sumZiCiDDt') / 2 / (1 + numPairs);
	[V D] = eig(sumZiCiDDt);
	eigVec = diag(D);
	Mtilde = V * diag(max(eigVec - tau, 0)) * V';
	distances = diag(XtDDtInvD * Mtilde * XtDDtInvD')';
	violations = max((distances - b) .* y, 0);
	z = z - delta(iter) * violations;
	activeInds = find(z < 0);
	normViolations = norm(violations);
	if ((normViolations < eps)),
		break;
	end;
end;

M = DDtInvD * Mtilde * DDtInvD';
