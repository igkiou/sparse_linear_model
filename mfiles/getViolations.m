function [violations, numViolations] = getViolations(K, X, constraintMat, betaVector, type)
%   Constraints are dij - dkl < - 1, beta = - 1, NIPS10
%	types: relational, bounds, targets

if (nargin < 5),
	type = 'relational';
end;

if (~isempty(X)),
	K = X' * K * X;
end;

if (strcmp(type, 'relational')),
	if(size(constraintMat, 2) ~= 4),
		error('CONSTRAINTMAT has the wrong format: second dimension is equal to %d instead of 4.',...
			size(constraintMat, 2));
	end;
	numConstraints = length(betaVector);
	violations = zeros(numConstraints, 1);
	for iterConstraint = 1:numConstraints,
		beta = betaVector(iterConstraint);
		coord1 = constraintMat(iterConstraint, 1);
		coord2 = constraintMat(iterConstraint, 2);
		coord3 = constraintMat(iterConstraint, 3);
		coord4 = constraintMat(iterConstraint, 4);
		dij = K(coord1, coord1) + K(coord2, coord2) - 2 * K(coord1, coord2);
		dkl = K(coord3, coord3) + K(coord4, coord4) - 2 * K(coord3, coord4);
		% dij - dkl < - 1, beta = - 1, NIPS10
		violations(iterConstraint) = max(dij - dkl - beta, 0);
	end;

	if (nargout >= 2),
		numViolations = sum(violations > 0);
	end;
elseif (strcmp(type, 'bounds')),
	if(size(constraintMat, 2) ~= 3),
		error('CONSTRAINTMAT has the wrong format: second dimension is equal to %d instead of 3.',...
			size(constraintMat, 2));
	end;
	numConstraints = length(betaVector);
	violations = zeros(numConstraints, 1);
	for iterConstraint = 1:numConstraints,
		beta = betaVector(iterConstraint);
		coord1 = constraintMat(iterConstraint, 1);
		coord2 = constraintMat(iterConstraint, 2);
		dij = K(coord1, coord1) + K(coord2, coord2) - 2 * K(coord1, coord2);
		class = constraintMat(iterConstraint, 3);
		if (class * (dij - beta) > 0),
			violations(iterCconstraint) = class * (dij - beta);
		end;
	end;

	if (nargout >= 2),
		numViolations = sum(abs(violations) > 0);
	end;
elseif (strcmp(type, 'targets')),
	if(size(constraintMat, 2) ~= 2),
		error('CONSTRAINTMAT has the wrong format: second dimension is equal to %d instead of 2.',...
			size(constraintMat, 2));
	end;
	numConstraints = length(betaVector);
	violations = zeros(numConstraints, 1);
	for iterConstraint = 1:numConstraints,
		beta = betaVector(iterConstraint);
		coord1 = constraintMat(iterConstraint, 1);
		coord2 = constraintMat(iterConstraint, 2);
		dij = K(coord1, coord1) + K(coord2, coord2) - 2 * K(coord1, coord2);
		violations(iterConstraint) = 0.5 * (beta - dij) ^ 2;
	end;

	if (nargout >= 2),
		numViolations = sum(abs(violations) > 0);
	end;
else
	error('Unknown type of distance constraints.');
end;
