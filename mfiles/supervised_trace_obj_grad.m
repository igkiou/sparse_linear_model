function [obj deriv] = supervised_trace_obj_grad(M, y, b, X, mu, tau)

numSamples = size(X, 2);
signalDim = size(X, 1);
M = reshape(M, [signalDim, signalDim]);
distances = diag(X' * M * X);
hingeTerm = y .* (distances - b);
violations = find(hingeTerm > 0);
obj = mu * sum(hingeTerm(violations)) + tau * trace(M);

if (nargout > 2),
% 	deriv = zeros(size(M));
% 	numViolations = length(violations);
% 	for iter = 1:numViolations,
% 		deriv = deriv + mu * X(:, violations(iter)) * X(:, violations(iter))';
% 	end;
	deriv = mu * X(:, violations) * diag(y(violations)) * X(:, violations)' + tau * eye(size(M));
	deriv = deriv(:);
end;
