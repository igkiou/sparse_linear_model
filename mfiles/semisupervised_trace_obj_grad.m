function [obj deriv] = semisupervised_trace_obj_grad(M, y, b, X, mu, tau, lambda, DDt, DDt2, VL, L)

numSamples = size(X, 2);
signalDim = size(X, 1);
M = reshape(M, [signalDim, signalDim]);
M = (M + M') / 2;
distances = diag(X' * M * X)';
hingeTerm = y .* (distances - b) / numSamples;
violations = find(hingeTerm > 0);
obj = mu * sum(hingeTerm(violations)) + tau * trace(M);

if (nargout < 2),
	objsemi = mahalanobis_obj_grad(M, DDt, DDt2, VL, L);
	obj = lambda / signalDim / signalDim * objsemi + obj;
	
elseif (nargout >= 2),
	[objsemi derivsemi] = mahalanobis_obj_grad(M, DDt, DDt2, VL, L);
	obj = lambda / signalDim / signalDim * objsemi + obj;
% 	deriv = zeros(size(M));
% 	numViolations = length(violations);
% 	for iter = 1:numViolations,
% 		deriv = deriv + mu * X(:, violations(iter)) * X(:, violations(iter))';
% 	end;
	deriv = mu * X(:, violations) * diag(y(violations)) * X(:, violations)' / numSamples + tau * eye(size(M));
	deriv = (deriv + deriv') / 2;
	deriv = deriv(:) + lambda  / signalDim / signalDim * derivsemi;
end;
