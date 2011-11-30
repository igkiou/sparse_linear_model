function [normViolations numViolations violations distances] =...
						supervised_obj_grad(X, inds1, inds2, y, b)

deltaX = X(:, inds1) - X(:, inds2);
distances = diag(deltaX' * M * deltaX)';
violations = max((distances - b) .* y, 0);
normViolations = norm(violations);
numViolations = sum(violations > 0);
