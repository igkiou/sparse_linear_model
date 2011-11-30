function [KNorm normVec] = kernel_ball_projection(K, radius)

if (nargin < 2),
	radius = 1;
end;

normVec = sqrt(diag(K))';
normFactor = normVec / radius;
normFactor(normFactor < 1) = 1;
KNorm = bsxfun(@rdivide, bsxfun(@rdivide, K, normFactor'), normFactor);

