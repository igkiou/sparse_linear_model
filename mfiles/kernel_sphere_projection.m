function [KNorm normVec] = kernel_sphere_projection(K, radius)

if (nargin < 2),
	radius = 1;
end;

normVec = sqrt(diag(K))';
normFactor = normVec / radius;
normFactor(normFactor == 0) = 1;
KNorm = bsxfun(@rdivide, bsxfun(@rdivide, K, normFactor'), normFactor);
