function [DNorm normVec] = l2_ball_projection(D, radius)

if (nargin < 2),
	radius = 1;
end;

normVec = sqrt(sum(D .^ 2, 1));
normFactor = normVec / radius;
normFactor(normFactor < 1) = 1;
DNorm = bsxfun(@rdivide, D, normFactor);

