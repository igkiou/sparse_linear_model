function [Mproj numEigUpdated] = projectionPSD(M, N, numEig, increment)

if (nargin < 4)
	increment = 100;
end;

M = reshape(M, [N N]);
M = (M + M') / 2;

% currNumEig = numEig;
% OK = 0;
% while ~OK
% 	[V, D] = laneig_modified_nocomplex(M,currNumEig,'AL');
% % 	[V, D] = eigs(M, currNumEig, 'LA', opts);
% 	OK = (D(currNumEig, currNumEig) <= 0)...
% 		|| (abs(D(currNumEig, currNumEig)) <= tol)...
% 		|| (currNumEig == N);
% 	currNumEig = min(currNumEig + increment, N);
% end
[V D] = safeEig(M);
eigVec = diag(D); 
tol = N * eps(eigVec(1));
numEigUpdated = sum((eigVec > 0) & (abs(eigVec) > tol));
V = V(:, 1:numEigUpdated);
Mproj = V * diag(eigVec(1:numEigUpdated)) * V';
