function M = semisupervised_trace_learning(X, y, b, Minit, mu, tau, lambda, DDt, DDt2, VL, L)
% Learning basis using Lagrange dual (with basis normalization)
%
% This code solves the following problem:
% 
%    minimize_D   0.5*||X - D*S||^2
%    subject to   ||D(:,j)||_2 <= l2norm, forall j=1...size(S,1)
% 
% The detail of the algorithm is described in the following paper:
% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
% Advances in Neural Information Processing Systems (NIPS) 19, 2007

% [N numSamples] = size(X);
% K = size(S, 1);
% 
% if (nargin < 3),
%     D = randn(N, K);
% else
% 	D = Dinit;
% end;
% 
% if (nargin < 4),
% 	family = 'P';
% end;



% BETA = 0.9;
BETA = 0.5;
ALPHA = 0.3;
EPS = 10 ^ - 12;
MAXITER = 200;
MAXITEROUT = 50;
increment = 100;

N = size(Minit, 1);
fproj = @(M) vec(projectionPSD(M, N, K));
fgrad = @(M) semisupervised_trace_obj_grad(M, y, b, X, mu, tau, lambda, DDt, DDt2, VL, L);

Minit = (Minit + Minit') / 2;
M = Minit(:);
iter = 0;
numEig = rank(Minit);
while (1),
	iter = iter + 1;
% 	[preObjVal, stp] = fgrad(M);
	[preObjVal, stp] = semisupervised_trace_obj_grad(M, y, b, X, mu, tau, lambda, DDt, DDt2, VL, L);
	disp(sprintf('iterOut: %d, objval', iter, preObjVal));
	stp = - stp(:);
	t = 1;
	p = - stp' * stp;
	for iterBT = 1:MAXITER,
		disp(sprintf('iterBT: %d', iterBT));
		Mnew = M + t * stp;
		[Mproj numEig] = projectionPSD(Mnew, N, numEig, increment);
% 		postObjVal = fgrad(Mproj(:));
		postObjVal = semisupervised_trace_obj_grad(Mproj(:), y, b, X, mu, tau, lambda, DDt, DDt2, VL, L);
		if (postObjVal < preObjVal + ALPHA * t * p),
			break;
		else  
			t = BETA * t;
		end;
	end;
	M = Mproj(:);
	
	if (abs(postObjVal - preObjVal) < EPS),
		break;
	end;
	if (iter > MAXITEROUT),
		break;
	end;
end;
M = reshape(M, [N N]);
