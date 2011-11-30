function D = l2exp_learn_basis_gradient_projection_backtracking(X, S, Dinit, family)
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

[N numSamples] = size(X);
K = size(S, 1);

if (nargin < 3),
    D = randn(N, K);
else
	D = Dinit;
end;

if (nargin < 4),
	family = 'P';
end;

BETA = 0.9;
ALPHA = 0.3;
EPS = 10 ^ - 12;
MAXITER = 200;
MAXITEROUT = 50;

fproj = @(D) vec(l2_ball_projection_mex(reshape(D, [N K])));
fgrad = @(D) basis_exp_obj_grad(D, X, S, family);

D = D(:);
iter = 0;
while (1),
	[preObjVal, stp] = fgrad(D);
	stp = - stp(:);
	t = 1;
	p = - stp' * stp;
	for iterBT = 1:MAXITER,
		Dnew = D + t * stp;
		postObjVal = fgrad(fproj(Dnew));
		if (postObjVal < preObjVal + ALPHA * t * p),
			break;
		else  
			t = BETA * t;
		end;
	end;
	D = fproj(Dnew);
	
	if (abs(postObjVal - preObjVal) < EPS),
		break;
	end;
	iter = iter + 1;
	if (iter > MAXITEROUT),
		break;
	end;
end;
D = reshape(D, [N K]);
