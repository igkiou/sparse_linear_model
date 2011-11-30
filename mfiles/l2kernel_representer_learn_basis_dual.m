function D = l2kernel_representer_learn_basis_dual(KXX, S, l2norm, Dinit)
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
%
% Written by Honglak Lee <hllee@cs.stanford.edu>
% Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

M = size(S, 1);

SSt = S * S';
SKXXSt = S * KXX * S';

if exist('Dinit', 'var')
    dual_lambda = diag(Dinit \ S' - SSt);
else
    dual_lambda = 10 * abs(rand(M, 1)); % any arbitrary initialization should be ok.
end

c = l2norm^2;
trKXX = trace(KXX);

lb = zeros(size(dual_lambda));
options = optimset('GradObj','on', 'Hessian','on');
% %  options = optimset('GradObj','on', 'Hessian','on', 'TolFun', 1e-7);
% dual_representer_obj_grad_hessian(dual_lambda, SSt, SKXXSt, trKXX, c)
[x, fval, exitflag, output] = fmincon(@(x) dual_representer_obj_grad_hessian(x, SSt, SKXXSt, trKXX, c), dual_lambda, ...
									[], [], [], [], lb, [], [], options);
% [x, fval, exitflag, output] = fmincon(@(x) dual_obj_grad(x, SSt, XSt, c, trXXt), dual_lambda, ...
% 									[], [], [], [], lb, [], [], options);
% [x, fval, exitflag, output] = fmincon(@(x) dual_obj_grad_mex(x, SSt, XSt', XSt'*XSt, trXXt, c), dual_lambda, ...
% 									[], [], [], [], lb, [], [], options);

% x = minimize_dual(dual_lambda, 100000, SSt, XSt', XSt'*XSt, trXXt, c);

% output.iterations
% fval_opt = -0.5*N*fval;
dual_lambda = x;

Dt = (SSt + diag(dual_lambda)) \ S;
D = Dt';
% fobjective_dual = fval_opt;
