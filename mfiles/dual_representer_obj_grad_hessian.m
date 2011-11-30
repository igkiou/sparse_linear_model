function [f, g, H] = dual_representer_obj_grad_hessian(dual_lambda, SSt, SKXXSt, trKXX, c)
% Compute the objective function value at x

M = size(SSt, 1); % L: signal dimension, M: number of dictionary elements

SStLambda = SSt + diag(dual_lambda);
SSt_inv = eye(size(SStLambda)) / SStLambda;
SKXXStSSt_inv = SKXXSt * SSt_inv;

f = - trace(SKXXStSSt_inv) + trKXX - c * sum(dual_lambda);
f = - f;

if nargout > 1,
	g = zeros(M, 1);
% 	temp = XSt * SSt_inv;
	SSt_invSKXXStSSt_inv = SSt_inv * SKXXStSSt_inv;
% 	temp = XSt / SStLambda;
% 	temp = (XSt / cholSStLambda) / cholSStLambda';
% 	g = sum(temp .^ 2) - c;
	g = diag(SSt_invSKXXStSSt_inv) - c;
	g = - g;
end;    
    
if nargout > 2
	H = - 2 .* (SSt_invSKXXStSSt_inv .* SSt_inv);
	H = - H;
end
