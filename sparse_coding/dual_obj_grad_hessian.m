function [f, g, H] = dual_obj_grad_hessian(dual_lambda, SSt, XSt, trXXt, c)
% Compute the objective function value at x

[L M] = size(XSt);

SStLambda = SSt + diag(dual_lambda);
SSt_inv = eye(size(SStLambda)) / SStLambda;
% cholSStLambda = chol(SStLambda);
temp = [];

if L > M,
	% (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
	f = - trace(SSt_inv * (XSt' * XSt)) + trXXt - c * sum(dual_lambda);
% 	f = - trace(SStLambda \ (XSt' * XSt)) + trXXt - c * sum(dual_lambda);
% 	f = - trace((cholSStLambda \ (cholSStLambda' \ (XSt' * XSt)))) + trXXt - c * sum(dual_lambda);
else
	% (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
	temp = XSt * SSt_inv; 
	f = - trace(temp * XSt') + trXXt - c * sum(dual_lambda);
% 	f = - trace(XSt * (SStLambda \ XSt')) + trXXt - c * sum(dual_lambda);
% 	f = - trace(XSt * (cholSStLambda \ (cholSStLambda' \ XSt'))) + trXXt - c * sum(dual_lambda);
end;

f = - f;

if nargout > 1,
	g = zeros(M, 1);
	if (isempty(temp)),
		temp = XSt * SSt_inv;
	end;
% 	temp = XSt / SStLambda;
% 	temp = (XSt / cholSStLambda) / cholSStLambda';
	g = sum(temp .^ 2) - c;
	g = - g;
end;    
    
if nargout > 2
	H = - 2 .* ((temp' * temp) .* SSt_inv);
	H = - H;
end
