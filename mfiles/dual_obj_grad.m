function [f, g] = dual_obj_grad(dual_lambda, SSt, XSt, trXXt, c)
% Compute the objective function value at x

[L M] = size(XSt);

SStLambda = SSt + diag(dual_lambda);
cholSStLambda = chol(SStLambda);

if L > M,
	f = - trace((cholSStLambda \ (cholSStLambda' \ (XSt' * XSt)))) + trXXt - c * sum(dual_lambda);
else
	f = - trace(XSt * (cholSStLambda \ (cholSStLambda' \ XSt'))) + trXXt - c * sum(dual_lambda);
end;

f = - f;

if nargout > 1,
	temp = (XSt / cholSStLambda) / cholSStLambda';
	g = sum(temp .^ 2) - c;
	g = - g';
end;    
