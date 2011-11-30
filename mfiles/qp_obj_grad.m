% function [obj grad] = tempobjgrad(Z, Y)

% obj = sum((Z(:) - Y(:)).^2) / 2;
% grad = Z - Y;

% function [obj grad] = tempobjgrad(W, X, Y)
% 
% grad = [];
% obj = 0;
% for i = 1:length(X)
%     grad = [grad, X{i}'*(X{i}*W(:,i)-Y{i})];
% 	obj = obj + 1/2 * sum(vec(X{i}*W(:,i)-Y{i}) .^ 2);
% end

function [obj deriv] = qp_obj_grad(A, X, D)

obj = norm(X - D * A, 'fro') ^ 2 / 2;
if (nargout >= 2),
	deriv = - D' * (X - D * A);
end;
