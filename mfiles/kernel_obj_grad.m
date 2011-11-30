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

function [obj deriv] = kernel_obj_grad(A, KXX, KDX, KDD)

KDDA = KDD * A;
obj = (KXX - 2 * A' * KDX + A' * KDDA) / 2;
if (nargout >= 2),
	deriv = - KDX + KDDA;
end;
