function [C Y] = li2nsvm_multiclass_fwd(X, w, b, class_name, kernelMatrix)

% function [C Y] = li2nsvm_multiclass_fwd(X, w, b, class_name):
% make multi-class prediction
% TODO: Make li2nsvm_fwd and li2nsvm_multiclass_fwd output formats
% compatible. Also, change in all their uses in svm-related functions.
if (nargin < 5),
	kernelMatrix = [];
end;

if (isempty(kernelMatrix)),
	Y = w' * X + repmat(b', [1 size(X, 2)]);
else
	Y = w' * kernelMatrix * X + repmat(b', [1 size(X, 2)]);
end;
C = oneofc_inv(Y, class_name);
% accuracy = sum(Yte==Cte)/size(Yte,1);
% fprintf('the accuracy is %f \n', accuracy);

