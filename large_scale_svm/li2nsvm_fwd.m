function [Ypred accuracy Yval] = li2nsvm_fwd(X, Y, w, b, kernelMatrix)

% function [Ypred accuracy] = li2nsvm_fwd(X, Y, w, b):
% make binary-class prediction
% TODO: Make li2nsvm_fwd and li2nsvm_multiclass_fwd output formats
% compatible. Also, change in all their uses in svm-related functions.
if (nargin < 5),
	kernelMatrix = [];
end;

if (isempty(kernelMatrix)),
	Yval = w' * X + b;
else
	Yval = w' * kernelMatrix * X + b;
end;
Ypred = sign(Yval);
accuracy = sum(Ypred == Y) / size(Y, 2) * 100;
% accuracy = sum(Yte==Cte)/size(Yte,1);
% fprintf('the accuracy is %f \n', accuracy);

