function [obj deriv] = lsqr_obj_grad(Phi, X, Y, alphaReg)
% objective function value and gradient of sensing matrix learning routine,
% for objective function ||L - L * Gamma' * Gamma * L||_fro ^ 2.
%
%	Note: Inputs are chosen to optimize performance. Optional arguments
%	should be precalculated before the optimization algorithm begins
%	running and provided, to significantly reduce running speed.
%
%	Inputs
%	Phi: vectorized form of sensing matrix.
%	DDt: D * D', where D is the dictionary.
%	DDt2: DDt ^ 2 (optional).
%	VL: V * L, see below (optional).
%	L: matrix L, see below (optional). L should be stored as sparse for
%	optimal speed.
%	In the above, [V, L] are the outputs of eig(DDt). 
%
%	Outputs
%	obj: value of objective function.
%	deriv: vectorized form of gradient.
%

% original signal size
n = size(X, 1);
m = size(Y, 1);
Phi = reshape(Phi, [m n]);

Err = Phi * X - Y;
obj = norm(Err, 'fro') ^ 2;

if (alphaReg > 0),
	obj = obj + alphaReg * norm(Phi, 'fro') ^ 2;
end;

if (nargout >= 2),
	deriv = 2 * Err * X';
	
	if (alphaReg > 0),
		deriv = deriv + alphaReg * 2 * Phi;
	end;
	
	deriv = deriv(:);
end;
