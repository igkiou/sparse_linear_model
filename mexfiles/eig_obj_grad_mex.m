% EIG_OBJ_GRAD_MEX Returns value and gradient with respect to Phi of
% eigenvalue form of Grammian-based objective function.
%
%	[l dl] = eig_obj_grad_mex(Phi, DDt2, DDt3, VL, L) returns the
%	value of the Grammian-based objective function, and its derivative with
%	respect to Phi, at the point specified by the inputs.
%
%	Inputs:
%	Phi		- projection matrix, M x N matrix, where N is the original
%			  signal dimension and M is the number of projections (reduced
%			  dimension). 
%	DDt2	- N x N matrix equal to (D*D')^2, where D is the dictionary
%			  used for the encoding. D is an N x K matrix, where K is the
%			  number of atoms in the dictionary (each column contains one
%			  N x 1 atom).
%	DDt3	- N x N matrix equal to (D*D')^3.
%	VL		- N x N matrix equal to V * L, where [V L] = eig(D*D').
%	L		- N x 1 or 1 x N matrix equal to diag(L).
%
%	Outputs:
%	l		- value of objective function, scalar. The exact function
%			  calculated is 
%
%				||L-L*Gamma'*Gamma*L||_fro^2
%
%			  where Gamma = Phi * V.
%	dl		- value of derivative with respect to Phi at the given point of
%			  the above objective function. 
