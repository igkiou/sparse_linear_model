% HUBER_OBJ_GRAD_MEX Returns value and gradient with respect to Phi of
% multitask huberized hinge loss objective function.
%
%	[l dl] = square_obj_grad_mex(Phi, X, Y, weights, bias) returns the
%	value of the objective function for the huberized hinge loss, and its
%	derivative with respect to Phi, at the point specified by the inputs.
%
%	Inputs:
%	Phi		- projection matrix, M x N matrix, where N is the original
%			  signal dimension and M is the number of projections (reduced
%			  dimension). 
%	X		- original data, N x P matrix, where P is the number of samples
%			  (each column is a signal).
%	Y		- labels for the classification task, P x T matrix, where T is
%			  the number of classification tasks. Labels must be -1 and +1
%			  for negative and positive, respectively (each column contains
%			  the P labels for the respective classification task). 
%	weights - weights used by the linear SVM, M x T matrix (each column
%			  contains the M weights for the linear SVM used for the
%			  respective classification task).
%	bias	- bias terms used by the linear SVM, 1 x T or T x 1 matrix
%			  (each element is the bias term for the linear SVM used for
%			  the respective classification task).
%
%	Outputs:
%	l		- value of objective function, scalar. The exact function
%			  calculated is 
%
%	1/P sum(i=1:T) sum(j=1:P) l(Y(j,i), X(:,j)'*Phi'*weights(:,i)+beta(i)
%
%			  where l is the huberized hinge loss.
%	dl		- value of derivative with respect to Phi at the given point of
%			  the above objective function. 
%
%	A sixth input argument may be provided, the wXtensor of size M*N x P,
%	but it is not used internally. It is only provided for back
%	compatibility.
