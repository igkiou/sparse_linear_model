% LI2NSVM_GRAD_HUBER_MEX Returns value and gradient with respect to the SVM
% parameters of the SVM objective function based on the huberized hinge
% loss. 
%
%	[l dl] = li2nsvm_grad_huber_mex(wb, X, Y, lambda, biasflag) returns the
%	value of SVM objective function based on the huberized hinge loss and
%	its derivative with respect to the SVM parameters [weights; bias], at
%	the point specified by the inputs. 
%
%	Inputs:
%	wb		 - SVM parameters, (M+1) x 1 or 1 x (M+1) matrix, where M is the
%			   number of features in the data. The first M elements
%			   correspond to the weights of the linear SVM and the M+1
%			   element is the bias term.
%	X		 - data, M x P matrix, where P is the number of samples (each
%			   column is a sample). 
%	Y		 - labels for the classification task, P x 1 or 1 x P matrix.
%			   Labels must be -1 and +1 for negative and positive,
%			   respectively.
%	lambda	 - regularization parameter, scalar.
%	biasflag - flag, scalar, determining whether a bias term is used, if
%			   set to 1, or whether calculations involving the bias are
%			   skipped, if set to 0 (optional, default 1).
%
%	Outputs:
%	l		 - value of objective function, scalar. The exact function
%			   calculated is 
%
%				sum(j=1:P) l(Y(j), X(:,j)'*weights+beta)+lambda*||w||_2^2
%
%			   where l is the huberized hinge loss.
%	dl		 - value of derivative at the given point of the above
%			   objective function, with respect to the vector
%			   [weights;bias], (M+1) x 1 matrix, where the first M elements 
%			   are the derivatives with respect to the weights and the M+1
%			   element is the derivative with respect to the bias term. If
%			   biasflag is set to 0, then dl[M+1] = 0.
