function kernelMatrix = kernel_gram(X1, X2, kernelType, param1, param2)
%KRAM Computes the Gram-matrix of data points X using a kernel function
%
%   K = kernel_gram(X1, X2, kernelType, param1, param2)
%
% Computes the Gram-matrix of data points X1 and X2 using the specified
% kernel function. If no kernel is specified, no kernel function is
% applied. The function KERNEL_GRAM is than equal to X1*X2'. The use of the
% function is different depending on the specified kernel function (because
% different kernel functions require different parameters. The
% possibilities are listed below.
% Linear kernel: K = kernel_gram(X1, X2, 'l')
%           which is parameterless
% Gaussian kernel: K = kernel_gram(X1, X2, 'g', s)
%           where s is the variance of the used Gaussian function (default = 1).
% Polynomial kernel: K = kernel_gram(X1, X2, 'p', R, d)
%           where R is the addition value and d the power number (default = 0 and 3)
% Sigmoid kernel: K = kernel_gram(X1, X2, 's', R, s)
%			where R is the addition value and s the scale factor (default = 0 and 1)
%			(note that in general the sigmoid is not positive-definite, and
%			therefore not a true kernel) 
% Sobolev kernel: K = kernel_gram(X1, X2, 'h', s)
%			where s is a parameter that controls smoothness (default = 1).
%
%	'l' or 'L': linear				k(a,b) = a'*b
%	'g' or 'G': Gaussian (default)	k(a,b) = exp(-0.5*||a-b||^2/s^2)
%	'p' or 'P': polynomial			k(a,b) = (a'*b+R)^d
%	'h' or 'H': Sobolev				k(a,b) = exp(-|a-b|/s)

if (nargin < 1),
	error('Not enough input arguments'); %#ok
end;

oneArgFlag = 0;
if ((nargin < 2) || isempty(X2)),
	oneArgFlag = 1;
	X2 = [];
elseif (size(X2, 1) ~= size(X1, 1)),
	error('The signal dimension (first dimension) of the second sample matrix does not match the signal dimension (first dimension) of the first sample matrix.'); %#ok
end;

if (nargin < 3),
	kernelType = 'g';
end


if ((kernelType == 'l') || (kernelType == 'L')),
	% Linear kernel
	if (oneArgFlag == 1),
		kernelMatrix = X1' * X1;
	else
		kernelMatrix = X1' * X2;
	end;

% TODO: Optimize to as much as possible avoid matrix replication due to
% calls by value, for all kernel types.	
elseif ((kernelType == 'g') || (kernelType == 'G')),
	% Gaussian kernel
	if (nargin < 4), 
		param1 = 1; 
	end;
	kernelMatrix = l2_distance(X1, X2, 0);
	if (oneArgFlag == 1),
		kernelMatrixSize1 = size(X1, 2);
		upperTriang = triu(ones(kernelMatrixSize1), 1);
		uKernelVec = kernelMatrix(upperTriang == 1);
		uKernelVec = exp(-(uKernelVec / (2 * param1 .^ 2)));
		kernelMatrix = 0.5 * eye(kernelMatrixSize1);
		kernelMatrix(upperTriang == 1) = uKernelVec;
		kernelMatrix = kernelMatrix + kernelMatrix';
% 		kernelMatrix(sub2ind([kernelMatrixSize1 kernelMatrixSize1], 1:kernelMatrixSize1, 1:kernelMatrixSize1)) = 0.5;
% 		kernelMatrix(upperTriang == 1) = uKernelVec;
	else
		kernelMatrix = exp(-(kernelMatrix / (2 * param1 .^ 2)));
	end;
	
elseif ((kernelType == 'p') || (kernelType == 'P')),
	% Polynomial kernel
	if (nargin < 4), 
		param1 = 0; 
	end;
	if (nargin < 5), 
		param2 = 3; 
	end;
	if (oneArgFlag == 1),
		kernelMatrix = X1' * X1;
		upperTriang = triu(ones(size(kernelMatrix, 1)), 1);
		uKernelVec = kernelMatrix(upperTriang == 1);
		uKernelVec = (uKernelVec + param1) .^ param2;
		kernelMatrix = 0.5 * diag((diag(kernelMatrix) + param1) .^ param2);
		kernelMatrix(upperTriang == 1) = uKernelVec;
		kernelMatrix = kernelMatrix + kernelMatrix';
	else
		kernelMatrix = X1' * X2;
		kernelMatrix = (kernelMatrix + param1) .^ param2;
	end;
	
elseif ((kernelType == 'h') || (kernelType == 'H')),
	% Sobolev kernel
	if (nargin < 4), 
		param1 = 1; 
	end;
	kernelMatrix = l1_distance(X1, X2);
	
	if (oneArgFlag == 1),
		kernelMatrixSize1 = size(X1, 2);
		upperTriang = triu(ones(kernelMatrixSize1), 1);
		uKernelVec = kernelMatrix(upperTriang == 1);
		uKernelVec = exp(- uKernelVec / param1);
		kernelMatrix = 0.5 * eye(kernelMatrixSize1);
		kernelMatrix(upperTriang == 1) = uKernelVec;
		kernelMatrix = kernelMatrix + kernelMatrix';
	else
		kernelMatrix = exp(- kernelMatrix / param1);
	end;		
else
	error('Unknown kernel function.'); %#ok
end;
