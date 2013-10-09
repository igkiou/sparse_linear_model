function [trainDataReduced, mapping] = kernel_pca_custom(trainData, numDimensions, gramMatrix, varargin)
%KERNEL_PCA Perform the kernel PCA algorithm
%
%   [trainDataReduced, mapping] = kernel_pca(trainData, numDimensions)
%   [trainDataReduced, mapping] = kernel_pca(trainData, numDimensions, kernel)
%   [trainDataReduced, mapping] = kernel_pca(trainData, numDimensions, kernel, param1)
%   [trainDataReduced, mapping] = kernel_pca(trainData, numDimensions, kernel, param1, param2)
%
% The function runs kernel PCA on a set of datapoints trainData. The variable
% numDimensions sets the number of dimensions of the feature points in the 
% embedded feature space (numDimensions >= 1, default = 2). 
% For numDimensions, you can also specify a number between 0 and 1, determining 
% the amount of variance you want to retain in the PCA step.
% The value of kernel determines the used kernel. Possible values are 'linear',
% 'gauss', 'poly', 'subsets', or 'princ_angles' (default = 'gauss'). For
% more info on setting the parameters of the kernel function, type HELP
% GRAM.
% The function returns the locations of the embedded trainingdata in 
% trainDataReduced. Furthermore, it returns information on the mapping in mapping.
%
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction v0.7.2b.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.

if (nargin < 3),
	gramMatrix = [];
end;

kernelType = 'g';
kernelParam1 = 1;
kernelParam2 = 1;
if (nargin > 3),
	kernelType = varargin{1};
	if ((length(varargin) > 1) && strcmp(class(varargin{2}), 'double')), 
		kernelParam1 = varargin{2}; 
	end;
	if ((length(varargin) > 2) && strcmp(class(varargin{3}), 'double')), 
		kernelParam2 = varargin{3}; 
	end;
end;

if (isempty(gramMatrix)),
	gramMatrix = kernel_gram_mex(trainData, [], kernelType, kernelParam1, kernelParam2);
end;

% Store the number of training and test points
numSamples = size(gramMatrix, 2);

mapping.column_sums = sum(gramMatrix) / numSamples;                       % column sums
mapping.total_sum   = sum(mapping.column_sums) / numSamples;     % total sum
gramMatrix = bsxfun(@minus, gramMatrix, mapping.column_sums);
gramMatrix = bsxfun(@minus, gramMatrix, mapping.column_sums');
gramMatrix = gramMatrix + mapping.total_sum;

% Compute first numDimensions eigenvectors and store these in V, store corresponding eigenvalues in L
gramMatrix(isnan(gramMatrix)) = 0;
gramMatrix(isinf(gramMatrix)) = 0;
gramMatrix = (gramMatrix + gramMatrix') / 2;
options = struct('disp',0);
[V, L] = eigs(gramMatrix, numDimensions, 'LA', options);

% Sort eigenvalues and eigenvectors in descending order
[L, ind] = sort(diag(L), 'descend');
L = L(1:numDimensions);
V = V(:,ind(1:numDimensions));

% Compute inverse of eigenvalues matrix L
invL = diag(1 ./ L);

% Compute square root of eigenvalues matrix L
sqrtL = diag(sqrt(L));

% Compute inverse of square root of eigenvalues matrix L
invsqrtL = diag(1 ./ diag(sqrtL));

% Compute the new embedded points for both K and Ktest-data
trainDataReduced = sqrtL * V';                     % = invsqrtL * V'* K

% Store information for out-of-sample extension
mapping.trainData = trainData;
mapping.V = V;
mapping.invsqrtL = invsqrtL;
mapping.kernelType = kernelType;
mapping.kernelParam1 = kernelParam1;
mapping.kernelParam2 = kernelParam2;

