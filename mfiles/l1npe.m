function [mappedX, mapping] = l1npe(X, no_dims, W)
%NPE Perform the Neighborhood Preserving Embedding algorithm
%
%       [mappedX, mapping] = npe(X, no_dims, k)
%       [mappedX, mapping] = npe(X, no_dims, k, eig_impl)
%		X has rows as elements.
%		W = At';
% 
% Runs the Neighborhood Preserving Embedding algorithm on dataset X to 
% reduce it to dimensionality no_dims. The number of neighbors that is used
% by LPP is specified by k (default = 12).
%
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction v0.7.2b.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, 2010
% University California, San Diego / Delft University of Technology


if size(X, 2) > size(X, 1)
	error('Number of samples should be higher than number of dimensions.');
end
if ~exist('no_dims', 'var')
	no_dims = 2; 
end

% Get dimensionality and number of dimensions
[n, d] = size(X);
mapping.mean = mean(X, 1);

ImW = eye(size(W)) - W;
M = ImW' * ImW;

% For sparse datasets, we might end up with NaNs or Infs in M. We just set them to zero for now...
M(isnan(M)) = 0;
M(isinf(M)) = 0;

% Compute XWX and XX and make sure these are symmetric
X = X';
WP = X' * M * X;
DP = X' * X;
DP = (DP + DP') / 2;
WP = (WP + WP') / 2;

% Solve generalized eigenproblem
if size(X, 1) > 1500 && no_dims < (size(X, 1) / 10)
	options.disp = 0;
	options.issym = 1;
	options.isreal = 0;
	[eigvector, eigvalue] = eigs(WP, DP, no_dims, 'SA', options);
else
	[eigvector, eigvalue] = eig(WP, DP);
end

% Sort eigenvalues in descending order and get smallest eigenvectors
[eigvalue, ind] = sort(diag(eigvalue), 'ascend');
eigvector = eigvector(:,ind(1:no_dims));

% Compute final linear basis and map data
mappedX = X * eigvector;
mapping.M = eigvector;
