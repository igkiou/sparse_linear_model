function dataMatrixReduced = kernel_pca_oos(dataMatrix, gramMatrix, mapping)
%TRANSFORM_SAMPLE_EST Performs out-of-sample extension of new datapoints
%
%   t_points = out_of_sample(points, mapping)
%
% Performs out-of-sample extension of the new datapoints in points. The 
% information necessary for the out-of-sample extension is contained in 
% mapping (this struct can be obtained from COMPUTE_MAPPING).
% The function returns the coordinates of the transformed points in t_points.
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


% Compute and center kernel matrix
if (isempty(gramMatrix)),
	gramMatrix = kernel_gram_mex(mapping.trainData, dataMatrix, mapping.kernelType, mapping.kernelParam1, mapping.kernelParam2);
end;
	
J = repmat(mapping.column_sums', [1 size(gramMatrix, 2)]);
gramMatrix = gramMatrix - repmat(sum(gramMatrix, 1), [size(gramMatrix, 1) 1]) - J + repmat(mapping.total_sum, [size(gramMatrix, 1) size(gramMatrix, 2)]);

% Compute transformed points
dataMatrixReduced = mapping.invsqrtL * mapping.V' * gramMatrix;
