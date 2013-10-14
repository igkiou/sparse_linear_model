% Clustering Toolbox
% Copyright (C) David Corney 2000			D.Corney@cs.ucl.ac.uk
%
%Clustering routines:
% 
% dckmeans	- k-means clustering
% dcEMGMM	- fits a Gaussian mixture model to data
% dcFuzzy	- fuzzy c-means clustering
% dcAgg		- agglomerative (hierarchical) clustering
% DoClustering		- call the above routines
% 
% 
%Utilities:
% ChooseInitialCentres - chooses k points from set 
% Contingency     - forms contingency matrix 
% LogFactorial    - calculates log of factorial
% MakeGaussData   - creates spherical Gaussian data sets
% MaxMax          - max value in n-d array
% MeanMean        - mean value in n-d array
% MinMin          - min value in n-d array
% PCAGraph        - plots graph using first 2/3 principal components
% PlotColour      - returns string for PLOT marker (colour,shape)
% RandIndex       - calculates adjusted Rand index for cluster solutions
% SumSum          - sum value in n-d array
%
%Plus:
% DcDemo		- simple script demonstrating some of these functions
%
