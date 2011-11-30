function [Vec W options] = trainSDA_MR(trainFeatures, trainLabels, semiSplit, DistanceType, NeighborMode, W)
%% set options
if (nargin < 3),
	semiSplit = ones(size(trainLabels));
else
	if (any(size(semiSplit) ~= size(trainLabels))),
		error('Size of semiSplit does not agree with size of labels. May have passed ReducedDim instead of semiSplit.');
	end;
end;	

if (nargin < 4),
	DistanceType = 'Euclidean';
end;

if (nargin < 5),
	NeighborMode = 'KNN';
end;

if (strcmp(DistanceType, 'Euclidean')),
	WeightMode = 'Binary';
elseif (strcmp(DistanceType, 'Cosine')),
	WeightMode = 'Cosine';
end;

%% construct graph weight matrix
if (~exist('W', 'var')),
	options = [];
	options.Metric = DistanceType;			% type of distance (do not change)
	options.NeighborMode = NeighborMode;	% supervised or not
	options.bSelfConnected = 1;				% self-connected nodes
	options.k = 5;							% number of neighbors for graph
	options.WeightMode = WeightMode;		% type of weights
	options.t = 10;							% parameter of heat kernel
	options.gnd = trainLabels;				% provide labels
	W = constructW(trainFeatures, options);
end;

%% use SDA
options = [];
options.W = W;					% unsupervised graph
options.ReguBeta = 0.01;
options.ReguAlpha = 0.01;		% regularization
options.KernelType = 'Gaussian';
options.ReguType = 'Ridge';
RegularizerOptions.ReguType = 'Ridge';
% options.keepMean = 1;
Vec = SDA_MR(trainLabels, trainFeatures, semiSplit, options, RegularizerOptions);
