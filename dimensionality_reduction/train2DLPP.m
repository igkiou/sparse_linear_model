function [U, V, eigvalue_U, eigvalue_V, posIdx] = train2DLPP(trainFeatures, trainLabels, ReducedDim, DistanceType, NeighborMode, W)
%% set options
if (nargin < 3),
	ReducedDim = 10;
else
	if (~isscalar(ReducedDim)),
		error('ReducedDim is not scalar. May have passed semiSplit instead of ReducedDim.');
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

%% use LPP
options = [];
options.ReducedDim = ReducedDim;				% number of dimensions
options.nRepeat = 10;
trainFeatures = reshape(trainFeatures',32,32,size(trainFeatures,1));
[U, V, eigvalue_U, eigvalue_V, posIdx] = TensorLPP(trainFeatures, W, options);
