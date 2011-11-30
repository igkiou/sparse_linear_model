function [Vec W options] = trainKernelSRLPP(trainFeatures, trainLabels,...
	ReducedDim, DistanceType, NeighborMode, gramTrainTrain, W)
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

%% use Kernel SR-LPP
options = [];
options.W = W;
options.ReguAlpha = 0.01;
options.KernelType = 'Gaussian';
options.t = 5;
options.ReguType = 'Ridge';
options.ReducedDim = ReducedDim;
if (~exist('gramTrainTrain', 'var')),
	Vec = KSR_caller(options, trainFeatures);
else
	options.Kernel = 1;
	Vec = KSR_caller(options, gramTrainTrain);
end;
