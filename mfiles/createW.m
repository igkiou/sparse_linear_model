function [W options] = createW(trainFeatures, trainLabels, DistanceType, NeighborMode, WeightMode)
%% set options

if (nargin < 3),
	DistanceType = 'Euclidean';
end;

if (nargin < 4),
	NeighborMode = 'KNN';
end;

if (nargin < 5),
	if (strcmp(DistanceType, 'Euclidean')),
		WeightMode = 'Binary';
	elseif (strcmp(DistanceType, 'Cosine')),
		WeightMode = 'Cosine';
	end;
end;

%% construct graph weight matrix
options = [];
options.Metric = DistanceType;			% type of distance (do not change)
options.NeighborMode = NeighborMode;	% supervised or not
options.bSelfConnected = 1;				% self-connected nodes
options.k = 10;							% number of neighbors for graph
options.WeightMode = WeightMode;		% type of weights
options.t = 10;							% parameter of heat kernel
options.gnd = trainLabels;				% provide labels
W = constructW(trainFeatures, options);
