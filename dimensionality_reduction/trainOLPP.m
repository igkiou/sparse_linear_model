function Vec = trainOLPP(trainFeatures, trainLabels, ReducedDim)
%% set options
if (nargin < 3),
	ReducedDim = 10;
else
	if (~isscalar(ReducedDim)),
		error('ReducedDim is not scalar. May have passed semiSplit instead of ReducedDim.');
	end;
end;

%% construct graph weight matrix
options = [];
options.Metric = 'Euclidean';			% type of distance (do not change)
options.NeighborMode = 'KNN';			% supervised or not
options.bSelfConnected = 1;				% self-connected nodes
options.k = 5;							% number of neighbors for graph
options.WeightMode = 'Binary';			% type of weights
options.t = 10;							% parameter of heat kernel
options.gnd = trainLabels;				% provide labels
W = constructW(trainFeatures, options);

%% use LPP
options = [];
options.PCARatio = 1;					% how much of PCA to use
options.ReducedDim = ReducedDim;				% number of dimensions
Vec = OLPP(W, options, trainFeatures);

%% use SR-LPP
% options = [];
% options.W = W;
% options.ReguAlpha = 0.01;
% options.ReguType = 'Ridge';
% options.ReducedDim = ReducedDim;
% Vec = SR_caller(options, trainFeatures);

%% SR-SparseLPP
% options = [];
% options.W = W;
% options.ReguAlpha = 0.001;
% options.ReguType = 'RidgeLasso';
% options.LassoCardi = [10:5:60];
% options.ReducedDim = ReducedDim;
% [eigvector] = SR_caller(options, trainFeatures);
% [nSmp,nFea] = size(trainFeatures);
% for i = 1:length(options.LassoCardi)
%   eigvector = eigvectorAll{i};  %projective functions with cardinality options.LassoCardi(i)
% 
%   if size(eigvector,1) == nFea + 1
% 	  Y = [trainFeatures ones(nSmp, 1)] * eigvector;
%   else
% 	  Y = trainFeatures * eigvector;
%   end
% end
