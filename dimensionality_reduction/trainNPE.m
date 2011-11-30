function Vec = trainNPE(trainFeatures, trainLabels, ReducedDim, NeighborMode)
%% set options
if (nargin < 3),
	ReducedDim = 10;
else
	if (~isscalar(ReducedDim)),
		error('ReducedDim is not scalar. May have passed semiSplit instead of ReducedDim.');
	end;
end;

if (nargin < 5),
	NeighborMode = 'KNN';
end;

%% use NPE;
options = [];
options.Regu = 0;
options.gnd = trainLabels;	
options.k = 5;
options.NeighborMode = NeighborMode;
options.PCARatio = 1;					% how much of PCA to use
options.ReducedDim = ReducedDim;				% number of dimensions
Vec = NPE(options, trainFeatures);
