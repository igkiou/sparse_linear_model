function [trainFeatReduced testFeatReduced] = reduceDimensionKernel(Vec, trainFeat, testFeat,...
					representersFeat, KTrain, KTest, kernelType, varargin)

if (nargin < 4),
	kernelType = 'g';
end

if (isempty(KTrain)),
	KTrain = kernel_gram_mex(trainFeat', representersFeat', kernelType, varargin{:});
end;
	
if (isempty(KTest) && (~isempty(testFeat))),
	KTest = kernel_gram_mex(testFeat', representersFeat', kernelType, varargin{:});
end;

trainFeatReduced = KTrain * Vec;
if (~isempty(testFeat)),
	testFeatReduced = KTest * Vec;
else
	testFeatReduced = [];
end;
