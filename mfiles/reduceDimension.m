function [trainFeatReduced testFeatReduced] = reduceDimension(Vec, trainFeat, testFeat, meanVector)

if (nargin >= 4),
	trainFeat = trainFeat - repmat(meanVector, size(trainFeat, 1), 1);
	if (~isempty(testFeat)),
		testFeat = testFeat - repmat(meanVector, size(testFeat, 1), 1);
	end;
end;

trainFeatReduced = trainFeat * Vec;
if (~isempty(testFeat)),
	testFeatReduced = testFeat * Vec;
else
	testFeatReduced = [];
end;
