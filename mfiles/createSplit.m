function [trainIdx testIdx] = createSplit(numTrain, samplesPerClass)

if (any(samplesPerClass < numTrain)),
	error('Not all classes have enough samples to create a split with %d training samples per class.', numTrain);
end;

numClasses = length(samplesPerClass);
totalSamples = sum(samplesPerClass);
trainSamples = numTrain * numClasses;
testSamples = totalSamples - trainSamples;

trainIdx = zeros(trainSamples, 1);
testIdx = zeros(testSamples, 1);
samplesPrevClassAccum = [0; vec(cumsum(samplesPerClass))];
for iterClass = 1:numClasses,
	inds = randperm(samplesPerClass(iterClass));
	trainIdx((iterClass - 1) * numTrain + (1:numTrain), 1) = samplesPrevClassAccum(iterClass) + inds(1:numTrain);
	testIdx(samplesPrevClassAccum(iterClass) - (iterClass - 1) * numTrain + (1:(samplesPerClass(iterClass) - numTrain)), 1) =...
					samplesPrevClassAccum(iterClass) + inds((numTrain + 1):samplesPerClass(iterClass));
end;
