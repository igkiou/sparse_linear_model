function [trainFeatures trainLabels testFeatures testLabels] = loadDataSet(dataSet, numTrain, useNorm, splitSelection, fea, gnd)

if (nargin < 3),
	useNorm = 0;
end;

if (nargin < 4),
	splitSelection = randint(1,1,50);
end;

if (nargin < 5),
	load(strcat('~/MATLAB/datasets_all/', dataSet, '/', dataSet, '_32x32.mat'));
end;

load(strcat('~/MATLAB/datasets_all/', dataSet, '/', num2str(numTrain), 'Train/', num2str(splitSelection)));

trainFeatures = fea(trainIdx,:);
trainLabels = gnd(trainIdx);
testFeatures = fea(testIdx,:);
testLabels = gnd(testIdx);

if (useNorm == 1)
	trainFeatures = normcols(trainFeatures')';
	testFeatures = normcols(testFeatures')';
elseif (useNorm == 2)
	trainFeatures = trainFeatures / 255;
	testFeatures = testFeatures / 255;
end;
