function [trainFeatures trainLabels testFeatures testLabels] = loadTextDataSet(params, numTrainPerClass, splitSelection)

if (nargin < 1),
	params = setParameters;
end;

if (nargin < 2),
	numTrainPerClass = 400;
end;

if (nargin < 3),
	splitSelection = 1;
end;

load(sprintf('~/MATLAB/sparse_linear_model/text_experiments/%dTrain/%d', numTrainPerClass, splitSelection));
[tngData reuData vocabData tngLabels] = getTextData(params);
trainFeatures = [tngData(:, trainIdx) reuData];
trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
testFeatures = tngData(:, testIdx);
testLabels = tngLabels(testIdx);

params.textDataSet = 'tng';
params.textVocabulary = 'tng';
params.textSelectionMethod = 'mutinfo';
params.textNumWords = 100;
