function [labeledIdx unsupervisedIdx semisplit] = createSemiSupervisedSplit(numLabeled, trainLabels)

numSamples = length(trainLabels);
labels = unique(trainLabels);
numLabels = length(labels);
labeledIdx = zeros(numLabels * numLabeled, 1);
unsupervisedIdx = zeros(numSamples - numLabels * numLabeled, 1);

currentUnsupervisedIdx = 0;
for iterLabel = 1:length(labels),
	labelSamples = find(trainLabels == labels(iterLabel));
	numLabelSamples = length(labelSamples);
	if (numLabelSamples < numLabeled),
		error('Not all classes have enough samples to create a split with %d labeled samples per class.', numLabeled);
	end;
	randIdx = randperm(numLabelSamples);
	labeledIdx((iterLabel - 1) * numLabeled + (1:numLabeled), 1) = labelSamples(randIdx(1:numLabeled));
	unsupervisedIdx(currentUnsupervisedIdx + (1:(numLabelSamples - numLabeled)), 1) =...
					labelSamples(randIdx((numLabeled + 1):numLabelSamples));
	currentUnsupervisedIdx = numLabelSamples - numLabeled;
end;
semisplit = false(size(trainLabels));
semisplit(labeledIdx) = true;
