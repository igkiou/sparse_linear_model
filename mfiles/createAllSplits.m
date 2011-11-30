numTrain = [400]; % how many training samples per class do you want in the split
% samplesPerClass: how many training samples exist in every class, vector, must be provided 
numSplits = 50;
for iterTrain = 1:length(numTrain),
	mkdir(sprintf('%dTrain', numTrain(iterTrain)));
	for iterRandom = 1:numSplits,
		[trainIdx testIdx] = createSplit(numTrain(iterTrain), samplesPerClass);
		save(sprintf('%dTrain/%d.mat', numTrain(iterTrain), iterRandom), 'trainIdx', 'testIdx');
	end;
end;
