function setDataSetExperiment(dataSet, numTrain)

vec = randperm(50);
eval(sprintf('splits%d = vec(1:5);', numTrain));
eval(sprintf('accuracies%d = runDataSetExperiment(''%s'', %d, 10:10:150, splits%d, 0, 4, 5);',...
				numTrain, dataSet, numTrain, numTrain));
eval(sprintf('save ''%s''_experiments/%dTrain.mat splits%d accuracies%d', dataSet, numTrain, numTrain, numTrain));
