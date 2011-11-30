function [accuracies, accuracies_1NN, accuracyLDA, accuracySRLDA, accuracyRLDA, accuracyTensorLDA] = ...
	runDataSetExperimentSupervised(dataSet, trainingSize, dimensionRange, splits, useNorm, nNN, accuracies)

if (nargin < 2),
	trainingSize = 5;
end;

if (nargin < 3),
	dimensionRange = 10:10:(trainingSize * 38);
end;

if (nargin < 4),
	splits = 1:50;
end;

if (nargin < 5),
	useNorm = 1;
end;

if (nargin < 6),
	nNN = 4;
end;

if (nargin < 7),
	accuracies = zeros(6, length(dimensionRange));
	accuracies_1NN = zeros(6, length(dimensionRange));
else
	accuraciesTemp = zeros(6, length(dimensionRange));
	accuraciesTemp(1:size(accuracies, 1), 1:size(accuracies, 2)) = accuracies;
	accuracies = accuraciesTemp;
end;

load(strcat('~/MATLAB/datasets_all/', dataSet, '/', dataSet, '_32x32.mat'));
numDims = length(dimensionRange);
numSplits = length(splits);
accuracies = accuracies * numSplits;

accuracyLDA.accuracy = 0;
accuracyLDA.accuracy_1NN = 0;
accuracySRLDA.accuracy = 0;
accuracySRLDA.accuracy_1NN = 0;
accuracyRLDA.accuracy = 0;
accuracyRLDA.accuracy_1NN = 0;
accuracyTensorLDA.accuracy = 0;
accuracyTensorLDA.accuracy_1NN = 0;
for iterSplit = 1:numSplits,
	disp(sprintf('Now running split %d out of %d.', iterSplit, numSplits));
	[fea_Train gnd_Train fea_Test gnd_Test] = loadDataSet(dataSet, trainingSize, useNorm, splits(iterSplit), fea, gnd);
	for iterDim = 1:numDims,
		disp(sprintf('Now running split %d out of %d, dimension %d out of %d.', iterSplit, numSplits, iterDim, numDims));
		
		[VecLPP W] = trainLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'Supervised');
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLPP, fea_Train, fea_Test);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(1, iterDim) = accuracies(1, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(1, iterDim) = accuracies_1NN(1, iterDim) + accuracy;
		
		VecSRLPP = trainSRLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'Supervised', W);
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSRLPP, fea_Train, fea_Test);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(2, iterDim) = accuracies(2, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(2, iterDim) = accuracies_1NN(2, iterDim) + accuracy;
		
		[VecLPP W] = trainLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'Supervised');
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLPP, fea_Train, fea_Test);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(3, iterDim) = accuracies(3, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(3, iterDim) = accuracies_1NN(3, iterDim) + accuracy;
		
		VecSRLPP = trainSRLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'Supervised', W);
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSRLPP, fea_Train, fea_Test);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(4, iterDim) = accuracies(4, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(4, iterDim) = accuracies_1NN(4, iterDim) + accuracy;

		VecTensorLPP = trainTensorLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'Supervised');
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecTensorLPP, fea_Train, fea_Test);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(5, iterDim) = accuracies(5, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(5, iterDim) = accuracies_1NN(5, iterDim) + accuracy;
		
		VecTensorLPP = trainTensorLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'Supervised');
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecTensorLPP, fea_Train, fea_Test);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(6, iterDim) = accuracies(6, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(6, iterDim) = accuracies_1NN(6, iterDim) + accuracy;
	end;
	VecLDA = trainLDA(fea_Train, gnd_Train);
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLDA, fea_Train, fea_Test);
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
	accuracyLDA.accuracy = accuracyLDA.accuracy + accuracy;
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
	accuracyLDA.accuracy_1NN = accuracyLDA.accuracy_1NN + accuracy;
	accuracyLDA.dimension = size(VecLDA, 2);
	
	VecSRLDA = trainSRLDA(fea_Train, gnd_Train);
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSRLDA, fea_Train, fea_Test);
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
	accuracySRLDA.accuracy = accuracySRLDA.accuracy + accuracy;
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
	accuracySRLDA.accuracy_1NN = accuracySRLDA.accuracy_1NN + accuracy;
	accuracySRLDA.dimension = size(VecSRLDA, 2);
	
	VecRLDA = trainRLDA(fea_Train, gnd_Train);
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecRLDA, fea_Train, fea_Test);
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
	accuracyRLDA.accuracy = accuracyRLDA.accuracy + accuracy;
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
	accuracyRLDA.accuracy_1NN = accuracyRLDA.accuracy_1NN + accuracy;
	accuracyRLDA.dimension = size(VecRLDA, 2);
	
	VecTensorLDA = trainTensorLDA(fea_Train, gnd_Train);
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecTensorLDA, fea_Train, fea_Test);
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
	accuracyTensorLDA.accuracy = accuracyTensorLDA.accuracy + accuracy;
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
	accuracyTensorLDA.accuracy_1NN = accuracyTensorLDA.accuracy_1NN + accuracy;
	accuracyTensorLDA.dimension = size(VecTensorLDA, 2);
	
end;

accuracies = accuracies / numSplits;
accuracies_1NN = accuracies_1NN / numSplits;

accuracyLDA.accuracy = accuracyLDA.accuracy / numSplits;
accuracyLDA.accuracy_1NN = accuracyLDA.accuracy_1NN / numSplits;

accuracySRLDA.accuracy = accuracySRLDA.accuracy / numSplits;
accuracySRLDA.accuracy_1NN = accuracySRLDA.accuracy_1NN / numSplits;

accuracyRLDA.accuracy = accuracyRLDA.accuracy / numSplits;
accuracyRLDA.accuracy_1NN = accuracyRLDA.accuracy_1NN / numSplits;

accuracyTensorLDA.accuracy = accuracyTensorLDA.accuracy / numSplits;
accuracyTensorLDA.accuracy_1NN = accuracyTensorLDA.accuracy_1NN / numSplits;
