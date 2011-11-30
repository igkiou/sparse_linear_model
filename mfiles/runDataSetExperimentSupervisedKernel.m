function [accuracies, accuracies_1NN, accuracyKernelSRLDA, accuracy2DLPPEuUn, accuracy2DLPPCosUn, accuracy2DLPPEuSup, accuracy2DLPPCosSup] = ...
	runDataSetExperimentSupervisedKernel(dataSet, trainingSize, dimensionRange, splits, useNorm, nNN, accuracies)

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
	accuracies = zeros(4, length(dimensionRange));
	accuracies_1NN = zeros(4, length(dimensionRange));
else
	accuraciesTemp = zeros(4, length(dimensionRange));
	accuraciesTemp(1:size(accuracies, 1), 1:size(accuracies, 2)) = accuracies;
	accuracies = accuraciesTemp;
end;

load(strcat('~/MATLAB/datasets_all/', dataSet, '/', dataSet, '_32x32.mat'));
numDims = length(dimensionRange);
numSplits = length(splits);
accuracies = accuracies * numSplits;

accuracyKernelSRLDA.accuracy = 0;
accuracyKernelSRLDA.accuracy_1NN = 0;

accuracy2DLPPEuUn.accuracy = 0;
accuracy2DLPPEuUn.accuracy_1NN = 0;

accuracy2DLPPCosUn.accuracy = 0;
accuracy2DLPPCosUn.accuracy_1NN = 0;

accuracy2DLPPEuSup.accuracy = 0;
accuracy2DLPPEuSup.accuracy_1NN = 0;

accuracy2DLPPCosSup.accuracy = 0;
accuracy2DLPPCosSup.accuracy_1NN = 0;

for iterSplit = 1:numSplits,
	disp(sprintf('Now running split %d out of %d.', iterSplit, numSplits));
	[fea_Train gnd_Train fea_Test gnd_Test] = loadDataSet(dataSet, trainingSize, useNorm, splits(iterSplit), fea, gnd);
	for iterDim = 1:numDims,
		disp(sprintf('Now running split %d out of %d, dimension %d out of %d.', iterSplit, numSplits, iterDim, numDims));
		
		[VecKernelSRLPP foo MRoptions] = trainKernelSRLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'KNN');
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionKernel(VecKernelSRLPP , fea_Train, fea_Test, MRoptions);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(1, iterDim) = accuracies(1, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(1, iterDim) = accuracies_1NN(1, iterDim) + accuracy;
		
		[VecKernelSRLPP foo MRoptions] = trainKernelSRLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'KNN');
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionKernel(VecKernelSRLPP , fea_Train, fea_Test, MRoptions);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(2, iterDim) = accuracies(2, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(2, iterDim) = accuracies_1NN(2, iterDim) + accuracy;
		
		[VecKernelSRLPP foo MRoptions] = trainKernelSRLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'Supervised');
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionKernel(VecKernelSRLPP , fea_Train, fea_Test, MRoptions);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(3, iterDim) = accuracies(3, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(3, iterDim) = accuracies_1NN(3, iterDim) + accuracy;
		
		[VecKernelSRLPP foo MRoptions] = trainKernelSRLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'Supervised');
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionKernel(VecKernelSRLPP , fea_Train, fea_Test, MRoptions);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(4, iterDim) = accuracies(4, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(4, iterDim) = accuracies_1NN(4, iterDim) + accuracy;
	end;
	[VecKernelSRLDA MRoptions] = trainKernelSRLDA(fea_Train, gnd_Train);
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionKernel(VecKernelSRLDA , fea_Train, fea_Test, MRoptions);
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
	accuracyKernelSRLDA.accuracy = accuracyKernelSRLDA.accuracy + accuracy;
	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
	accuracyKernelSRLDA.accuracy_1NN = accuracyKernelSRLDA.accuracy_1NN + accuracy;
	accuracyKernelSRLDA.dimension = size(VecKernelSRLDA, 2);
	
% 	[U, V, eigvalue_U, eigvalue_V, posIdx] = train2DLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'KNN');
% 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionTensor(fea_Train, fea_Test, U, V, posIdx);
% 	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 	accuracy2DLPPEuUn.accuracy = accuracy2DLPPEuUn.accuracy + accuracy;
% 	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 	accuracy2DLPPEuUn.accuracy_1NN = accuracy2DLPPEuUn.accuracy_1NN + accuracy;
% 	accuracy2DLPPEuUn.dimension = size(fea_Test_Reduced, 2);
% 	
% 	[U, V, eigvalue_U, eigvalue_V, posIdx] = train2DLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'KNN');
% 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionTensor(fea_Train, fea_Test, U, V, posIdx);
% 	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 	accuracy2DLPPCosUn.accuracy = accuracy2DLPPCosUn.accuracy + accuracy;
% 	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 	accuracy2DLPPCosUn.accuracy_1NN = accuracy2DLPPCosUn.accuracy_1NN + accuracy;
% 	accuracy2DLPPCosUn.dimension = size(fea_Test_Reduced, 2);
% 	
% 	[U, V, eigvalue_U, eigvalue_V, posIdx] = train2DLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'Supervised');
% 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionTensor(fea_Train, fea_Test, U, V, posIdx);
% 	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 	accuracy2DLPPEuSup.accuracy = accuracy2DLPPEuSup.accuracy + accuracy;
% 	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 	accuracy2DLPPEuSup.accuracy_1NN = accuracy2DLPPEuSup.accuracy_1NN + accuracy;
% 	accuracy2DLPPEuSup.dimension = size(fea_Test_Reduced, 2);
% 	
% 	[U, V, eigvalue_U, eigvalue_V, posIdx] = train2DLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'Supervised');
% 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionTensor(fea_Train, fea_Test, U, V, posIdx);
% 	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 	accuracy2DLPPCosSup.accuracy = accuracy2DLPPCosSup.accuracy + accuracy;
% 	[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 	accuracy2DLPPCosSup.accuracy_1NN = accuracy2DLPPCosSup.accuracy_1NN + accuracy;
% 	accuracy2DLPPCosSup.dimension = size(fea_Test_Reduced, 2);
end;

accuracies = accuracies / numSplits;
accuracies_1NN = accuracies_1NN / numSplits;

accuracyKernelSRLDA.accuracy = accuracyKernelSRLDA.accuracy / numSplits;
accuracyKernelSRLDA.accuracy_1NN = accuracyKernelSRLDA.accuracy_1NN / numSplits;

accuracy2DLPPEuUn.accuracy = accuracy2DLPPEuUn.accuracy / numSplits;
accuracy2DLPPEuUn.accuracy_1NN = accuracy2DLPPEuUn.accuracy_1NN / numSplits;

accuracy2DLPPCosUn.accuracy = accuracy2DLPPCosUn.accuracy / numSplits;
accuracy2DLPPCosUn.accuracy_1NN = accuracy2DLPPCosUn.accuracy_1NN / numSplits;

accuracy2DLPPEuSup.accuracy = accuracy2DLPPEuSup.accuracy / numSplits;
accuracy2DLPPEuSup.accuracy_1NN = accuracy2DLPPEuSup.accuracy_1NN / numSplits;

accuracy2DLPPCosSup.accuracy = accuracy2DLPPCosSup.accuracy / numSplits;
accuracy2DLPPCosSup.accuracy_1NN = accuracy2DLPPCosSup.accuracy_1NN / numSplits;
