function [accuracies accuracies_1NN] = runDataSetExperiment(dataSet, trainingSize, dimensionRange, splits, useNorm, nNN, numSCTries, accuracies, accuracies_1NN)

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
	numSCTries = 5;
end;

if (nargin < 8),
	accuracies = zeros(11, length(dimensionRange));
	accuracies_1NN = zeros(11, length(dimensionRange));
else
	accuraciesTemp = zeros(11, length(dimensionRange));
	accuraciesTemp(1:size(accuracies, 1), 1:size(accuracies, 2)) = accuracies;
	accuracies = accuraciesTemp;
	accuraciesTemp = zeros(11, length(dimensionRange));
	accuraciesTemp(1:size(accuracies_1NN, 1), 1:size(accuracies_1NN, 2)) = accuracies_1NN;
	accuracies_1NN = accuraciesTemp;
end;

load(strcat('~/MATLAB/datasets_all/', dataSet, '/', dataSet, '_32x32.mat'));
numDims = length(dimensionRange);
numSplits = length(splits);
accuracies = numSplits * accuracies;
accuracies_1NN = numSplits * accuracies_1NN;

for iterSplit = 1:numSplits,
	disp(sprintf('Now running split %d out of %d.', iterSplit, numSplits));
	for iterDim = 1:numDims,
		fprintf('Now running split %d out of %d, dimension %d out of %d.', iterSplit, numSplits, iterDim, numDims);
		[fea_Train gnd_Train fea_Test gnd_Test] = loadDataSet(dataSet, trainingSize, useNorm, splits(iterSplit), fea, gnd);
% 		if (useNorm == 1),
% 			fea_Train = normcols(fea_Train')';
% 			fea_Test = normcols(fea_Test')';
% 		end;
% 		fprintf(' LPPEu,');
% 		[VecLPP W] = trainLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'KNN');
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLPP, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(1, iterDim) = accuracies(1, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(1, iterDim) = accuracies_1NN(1, iterDim) + accuracy;
% 		
% 		fprintf(' OLPP,');
% 		VecOLPP = trainOLPP(fea_Train, gnd_Train, dimensionRange(iterDim));
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecOLPP, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(2, iterDim) = accuracies(2, iterDim) + accuracy;
% 		
% 		fprintf(' SRLPPEu,');
% 		VecSRLPP = trainSRLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'KNN', W);
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSRLPP, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(3, iterDim) = accuracies(3, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(3, iterDim) = accuracies_1NN(3, iterDim) + accuracy;
		
% 		fprintf(' learn,');
% % 		accuracyAccum = 0;
% % 		accuracyAccum_1NN = 0;
% % 		for iterSC = 1:numSCTries,
% 		VecSP = learn_sensing_exact(fea_Train', dimensionRange(iterDim))';
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSP, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(4, iterDim) = accuracies(4, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(4, iterDim) = accuracies_1NN(4, iterDim) + accuracy;
% % 		accuracyAccum_1NN = accuracyAccum_1NN + accuracy;
% % 		end;
% % 		accuracies(4, iterDim) = accuracies(4, iterDim) + accuracyAccum / numSCTries;
% % 		accuracies_1NN(4, iterDim) = accuracies_1NN(4, iterDim) + accuracyAccum_1NN / numSCTries;

% 		fprintf(' random,');
% 		accuracyAccum = 0;
% 		accuracyAccum_1NN = 0;
% 		for iterSC = 1:numSCTries,
% 			VecRand = random_sensing(fea_Train', dimensionRange(iterDim))';
% 			[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecRand, fea_Train, fea_Test);
% 			[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 			accuracyAccum = accuracyAccum + accuracy;
% 			[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 			accuracyAccum_1NN = accuracyAccum_1NN + accuracy;
% 		end;
% 		accuracies(5, iterDim) = accuracies(5, iterDim) + accuracyAccum / numSCTries;
% 		accuracies_1NN(5, iterDim) = accuracies_1NN(5, iterDim) + accuracyAccum_1NN / numSCTries;
% 
% 		fprintf(' PCA,');
% 		VecPCA = trainPCA(fea_Train, gnd_Train, dimensionRange(iterDim));
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecPCA, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(6, iterDim) = accuracies(6, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(6, iterDim) = accuracies_1NN(6, iterDim) + accuracy;
% 		
% 		fprintf(' LPPCos,');
% 		[VecLPP W] = trainLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'KNN');
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLPP, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(7, iterDim) = accuracies(7, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(7, iterDim) = accuracies_1NN(7, iterDim) + accuracy;
% 		
% 		fprintf(' SRLPPCos,');
% 		VecSRLPP = trainSRLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'KNN', W);
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSRLPP, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(8, iterDim) = accuracies(8, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(8, iterDim) = accuracies_1NN(8, iterDim) + accuracy;
% 
% 		fprintf(' TensorLPPEu,');
% 		VecTensorLPP = trainTensorLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Euclidean', 'KNN');
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecTensorLPP, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(9, iterDim) = accuracies(9, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(9, iterDim) = accuracies_1NN(9, iterDim) + accuracy;
% 		
% 		fprintf(' TensorLPPCos,');
% 		VecTensorLPP = trainTensorLPP(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'KNN');
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecTensorLPP, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(10, iterDim) = accuracies(10, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(10, iterDim) = accuracies_1NN(10, iterDim) + accuracy;
%
% 		fprintf(' EigenFaces,');
% 		VecEigen = trainEigenFaces(fea_Train, gnd_Train, dimensionRange(iterDim));
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecEigen, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(11, iterDim) = accuracies(11, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(11, iterDim) = accuracies_1NN(11, iterDim) + accuracy;% 	
%		
% 		fprintf(' LPPCosEigenvalues,');
% 		[VecLPP W] = trainLPPEigenvalues(fea_Train, gnd_Train, dimensionRange(iterDim), 'Cosine', 'KNN');
% 		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLPP, fea_Train, fea_Test);
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
% 		accuracies(12, iterDim) = accuracies(12, iterDim) + accuracy;
% 		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% 		accuracies_1NN(12, iterDim) = accuracies_1NN(12, iterDim) + accuracy;
		
		fprintf(' NPE,');
		VecNPE = trainNPE(fea_Train, gnd_Train, dimensionRange(iterDim), 'KNN');
		[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecNPE, fea_Train, fea_Test);
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, nNN, gnd_Test, gnd_Train);
		accuracies(11, iterDim) = accuracies(11, iterDim) + accuracy;
		[results accuracy] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
		accuracies_1NN(11, iterDim) = accuracies_1NN(11, iterDim) + accuracy;
		
		fprintf('\n');
	end;
end;

accuracies = accuracies / numSplits;
accuracies_1NN = accuracies_1NN / numSplits;
