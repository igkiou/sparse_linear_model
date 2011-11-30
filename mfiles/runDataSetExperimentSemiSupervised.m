function accuracies = ...
	runDataSetExperimentSemiSupervised(dataSet, trainingSize, labeledSize, splits, useNorm, nNN, saveOption)

if (nargin < 2),
	trainingSize = 5;
end;

if (nargin < 3),
	labeledSize = 10;
end;

if (nargin < 4),
	splits = 1;
end;

if (nargin < 5),
	useNorm = 1;
end;

if (nargin < 6),
	nNN = 1;
end;

if (nargin < 7),
	saveOption = 1;
end;

load(strcat('~/MATLAB/datasets_all/', dataSet, '/', dataSet, '_32x32.mat'));
load(strcat('~/MATLAB/sparse_linear_model/', dataSet, '_experiments/', num2str(trainingSize), 'TrainNew.mat'));

eval(sprintf('[fea_Train gnd_Train fea_Test gnd_Test] = loadDataSet(dataSet, trainingSize, useNorm, splits%d(1), fea, gnd);', trainingSize));

if (useNorm == 1),
	fea_Train = normcols(fea_Train')';
	fea_Test = normcols(fea_Test')';
end;
load(strcat('~/MATLAB/sparse_linear_model/', dataSet, '_experiments/semisplits', num2str(labeledSize), '.mat'));
% load(strcat('~/MATLAB/sparse_linear_model/', dataSet, '_experiments/semisplits', num2str(labeledSize), '_40.mat'));
uniqueLabels = length(unique(gnd_Train));
% dimension = ceil(uniqueLabels / 10) * 10;
dimension = uniqueLabels - 1;
load(strcat('~/MATLAB/sparse_linear_model/', dataSet, '_experiments/dictionaryPhi', num2str(dimension), 'origNorm.mat'));

numSplits = length(splits);
accuracies = zeros(16, numSplits);

VecLPPEuclidean = trainLPP(fea_Train, gnd_Train, dimension, 'Euclidean', 'KNN');
VecLPPCosine = trainLPP(fea_Train, gnd_Train, dimension, 'Cosine', 'KNN');
VecSRLPPEuclidean = trainSRLPP(fea_Train, gnd_Train, dimension, 'Euclidean', 'KNN');
VecSRLPPCosine = trainSRLPP(fea_Train, gnd_Train, dimension, 'Cosine', 'KNN');
VecPCA = trainPCA(fea_Train, gnd_Train, dimension);

for iterSplit = 1:numSplits,
	disp(sprintf('Now running semisplit %d out of %d.', iterSplit, numSplits));
	
% % 	VecLMNNSemiNoInit = lmnn_rect(fea_Train(semisplit(:, splits(iterSplit)), :)', gnd_Train(semisplit(:, splits(iterSplit)), :)',...
% % 						3, randn(dimension, size(fea_Train, 2)), dimension, D)';
% % 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLMNNSemiNoInit, fea_Train, fea_Test);
% % 	[results accuracies(1, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
% % 						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
% % 	
% % 	VecLMNNSemiInit = lmnn_rect(fea_Train(semisplit(:, splits(iterSplit)), :)', gnd_Train(semisplit(:, splits(iterSplit)), :)',...
% % 						3, Phi, dimension, D)';
% % 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLMNNSemiInit, fea_Train, fea_Test);
% % 	[results accuracies(2, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
% % 						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
% % 				
% % 	VecLMNNNoInit = lmnn_rect(fea_Train(semisplit(:, splits(iterSplit)), :)', gnd_Train(semisplit(:, splits(iterSplit)), :)',...
% % 						3, randn(dimension, size(fea_Train, 2)), dimension)';
% % 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLMNNNoInit, fea_Train, fea_Test);
% % 	[results accuracies(3, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
% % 						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
% % 	
% % 	VecLMNNInit = lmnn_rect(fea_Train(semisplit(:, splits(iterSplit)), :)',gnd_Train(semisplit(:, splits(iterSplit)), :)',...
% % 						3, Phi, dimension)';				
% % 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLMNNInit, fea_Train, fea_Test);
% % 	[results accuracies(4, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
% % 						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
	
	VecSDA = trainSDA(fea_Train, gnd_Train, semisplit(:, splits(iterSplit)), 'Euclidean', 'KNN');
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSDA, fea_Train, fea_Test);
	[results accuracies(5, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
	
	VecLDA = trainLDA(fea_Train(semisplit(:, splits(iterSplit)), :), gnd_Train(semisplit(:, splits(iterSplit)), :));
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLDA, fea_Train, fea_Test);
	[results accuracies(6, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
	
	VecSDA = trainSDA(fea_Train, gnd_Train, semisplit(:, splits(iterSplit)), 'Cosine', 'KNN');
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSDA, fea_Train, fea_Test);
	[results accuracies(7, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
	
% % 	[VecSDA_MR foo MRoptions] = trainSDA_MR(fea_Train, gnd_Train, semisplit(:, splits(iterSplit)), 'Euclidean', 'KNN');
% % 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionKernel(VecSDA_MR, fea_Train, fea_Test, MRoptions);
% % 	[results accuracies(8, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
% % 						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
% % 	
% % 	[VecSDA_MR foo MRoptions] = trainSDA_MR(fea_Train, gnd_Train, semisplit(:, splits(iterSplit)), 'Cosine', 'KNN');
% % 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimensionKernel(VecSDA_MR, fea_Train, fea_Test, MRoptions);
% % 	[results accuracies(9, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
% % 						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
% % 					
% % 	VecIncoherence = trainSRLDA_incoherence1(fea_Train(semisplit(:, splits(iterSplit)), :), gnd_Train(semisplit(:, splits(iterSplit)), :), D, randn(size(Phi')));
% % 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecIncoherence, fea_Train, fea_Test);
% % 	[results accuracies(10, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
% % 						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
					
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(Phi', fea_Train, fea_Test);
	[results accuracies(11, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
					
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLPPEuclidean, fea_Train, fea_Test);
	[results accuracies(12, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
					
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecLPPCosine, fea_Train, fea_Test);
	[results accuracies(13, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
					
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSRLPPEuclidean, fea_Train, fea_Test);
	[results accuracies(14, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
					
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecSRLPPCosine, fea_Train, fea_Test);
	[results accuracies(15, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
					
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(VecPCA, fea_Train, fea_Test);
	[results accuracies(16, iterSplit)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced(semisplit(:, splits(iterSplit)), :), nNN,...
						gnd_Test, gnd_Train(semisplit(:, splits(iterSplit)), :));
	
	if (saveOption == 1),
		save(strcat('~/MATLAB/sparse_linear_model/', dataSet, '_experiments/semisupervised', num2str(labeledSize),...
			'_iter', num2str(splits(iterSplit)), '.mat'),... 'VecLMNNSemiNoInit', 
			'VecLMNNSemiInit', 'VecLMNNNoInit', 'VecLMNNInit', 'VecSDA', 'VecLDA');
	end;
end;
