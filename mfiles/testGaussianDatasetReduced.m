% datasetName = 'tiny';
% datasetNameSecondary = 'tinySIFT';
% load tiny_experiments/cifar_32x32_SIFT.mat fea_Train_SIFT_Norm fea_Test_SIFT_Norm
% load /home/igkiou/MATLAB/datasets_all/cifar-10-batches-mat/cifar_32x32.mat gnd_Train gnd_Test
% trainFeatures = fea_Train_SIFT_Norm';
% testFeatures = fea_Test_SIFT_Norm';
datasetName = 'tiny';
datasetNameSecondary = 'tinyPCA';
load tiny_experiments/cifar_32x32_PCA.mat fea_Train_PCA fea_Test_PCA
load /home/igkiou/MATLAB/datasets_all/cifar-10-batches-mat/cifar_32x32.mat gnd_Train gnd_Test
trainFeatures = fea_Train_PCA';
testFeatures = fea_Test_PCA';
trainLabels = gnd_Train';
testLabels = gnd_Test';
svmparams = setParameters;
svmparams.allVerboseMode = -1;
reducedDim = [2 5 10:10:60 64];
numDims = length(reducedDim);

folder_contents = ls(sprintf('%s_experiments/%sGaussianDictionary*', datasetName, datasetNameSecondary));
tempNames = regexp(folder_contents, '\s\s*', 'split');
numDicts = length(tempNames);
fid = fopen(sprintf('%s_runs_reduced_gaussian.txt', datasetNameSecondary), 'wt');
for iterDict = 1:(numDicts - 1),
	load(sprintf('%s', tempNames{iterDict}));
	fprintf(fid, '%s\n', tempNames{iterDict});
	fprintf('Dictionary %s. ', tempNames{iterDict});
	D_gaussian = D;
	accuracy_D_hinge = zeros(1, numDims);
	accuracy_D_huber = zeros(1, numDims);
	accuracy_D_knn = zeros(1, numDims);
	Vec = learn_sensing_exact(D_gaussian, reducedDim(end))';
	[fea_Train_Reduced_Large fea_Test_Reduced_Large] = reduceDimension(Vec, trainFeatures', testFeatures'); 

	for iterDim = 1:numDims,
		fprintf('Dimension %d our of %d. ', iterDim, numDims);
		fea_Train_Reduced = fea_Train_Reduced_Large(:, 1:reducedDim(iterDim));
		fea_Test_Reduced = fea_Test_Reduced_Large(:, 1:reducedDim(iterDim));
% 		svmparams.svmLossFunction = 'hinge';
% 		fprintf('D hinge. ');
% 		accuracy_D_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
% 		svmparams.svmLossFunction = 'huber';
% 		fprintf('D huber. ');
% 		accuracy_D_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
% 		fprintf('\n');
		fprintf('D KNN. ');
		[foo accuracy_D_knn(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testLabels', trainLabels');
		fprintf('\n');
	end;
	fprintf(fid, '%g ', accuracy_D_hinge);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_D_huber);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_D_knn);
	fprintf(fid, '\n');
	clear D params memoryparams gradientparams
end;
fclose(fid);
