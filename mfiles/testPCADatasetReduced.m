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

accuracy_PCA_hinge = zeros(1, numDims);
accuracy_PCA_huber = zeros(1, numDims);
accuracy_PCA_knn = zeros(1, numDims);
accuracy_PCA_kernel_hinge = zeros(1, numDims);
accuracy_PCA_kernel_huber = zeros(1, numDims);
accuracy_PCA_kernel_knn = zeros(1, numDims);

% [fea_Train_Reduced_PCA, mapping] = kernel_pca(fea_Train_SIFT_Norm, 100, 'gauss', sigma);
% fea_Test_Reduced_PCA = out_of_sample(fea_Test_SIFT_Norm, mapping);
% disp('gramTrainTrain.');
% gramTrainTrain = kernel_gram(trainFeatures, [], 'g', sigma);
% disp('Large PCA.');
% [fea_Train_Reduced_Large, mapping] = kernel_pca_custom(trainFeatures, reducedDim(end), gramTrainTrain, 'g', sigma);
% clear gramTrainTrain
% fea_Train_Reduced_Large = fea_Train_Reduced_Large';
% disp('gramTrainTest.');
% gramTrainTest = kernel_gram(trainFeatures, testFeatures, 'g', sigma);
% fea_Test_Reduced_Large = kernel_pca_oos(testFeatures, gramTrainTest, mapping)';
% clear gramTrainTest
load tiny_experiments/cifar_32x32_PCA_Reduced_KernelPCA
fid = fopen(sprintf('%s_runs_reduced_pca.txt', datasetNameSecondary), 'wt');

options.ReducedDim = reducedDim(end);
[Vec eigVal sampleMean] = PCA(trainFeatures', options);
clear eigVal
[fea_Train_Large_PCA fea_Test_Large_PCA] = reduceDimension(Vec, trainFeatures', testFeatures', sampleMean); 

for iterDim = 1:numDims,
	fprintf('Dimension %d our of %d. ', iterDim, numDims);

	fea_Train_Reduced = fea_Train_Large_PCA(:, 1:reducedDim(iterDim));
	fea_Test_Reduced = fea_Test_Large_PCA(:, 1:reducedDim(iterDim));
	svmparams.svmLossFunction = 'hinge';
	fprintf('PCA hinge. ');
	accuracy_PCA_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('PCA huber. ');
	accuracy_PCA_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	fprintf('PCA KNN. ');
	[foo accuracy_PCA_knn(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testLabels', trainLabels');

	fea_Train_Reduced = fea_Train_Reduced_Large(:, 1:reducedDim(iterDim));
	fea_Test_Reduced = fea_Test_Reduced_Large(:, 1:reducedDim(iterDim));
	svmparams.svmLossFunction = 'hinge';
	fprintf('PCA kernel hinge. ');
	accuracy_PCA_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('PCA kernel huber. ');
	accuracy_PCA_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	fprintf('PCA kernel KNN. ');
	[foo accuracy_PCA_kernel_knn(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testLabels', trainLabels');

	fprintf('\n');
end;
fprintf(fid, 'PCA\n');
fprintf(fid, '%g ', accuracy_PCA_hinge);
fprintf(fid, '\n');
fprintf(fid, '%g ', accuracy_PCA_huber);
fprintf(fid, '\n');
fprintf(fid, '%g ', accuracy_PCA_knn);
fprintf(fid, '\n');
fprintf(fid, 'Kernel PCA\n');
fprintf(fid, '%g ', accuracy_PCA_kernel_hinge);
fprintf(fid, '\n');
fprintf(fid, '%g ', accuracy_PCA_kernel_huber);
fprintf(fid, '\n');
fprintf(fid, '%g ', accuracy_PCA_kernel_knn);
fprintf(fid, '\n');
fclose(fid);
