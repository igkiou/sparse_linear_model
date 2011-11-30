load brodatz_experiments/brodatzDataAllUnnorm
load brodatz_experiments/brodatzKernelDictionaryAll_iter300_gamma128
D_kernel = D;
load brodatz_experiments/brodatzGaussianDictionaryAll_memory
gamma = 1/128;
sigma = sqrt(0.5/gamma);
reducedDim = [2 5 10:10:60 64];
numDims = length(reducedDim);
accuracy_PCA_hinge = zeros(1, numDims);
accuracy_PCA_huber = zeros(1, numDims);
accuracy_D_hinge = zeros(1, numDims);
accuracy_D_huber = zeros(1, numDims);
accuracy_PCA_kernel_hinge = zeros(1, numDims);
accuracy_PCA_kernel_huber = zeros(1, numDims);
accuracy_D_kernel_hinge = zeros(1, numDims);
accuracy_D_kernel_huber = zeros(1, numDims);
svmparams = setParameters;
svmparams.allVerboseMode = -1;

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
load brodatz_experiments/reducedbrodatz

disp('gramDD.');
gramDD = kernel_gram(D_kernel, [], 'g', sigma);
disp('gramTrainD.');
gramTrainD = kernel_gram(trainFeatures, D_kernel, 'g', sigma);
disp('gramTestD.');
gramTestD = kernel_gram(testFeatures, D_kernel, 'g', sigma);

for iterDim = 1:numDims,
	fprintf('Dimension %d our of %d. ', iterDim, numDims);
	
	options.ReducedDim = reducedDim(iterDim);
	[Vec eigVal sampleMean] = PCA(trainFeatures', options);
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, trainFeatures', testFeatures', sampleMean); 
	svmparams.svmLossFunction = 'hinge';
	fprintf('PCA hinge. ');
	accuracy_PCA_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('PCA huber. ');
	accuracy_PCA_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	
	Vec = learn_sensing_exact(D, reducedDim(iterDim))';
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, trainFeatures', testFeatures'); 
	svmparams.svmLossFunction = 'hinge';
	fprintf('D hinge. ');
	accuracy_D_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('D huber. ');
	accuracy_D_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);

% 	[fea_Train_Reduced, mapping] = kernel_pca_custom(trainFeatures, reducedDim(iterDim), gramTrainTrain, 'g', sigma);
% 	fea_Test_Reduced = kernel_pca_oos(testFeatures, gramTrainTest, mapping)';
	fea_Train_Reduced = fea_Train_Reduced_Large(:, 1:reducedDim(iterDim));
	fea_Test_Reduced = fea_Test_Reduced_Large(:, 1:reducedDim(iterDim));
	svmparams.svmLossFunction = 'hinge';
	fprintf('PCA kernel hinge. ');
	accuracy_PCA_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('PCA kernel huber. ');
	accuracy_PCA_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	
	Vec = learn_sensing_exact_kernel(D_kernel, reducedDim(iterDim), [], gramDD, 'g', sigma)';
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimensionKernel(Vec, trainFeatures', testFeatures',...
					D_kernel, gramTrainD, gramTestD, 'g', sigma);
	svmparams.svmLossFunction = 'hinge';
	fprintf('D kernel hinge. ');
	accuracy_D_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('D kernel huber. ');
	accuracy_D_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
	
	fprintf('\n');
end;
save brodatz_experiments/accuracies_small_All_short accuracy*
