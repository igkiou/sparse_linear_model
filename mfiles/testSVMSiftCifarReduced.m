reducedDim = 10:10:100;
numDims = length(reducedDim);
% accuracy_PCA_hinge = zeros(1, numDims);
% accuracy_PCA_huber = zeros(1, numDims);
% accuracy_D_hinge = zeros(1, numDims);
% accuracy_D_huber = zeros(1, numDims);
% accuracy_D_custom_hinge = zeros(1, numDims);
% accuracy_D_custom_huber = zeros(1, numDims);
% accuracy_PCA_Graz_hinge = zeros(1, numDims);
% accuracy_PCA_Graz_huber = zeros(1, numDims);
% accuracy_D_Graz_hinge = zeros(1, numDims);
% accuracy_D_Graz_huber = zeros(1, numDims);
% accuracy_D_Graz_custom_hinge = zeros(1, numDims);
% accuracy_D_Graz_custom_huber = zeros(1, numDims);
accuracy_PCA_kernel_hinge = zeros(1, numDims);
accuracy_PCA_kernel_huber = zeros(1, numDims);
accuracy_D_kernel_hinge = zeros(1, numDims);
accuracy_D_kernel_huber = zeros(1, numDims);
accuracy_PCA_Graz_kernel_hinge = zeros(1, numDims);
accuracy_PCA_Graz_kernel_huber = zeros(1, numDims);
accuracy_D_Graz_kernel_hinge = zeros(1, numDims);
accuracy_D_Graz_kernel_huber = zeros(1, numDims);
svmparams = setParameters;
svmparams.allVerboseMode = -1;

[fea_Train_Reduced_PCA, mapping] = kernel_pca(fea_Train_SIFT_Norm, 100, 'gauss', sigma);
fea_Test_Reduced_PCA = out_of_sample(fea_Test_SIFT_Norm, mapping);

% disp('gramGrazGraz');
% gramGrazGraz = kernel_gram_mex(GrazData', [], 'G', sigma);
% disp('gramGrazTrain');
% gramGrazTrain = kernel_gram_mex(GrazData', fea_Train_SIFT_Norm', 'G', sigma);
% disp('gramGrazTest');
% gramGrazTest = kernel_gram_mex(GrazData', fea_Test_SIFT_Norm', 'G', sigma);
disp('gramTrainD');
gramTrainD = kernel_gram_mex(fea_Train_SIFT_Norm', D_kernel, 'G', sigma);
disp('gramTestD');
gramTestD = kernel_gram_mex(fea_Test_SIFT_Norm', D_kernel, 'G', sigma);
disp('gramTrainDGraz');
gramTrainDGraz = kernel_gram_mex(fea_Train_SIFT_Norm', D_Graz_kernel, 'G', sigma);
disp('gramTestDGraz');
gramTestDGraz = kernel_gram_mex(fea_Test_SIFT_Norm', D_Graz_kernel, 'G', sigma);

for iterDim = 1:numDims,
	fprintf('Dimension %d our of %d. ', iterDim, numDims);
	
% 	options.ReducedDim = reducedDim(iterDim);
% 	[Vec eigVal sampleMean] = PCA(fea_Train_SIFT_Norm, options);
% 	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train_SIFT_Norm, fea_Test_SIFT_Norm, sampleMean); 
% 	svmparams.svmLossFunction = 'hinge';
% 	fprintf('PCA hinge. ');
% 	accuracy_PCA_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 	svmparams.svmLossFunction = 'huber';
% 	fprintf('PCA huber. ');
% 	accuracy_PCA_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 	
% 	Vec = learn_sensing_exact(D, reducedDim(iterDim))';
% 	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train_SIFT_Norm, fea_Test_SIFT_Norm); 
% 	svmparams.svmLossFunction = 'hinge';
% 	fprintf('D hinge. ');
% 	accuracy_D_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 	svmparams.svmLossFunction = 'huber';
% 	fprintf('D huber. ');
% 	accuracy_D_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 	
% 	Vec = learn_sensing_exact(D_custom, reducedDim(iterDim))';
% 	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train_SIFT_Norm, fea_Test_SIFT_Norm); 
% 	svmparams.svmLossFunction = 'hinge';
% 	fprintf('D_custom hinge. ');
% 	accuracy_D_custom_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 	svmparams.svmLossFunction = 'huber';
% 	fprintf('D_custom huber. ');
% 	accuracy_D_custom_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 	
% 	options.ReducedDim = reducedDim(iterDim);
% 	[Vec eigVal sampleMean] = PCA(GrazData', options);
% 	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train_SIFT_Norm, fea_Test_SIFT_Norm, sampleMean); 
% 	svmparams.svmLossFunction = 'hinge';
% 	fprintf('PCA hinge. ');
% 	accuracy_PCA_Graz_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 	svmparams.svmLossFunction = 'huber';
% 	fprintf('PCA huber. ');
% 	accuracy_PCA_Graz_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 
% 	Vec = learn_sensing_exact(D_Graz, reducedDim(iterDim))';
% 	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train_SIFT_Norm, fea_Test_SIFT_Norm); 
% 	svmparams.svmLossFunction = 'hinge';
% 	fprintf('D Graz hinge. ');
% 	accuracy_D_Graz_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 	svmparams.svmLossFunction = 'huber';
% 	fprintf('D Graz huber. ');
% 	accuracy_D_Graz_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 
% 	Vec = learn_sensing_exact(D_Graz_custom, reducedDim(iterDim))';
% 	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train_SIFT_Norm, fea_Test_SIFT_Norm); 
% 	svmparams.svmLossFunction = 'hinge';
% 	fprintf('D Graz custom hinge. ');
% 	accuracy_D_Graz_custom_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
% 	svmparams.svmLossFunction = 'huber';
% 	fprintf('D Graz custom huber. ');
% 	accuracy_D_Graz_custom_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);

	fea_Train_Reduced = fea_Train_Reduced_PCA(:, 1:reducedDim(iterDim));
	fea_Test_Reduced = fea_Test_Reduced_PCA(:, 1:reducedDim(iterDim));
	svmparams.svmLossFunction = 'hinge';
	fprintf('PCA kernel hinge. ');
	accuracy_PCA_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('PCA kernel huber. ');
	accuracy_PCA_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);

	[foo, mapping] = kernel_pca_custom(GrazData', reducedDim(iterDim), gramGrazGraz, 'g', sigma);
	fea_Train_Reduced = kernel_pca_oos(fea_Train_SIFT_Norm', gramGrazTrain, mapping)';
	fea_Test_Reduced = kernel_pca_oos(fea_Test_SIFT_Norm', gramGrazTest, mapping)';
	svmparams.svmLossFunction = 'hinge';
	fprintf('PCA kernel hinge. ');
	accuracy_PCA_Graz_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('PCA kernel huber. ');
	accuracy_PCA_Graz_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
	
	Vec = learn_sensing_exact(D_kernel, reducedDim(iterDim))';
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimensionKernel(Vec, fea_Train_SIFT_Norm, fea_Test_SIFT_Norm,...
					D_kernel, gramTrainD, gramTestD, 'g', sigma);
	svmparams.svmLossFunction = 'hinge';
	fprintf('D kernel hinge. ');
	accuracy_D_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('D kernel huber. ');
	accuracy_D_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);

	Vec = learn_sensing_exact(D_Graz_kernel, reducedDim(iterDim))';
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimensionKernel(Vec, fea_Train_SIFT_Norm, fea_Train_SIFT_Norm,...
					D_kernel, gramTrainDGraz, gramTestDGraz, 'g', sigma);
	svmparams.svmLossFunction = 'hinge';
	fprintf('D Graz kernel hinge. ');
	accuracy_D_Graz_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('D Graz kernel huber. ');
	accuracy_D_Graz_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', [], [], svmparams);
	
	fprintf('\n');
end;
