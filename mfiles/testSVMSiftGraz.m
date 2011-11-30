% reducedDim = 20;
% options.ReducedDim = reducedDim;
% [Vec eigVal sampleMean] = PCA(trainX', options);
% [trainXr testXr] = reduceDimension(Vec, trainX', testX', sampleMean); trainXr = trainXr'; testXr = testXr';
% [acc model labels stdVec] = run_svm(trainXr, trainL, testXr, testL, [], [], params);
% 
% load SIFTGaussianDictionary
% Phi = learn_sensing_exact(D, reducedDim);
% [trainXr testXr] = reduceDimension(Phi', trainX', testX'); trainXr = trainXr'; testXr = testXr';
% [acc model labels stdVec] = run_svm(trainXr, trainL, testXr, testL, [], [], params);
% 
% [acc model labels stdVec] = run_svm(trainX, trainL, testX, testL, [], [], params);
% 
% trainA = l1qp_featuresign_mex(trainX, D, 0.3);
% testA = l1qp_featuresign_mex(testX, D, 0.3);
% [acc model labels stdVec] = run_svm(trainA, trainL, testA, testL, [], [], params);

%%
% disp('Segment 1');
% params = setParameters;
% params.svmLossFunction = 'hinge';
% [a1 b1 c1 d1] = run_svm(fea_Train', gnd_Train', fea_Test', gnd_Test', [], [], params);
% 
% params.svmLossFunction = 'huber';
% [a1 b1 c1 d1] = run_svm(fea_Train', gnd_Train', fea_Test', gnd_Test', [], [], params);

% [result accuracy] = knn_classify(fea_Test, fea_Train, 1, gnd_Test, gnd_Train);
% fprintf('KNN accuracy: %lf\n', accuracy);

%%
% disp('Segment 2');
% lassoparams.lambda = 0.3;
% A_Train_Lasso = mexLasso(fea_Train', D_Graz, lassoparams);
% A_Test_Lasso = mexLasso(fea_Test', D_Graz, lassoparams);
% 
% params = setParameters;
% params.svmLossFunction = 'hinge';
% [a1 b1 c1 d1] = run_svm(A_Train_Lasso, gnd_Train', A_Test_Lasso, gnd_Test', [], [], params);
% 
% params.svmLossFunction = 'huber';
% [a1 b1 c1 d1] = run_svm(A_Train_Lasso, gnd_Train', A_Test_Lasso, gnd_Test', [], [], params);
% 
% % [result accuracy] = knn_classify(A_Test_Lasso', A_Train_Lasso', 1, gnd_Test, gnd_Train);
% % fprintf('KNN accuracy: %lf\n', accuracy);
% 
%%
% disp('Segment 3');
% lassoparams.lambda = 0.3;
% A_Train_Lasso = mexLasso(fea_Train', D_Graz_custom, lassoparams);
% A_Test_Lasso = mexLasso(fea_Test', D_Graz_custom, lassoparams);
% 
% params = setParameters;
% params.svmLossFunction = 'hinge';
% [a1 b1 c1 d1] = run_svm(A_Train_Lasso, gnd_Train', A_Test_Lasso, gnd_Test', [], [], params);
% 
% params.svmLossFunction = 'huber';
% [a1 b1 c1 d1] = run_svm(A_Train_Lasso, gnd_Train', A_Test_Lasso, gnd_Test', [], [], params);
% 
% % [result accuracy] = knn_classify(A_Test_Lasso', A_Train_Lasso', 1, gnd_Test, gnd_Train);
% % fprintf('KNN accuracy: %lf\n', accuracy);
% 
%%
% disp('Segment 4');
% lassoparams.lambda = 0.3;
% A_Train_Lasso = l1qp_featuresign_mex(fea_Train', D_Graz_custom, lassoparams);
% A_Test_Lasso = l1qp_featuresign_mex(fea_Test', D_Graz_custom, lassoparams);
% 
% params = setParameters;
% params.svmLossFunction = 'hinge';
% [a1 b1 c1 d1] = run_svm(A_Train_Lasso, gnd_Train', A_Test_Lasso, gnd_Test', [], [], params);
% 
% params.svmLossFunction = 'huber';
% [a1 b1 c1 d1] = run_svm(A_Train_Lasso, gnd_Train', A_Test_Lasso, gnd_Test', [], [], params);

% [result accuracy] = knn_classify(A_Test_Lasso', A_Train_Lasso', 1, gnd_Test, gnd_Train);
% fprintf('KNN accuracy: %lf\n', accuracy);
%
%%
% reducedDim = 20;
% options.ReducedDim = reducedDim;
% [Vec eigVal sampleMean] = PCA(trainX', options);
% [trainXr testXr] = reduceDimension(Vec, trainX', testX', sampleMean); trainXr = trainXr'; testXr = testXr';
% [acc model labels stdVec] = run_svm(trainXr, trainL, testXr, testL, [], [], params);
% 
% load SIFTGaussianDictionary
% Phi = learn_sensing_exact(D, reducedDim);
% [trainXr testXr] = reduceDimension(Phi', trainX', testX'); trainXr = trainXr'; testXr = testXr';
% [acc model labels stdVec] = run_svm(trainXr, trainL, testXr, testL, [], [], params);
% 
% [acc model labels stdVec] = run_svm(trainX, trainL, testX, testL, [], [], params);
% 
% trainA = l1qp_featuresign_mex(trainX, D, 0.3);
% testA = l1qp_featuresign_mex(testX, D, 0.3);
% [acc model labels stdVec] = run_svm(trainA, trainL, testA, testL, [], [], params);

%%
% disp('Segment 2');
% A_Train_Kernel = l1kernel_featuresign_mex(trainSIFTFeatures, D_kernel, kernelgradientparams.codinglambda, 'g', [], sigma);
% A_Test_Kernel = l1kernel_featuresign_mex(testSIFTFeatures', D_kernel, kernelgradientparams.codinglambda, 'g', [], sigma);
% 
% params = setParameters;
% params.svmLossFunction = 'hinge';
% [a1 b1 c1 d1] = run_svm(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);
% 
% params.svmLossFunction = 'huber';
% [a1 b1 c1 d1] = run_svm(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);

% [result accuracy] = knn_classify(A_Test_Kernel', A_Train_Kernel', 1, gnd_Test, gnd_Train);
% fprintf('KNN accuracy: %lf\n', accuracy);

%%
disp('Segment 3');
lassoparams.lambda = 0.3;
% A_Train_Kernel = l1kernel_featuresign_mex(trainSIFTFeatures, D_Graz_kernel, kernelgradientparams.codinglambda, 'g', [], sigma);
% A_Test_Kernel = l1kernel_featuresign_mex(testSIFTFeatures, D_Graz_kernel, kernelgradientparams.codinglambda, 'g', [], sigma);

params = setParameters;
params.svmLossFunction = 'hinge';
[a1 b1 c1 d1] = run_svm_binary(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);

params.svmLossFunction = 'huber';
[a1 b1 c1 d1] = run_svm_binary(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);

% [result accuracy] = knn_classify(A_Test_Kernel', A_Train_Kernel', 1, gnd_Test, gnd_Train);
% fprintf('KNN accuracy: %lf\n', accuracy);

%%
% disp('Segment 4');
% lassoparams.lambda = 0.3;
% A_Train_Kernel = l1qp_featuresign_mex(fea_Train_SIFT_Norm', D_Graz_custom, lassoparams);
% A_Test_Kernel = l1qp_featuresign_mex(fea_Test_SIFT_Norm', D_Graz_custom, lassoparams);
% 
% params = setParameters;
% params.svmLossFunction = 'hinge';
% [a1 b1 c1 d1] = run_svm(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);
% 
% params.svmLossFunction = 'huber';
% [a1 b1 c1 d1] = run_svm(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);
% 
% [result accuracy] = knn_classify(A_Test_Kernel', A_Train_Kernel', 1, gnd_Test, gnd_Train);
% fprintf('KNN accuracy: %lf\n', accuracy);

