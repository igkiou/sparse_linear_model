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
% [a1 b1 c1 d1] = run_svm(fea_Train_SIFT_Norm', gnd_Train', fea_Test_SIFT_Norm', gnd_Test', [], [], params);
% 
% params.svmLossFunction = 'huber';
% [a1 b1 c1 d1] = run_svm(fea_Train_SIFT_Norm', gnd_Train', fea_Test_SIFT_Norm', gnd_Test', [], [], params);
% 
% [result accuracy] = knn_classify(fea_Test_SIFT_Norm, fea_Train_SIFT_Norm, 1, gnd_Test, gnd_Train);
% fprintf('KNN accuracy: %lf\n', accuracy);

%%
disp('Segment 2');
A_Train_Kernel = l1kernel_featuresign_mex(fea_Train_SIFT_Norm', D_kernel, kernelgradientparams.codinglambda, 'g', [], sigma);
A_Test_Kernel = l1kernel_featuresign_mex(fea_Test_SIFT_Norm', D_kernel, kernelgradientparams.codinglambda, 'g', [], sigma);

params = setParameters;
params.svmLossFunction = 'hinge';
[a1 b1 c1 d1] = run_svm(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);

params.svmLossFunction = 'huber';
[a1 b1 c1 d1] = run_svm(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);

% [result accuracy] = knn_classify(A_Test_Kernel', A_Train_Kernel', 1, gnd_Test, gnd_Train);
% fprintf('KNN accuracy: %lf\n', accuracy);

%%
disp('Segment 3');
lassoparams.lambda = 0.3;
A_Train_Kernel = l1kernel_featuresign_mex(fea_Train_SIFT_Norm', D_Graz_kernel, kernelgradientparams.codinglambda, 'g', [], sigma);
A_Test_Kernel = l1kernel_featuresign_mex(fea_Test_SIFT_Norm', D_Graz_kernel, kernelgradientparams.codinglambda, 'g', [], sigma);

params = setParameters;
params.svmLossFunction = 'hinge';
[a1 b1 c1 d1] = run_svm(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);

params.svmLossFunction = 'huber';
[a1 b1 c1 d1] = run_svm(A_Train_Kernel, gnd_Train', A_Test_Kernel, gnd_Test', [], [], params);

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

