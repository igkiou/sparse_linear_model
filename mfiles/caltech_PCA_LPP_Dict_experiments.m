%%
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_feataverage39.mat
K_train_train = K;
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_test_feataverage39.mat
K_test_train = K;
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_labels
K_train_train = double(K_train_train);
K_test_train = double(K_test_train);
KXX = K_train_train;
KYX = K_test_train;

% %%
% params.codinglambda = 0.1;
% params.iternum = 100;
% params.blockratio = 0.1;
% params.printinfo = 1;
% params.errorinfo = 0;
% params.distclear = 0;
% params.dictsize = 1000;
% params.initdict = randn(3060,1000);
% D = kerneldictgradient_representer(KXX, params);
% 
% %%
% % gramTrainTrain = KXX;
% constructKernelW
% Vec = trainKernelSRLPP([], [], 1000, 'Cosine', 'KNN', KXX, WEu);
% fea_Train_Reduced_Large = KXX * Vec;
% fea_Train_Reduced_Large = fea_Train_Reduced_Large';
% fea_Test_Reduced_Large = K_test_train * Vec;
% fea_Test_Reduced_Large = fea_Test_Reduced_Large';
% % KXXLPP = l2_distance(fea_Train_Reduced_Large);
% % KXXLPP = - KXXLPP;
% 
% %%
% [fea_Train_Reduced_Large_PCA, mapping] = kernel_pca_custom(K_train_train, 1000, K_train_train);
% fea_Test_Reduced_Large_PCA = kernel_pca_oos([], K_test_train', mapping);
% 
%%
% load newdicts1 D1
% D = D1;
% KDD = D'*K_train_train*D;
% KDD = (KDD + KDD') / 2;
% Vec = learn_sensing_exact_kernel(D, 500, [], KDD)';
% KXX = K_train_train;
% KXD = KXX * D;
% KYD = KYX * D;
% [fea_Train_Reduced_Large_Dict1 fea_Test_Reduced_Large_Dict1] = reduceDimensionKernel(Vec, [], NaN, [], KXD, KYD);
% fea_Train_Reduced_Large_Dict1 = fea_Train_Reduced_Large_Dict1';
% fea_Test_Reduced_Large_Dict1 = fea_Test_Reduced_Large_Dict1';

% %%
% load newdicts1 D
% KDD = D'*K_train_train*D;
% KDD = (KDD + KDD') / 2;
% Vec = learn_sensing_exact_kernel(D, 500, [], KDD)';
% KXX = K_train_train;
% KXD = KXX * D;
% KYX = K_test_train;
% KYD = KYX * D;
% [fea_Train_Reduced_Large_Dict2 fea_Test_Reduced_Large_Dict2] = reduceDimensionKernel(Vec, [], NaN, [], KXD, KYD);
% fea_Train_Reduced_Large_Dict2 = fea_Train_Reduced_Large_Dict2';
% fea_Test_Reduced_Large_Dict2 = fea_Test_Reduced_Large_Dict2';

%%
load newdicts3 D
% D = D2;
KDD = D'*K_train_train*D;
KDD = (KDD + KDD') / 2;
% lvec = eig(KDD);
% beta = lvec(2) / 10;
% KDD = KDD + beta * eye(size(KDD));
Vec = learn_sensing_exact_kernel(D, 1000, [], KDD)';
KXX = K_train_train;
KXD = KXX * D;
KYX = K_test_train;
KYD = KYX * D;
[fea_Train_Reduced_Large_Dict3 fea_Test_Reduced_Large_Dict3] = reduceDimensionKernel(Vec, [], NaN, [], KXD, KYD);
fea_Train_Reduced_Large_Dict3 = fea_Train_Reduced_Large_Dict3';
fea_Test_Reduced_Large_Dict3 = fea_Test_Reduced_Large_Dict3';

%%
params = setParameters;
params.svmLossFunction = 'hinge';
dimensions = [10:10:90 100:100:1000];
numDimensions = length(dimensions);
accuracy = zeros(3, numDimensions);
for iterD = 1:numDimensions,
	fprintf('Now doing dimension %d/%d, %d.\n', iterD, numDimensions, dimensions(iterD));
% 	if (dimensions(iterD) <= 500),
% 		feaTrain = fea_Train_Reduced_Large_Dict1(1:dimensions(iterD), :);
% 		feaTest = fea_Test_Reduced_Large_Dict1(1:dimensions(iterD), :);
% 		accuracy(1, iterD) = run_svm(feaTrain, tr_label', feaTest, te_label', params);
% 
% 		feaTrain = fea_Train_Reduced_Large_Dict2(1:dimensions(iterD), :);
% 		feaTest = fea_Test_Reduced_Large_Dict2(1:dimensions(iterD), :);
% 		accuracy(2, iterD) = run_svm(feaTrain, tr_label', feaTest, te_label', params);
% 	end;
	
	feaTrain = fea_Train_Reduced_Large_Dict3(1:dimensions(iterD), :);
	feaTest = fea_Test_Reduced_Large_Dict3(1:dimensions(iterD), :);
	accuracy(3, iterD) = run_svm(feaTrain, tr_label', feaTest, te_label', params);
end;
