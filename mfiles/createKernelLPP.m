system(sprintf('scp igkiou@140.247.62.217:MATLAB/sparse_linear_model/brodatz_experiments/brodatz_All_norm0_patch8_large1_distinct.mat brodatz_experiments/'));
system(sprintf('scp igkiou@140.247.62.217:MATLAB/sparse_linear_model/brodatz_experiments/brodatz_All_norm0_patch10_large1_distinct.mat brodatz_experiments/'));

load brodatz_experiments/brodatz_All_norm0_patch8_large1_distinct
gramTrainTrain = kernel_gram_mex(trainFeatures, [], 'g', 8);
[Vec W options] = trainKernelSRLPP(trainFeatures', trainLabels', 100, 'Euclidean', 'KNN', gramTrainTrain);
gramTrainTest = kernel_gram_mex(trainFeatures, testFeatures, 'g', 8);
fea_Train_Reduced_Large = gramTrainTrain * Vec;
fea_Test_Reduced_Large = gramTrainTest' * Vec;
save brodatz_experiments/brodatz_All_LPP100_norm0_patch8_large1_distinct.mat fea_Train_Reduced_Large fea_Test_Reduced_Large
clear all

load brodatz_experiments/brodatz_All_norm0_patch10_large1_distinct
gramTrainTrain = kernel_gram_mex(trainFeatures, [], 'g', 8);
[Vec W options] = trainKernelSRLPP(trainFeatures', trainLabels', 100, 'Euclidean', 'KNN', gramTrainTrain);
gramTrainTest = kernel_gram_mex(trainFeatures, testFeatures, 'g', 8);
fea_Train_Reduced_Large = gramTrainTrain * Vec;
fea_Test_Reduced_Large = gramTrainTest' * Vec;
save brodatz_experiments/brodatz_All_LPP100_norm0_patch10_large1_distinct.mat fea_Train_Reduced_Large fea_Test_Reduced_Large
clear all
