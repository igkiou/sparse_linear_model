%%
% slideType = 'sliding';
% largeExperiment = 0;
% useNorm = 0;
% patchSize = [8 10 12];
% class1 = -1;
% class2 = -1;
% if ((class1 < 0) || (class2 < 0)),
% 	classString = 'All';
% else
% 	classString = sprintf('%d%d', class1, class2);
% end;
% load(sprintf('brodatz_experiments/brodatz_%s_norm%d_patch%d_large%d_%s',...
% 	classString, useNorm, patchSize, largeExperiment, slideType),...
% 	'trainFeatures', 'testFeatures');
load pcadebug trainFeatures testFeatures
fea_Train = trainFeatures';
fea_Test = testFeatures';

gamma = 1/128;
sigma = sqrt(0.5/gamma);
numDimensions = 5;

disp('gramTrainTrain.');
gramTrainTrain = kernel_gram_mex(fea_Train', [], 'g', sigma);

disp('Large PCA.');
disp('Preprocess gramTrainTrain.');
numSamples = size(fea_Train', 2);
mapping.column_sums = sum(gramTrainTrain) / numSamples;
mapping.total_sum   = sum(mapping.column_sums) / numSamples;
% J = ones(numSamples, 1) * mapping.column_sums;
% gramTrainTrain = gramTrainTrain - J - J';
% clear J
% gramTrainTrain = gramTrainTrain + mapping.total_sum;
% gramTrainTrain = (gramTrainTrain + gramTrainTrain') / 2;
gramTrainTrain = bsxfun(@minus, gramTrainTrain, mapping.column_sums);
gramTrainTrain = bsxfun(@minus, gramTrainTrain, mapping.column_sums');
gramTrainTrain = gramTrainTrain + mapping.total_sum;
gramTrainTrain = (gramTrainTrain + gramTrainTrain') / 2;

disp('Do eigendecomposition.');
options = struct('disp',0);
[V, L] = eigs(gramTrainTrain, numDimensions, 'LA', options);
clear gramTrainTrain
[L, ind] = sort(diag(L), 'descend');
L = L(1:numDimensions);
V = V(:,ind(1:numDimensions));

disp('Map training points.');
sqrtL = diag(sqrt(L));
fea_Train_Reduced_Large = sqrtL * V';
fea_Train_Reduced_Large = fea_Train_Reduced_Large';

disp('Prepare "mapping" struct.');
invsqrtL = diag(1 ./ diag(sqrtL));
mapping.trainData = fea_Train';
mapping.V = V;
mapping.invsqrtL = invsqrtL;
mapping.kernelType = 'g';
mapping.kernelParam1 = sigma;
mapping.kernelParam2 = 1;

disp('gramTrainTest.');
gramTrainTest = kernel_gram_mex(fea_Train', fea_Test', 'g', sigma);

disp('Map testing points.');
fea_Test_Reduced_Large = kernel_pca_oos([], gramTrainTest, mapping)';
clear gramTrainTest
% save(sprintf('brodatz_experiments/brodatz_%s_PCA%d_norm%d_patch%d_large%d_%s',...
% 	classString, numDimensions, useNorm, patchSize, largeExperiment, slideType),...
% 	'fea_Train_Reduced_Large', 'fea_Test_Reduced_Large');
% clear all
