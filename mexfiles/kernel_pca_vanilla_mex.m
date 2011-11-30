%%
slideType = 'sliding';
largeExperiment = 0;
useNorm = 0;
patchSize = 8;
class1 = 8;
class2 = 84;
if ((class1 < 0) || (class2 < 0)),
	classString = 'All';
else
	classString = sprintf('%d%d', class1, class2);
end;
numDimensions = 100;
system(sprintf('scp igkiou@140.247.62.217:MATLAB/sparse_classification/brodatz_experiments/brodatz_%s_norm%d_patch%d_large%d_%s.mat brodatz_experiments/',...
	classString, useNorm, patchSize, largeExperiment, slideType));
load(sprintf('brodatz_experiments/brodatz_%s_norm%d_patch%d_large%d_%s',...
	classString, useNorm, patchSize, largeExperiment, slideType),...
	'trainFeatures', 'testFeatures');
% load pcadebug trainFeatures testFeatures
num2use = 70000;
numPerClass = num2use / 2;
numTrain = size(trainFeatures, 2);
numTrainPerClass = numTrain / 2;
inds1 = randperm(numTrainPerClass);
trainFeaturesTemp = trainFeatures(:, inds1(1:numPerClass));
inds2 = randperm(numTrainPerClass);
trainFeaturesTemp = [trainFeaturesTemp trainFeatures(:, numTrainPerClass + (inds2(1:numPerClass)))];
trainFeatures = trainFeaturesTemp;
save(sprintf('brodatz_experiments/brodatz_%s_PCA%d_norm%d_patch%d_large%d_%s_sample',...
	classString, numDimensions, useNorm, patchSize, largeExperiment, slideType),...
	'inds1', 'inds2');
fea_Train = trainFeatures;
fea_Test = testFeatures;
clear trainFeatures testFeatures inds1 inds2 trainFeaturesTemp

gamma = 1/128;
sigma = sqrt(0.5/gamma);
[fea_Train_Reduced_Large fea_Test_Reduced_Large] = kernel_pca_mex(fea_Train, fea_Test, numDimensions, 'g', sigma);
fea_Train_Reduced_Large = fea_Train_Reduced_Large';
fea_Test_Reduced_Large = fea_Test_Reduced_Large';

save(sprintf('brodatz_experiments/brodatz_%s_PCA%d_norm%d_patch%d_large%d_%s',...
	classString, numDimensions, useNorm, patchSize, largeExperiment, slideType),...
	'fea_Train_Reduced_Large', 'fea_Test_Reduced_Large');
clear all
