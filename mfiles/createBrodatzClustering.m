slideType = 'sliding';
useNorm = 0;
patchSize = 8;
largeExperiment = 0;
fiveClasses = {'5c', '5m', '5v', '5v2', '5v3'};
classString = fiveClasses{4};

disp('Started creating dataset.');
I = im2double(imread(sprintf('~/MATLAB/datasets_all/brodatz/training/5-texture/Nat-%s.pgm',... 
	classString)));
trainFeatures = im2col(I, [patchSize patchSize], slideType);
if (useNorm == 1),
	trainFeatures = normcols(trainFeatures);
end;

save(sprintf('brodatz_experiments/brodatz_%s_norm%d_patch%d_large%d_%s',...
	classString, useNorm, patchSize, largeExperiment, slideType),...
	'trainFeatures');

disp('Started training memory Gaussian dictionary.');
memoryparams.K = 512;
memoryparams.batchsize = size(trainFeatures, 2);
memoryparams.iter = 300;
memoryparams.lambda = 0.15;
D = mexTrainDL_Memory(trainFeatures, memoryparams);
save(sprintf('brodatz_experiments/brodatzGaussianDictionary%s_memory_norm%d_patch%d_large%d_%s',...
	classString, useNorm, patchSize, largeExperiment, slideType),...
	'D', 'memoryparams');
clear D
clear memoryparams

disp('Started training kernel dictionary.');
kernelgradientparams.initdict = NaN;
kernelgradientparams.dictsize = 512;
kernelgradientparams.iternum = 300;
kernelgradientparams.iternum2 = 10;
kernelgradientparams.blockratio = 3000 / size(trainFeatures, 2);
kernelgradientparams.codinglambda = 0.1500;
kernelgradientparams.dictclear = 0;
kernelgradientparams.kerneltype = 'G';
kernelgradientparams.kernelparam1 = 8;
kernelgradientparams.kernelparam2 = 1;
kernelgradientparams.printinfo = 1;
kernelgradientparams.errorinfo = 0;
D = kerneldictgradient(trainFeatures, kernelgradientparams);
save(sprintf('brodatz_experiments/brodatzKernelDictionary%s_iter300_norm%d_patch%d_large%d_%s',...
	classString, useNorm, patchSize, largeExperiment, slideType),...
	'D', 'kernelgradientparams');

gamma = 1/128;
sigma = sqrt(0.5/gamma);
numDimensions = 100;
[fea_Train_Reduced_Large fea_Test_Reduced_Large] = kernel_pca_mex(trainFeatures, [], numDimensions, 'g', sigma);
fea_Train_Reduced_Large = fea_Train_Reduced_Large';
fea_Test_Reduced_Large = fea_Test_Reduced_Large';

save(sprintf('brodatz_experiments/brodatz_%s_PCA%d_norm%d_patch%d_large%d_%s',...
	classString, numDimensions, useNorm, patchSize, largeExperiment, slideType),...
	'fea_Train_Reduced_Large', 'fea_Test_Reduced_Large');
clear all
