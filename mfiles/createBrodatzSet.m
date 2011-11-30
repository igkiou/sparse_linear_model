slideType = 'distinct';
largeExperiment = 1;
useNorm = 1;
patchSize = 8;
disp('Started creating dataset.');
[G1 G2] = getTexturePatches(12, patchSize, slideType, useNorm, largeExperiment);
[G3 G4] = getTexturePatches(17, patchSize, slideType, useNorm, largeExperiment);
[G5 G6] = getTexturePatches(5, patchSize, slideType, useNorm, largeExperiment);
[G7 G8] = getTexturePatches(92, patchSize, slideType, useNorm, largeExperiment);
[G9 G10] = getTexturePatches(8, patchSize, slideType, useNorm, largeExperiment);
[G11 G12] = getTexturePatches(84, patchSize, slideType, useNorm, largeExperiment);
[G13 G14] = getTexturePatches(4, patchSize, slideType, useNorm, largeExperiment);
trainFeatures = [G1 G3 G5 G7 G9 G11 G13];
testFeatures = [G2 G4 G6 G8 G10 G12 G14];
trainLabels = [0 * ones(1, size(G1, 2)), 1 * ones(1, size(G3, 2)), 2 * ones(1, size(G5, 2)), 3 * ones(1, size(G7, 2)), ...
	4 * ones(1, size(G9, 2)), 5 * ones(1, size(G11, 2)), 6 * ones(1, size(G13, 2))];
testLabels = [0 * ones(1, size(G2, 2)), 1 * ones(1, size(G4, 2)), 2 * ones(1, size(G6, 2)), 3 * ones(1, size(G8, 2)), ...
	4 * ones(1, size(G10, 2)), 5 * ones(1, size(G12, 2)), 6 * ones(1, size(G14, 2))];
save(sprintf('brodatz_experiments/brodatz_All_norm%d_patch%d_large%d_%s',...
	useNorm, patchSize, largeExperiment, slideType),...
	'trainFeatures', 'testFeatures', 'trainLabels', 'testLabels');

disp('Started training memory Gaussian dictionary.');
memoryparams.K = 512;
memoryparams.batchsize = size(trainFeatures, 2);
memoryparams.iter = 200;
memoryparams.lambda = 0.15;
D = mexTrainDL_Memory(trainFeatures, memoryparams);
save(sprintf('brodatz_experiments/brodatzGaussianDictionaryAll_memory_norm%d_patch%d_large%d_%s',...
	useNorm, patchSize, largeExperiment, slideType),...
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
save(sprintf('brodatz_experiments/brodatzKernelDictionaryAll_iter500_norm%d_patch%d_large%d_%s',...
	useNorm, patchSize, largeExperiment, slideType),...
	'D', 'kernelgradientparams');

% slideType = 'sliding';
% largeExperiment = 0;
% useNorm = 1;
% patchSize = 8;
% class1 = 12;
% class2 = 17;
% 
% disp('Started creating dataset.');
% [G1 G2] = getTexturePatches(class1, patchSize, slideType, useNorm, largeExperiment);
% [G3 G4] = getTexturePatches(class2, patchSize, slideType, useNorm, largeExperiment);
% trainFeatures = [G1 G3];
% Itest = im2double(imread(sprintf('~/MATLAB/datasets_all/brodatz/training/2-texture/D%dD%d.pgm', class1, class2)));
% testFeatures = im2col(Itest, [patchSize patchSize], slideType);
% if (useNorm == 1)
% 	testFeatures = normcols(testFeatures);
% end;
% trainLabels = [ones(1, size(G1, 2)) -1 * ones(1, size(G3, 2))];
% testLabels = [ones(1, ceil(size(testFeatures, 2) / 2)) -1 * ones(1, floor(size(testFeatures, 2) / 2))];
% save(sprintf('brodatz_experiments/brodatz_%d%d_norm%d_patch%d_large%d_%s',...
% 	class1, class2, useNorm, patchSize, largeExperiment, slideType),...
% 	'trainFeatures', 'testFeatures', 'trainLabels', 'testLabels');
% disp('Started training memory Gaussian dictionary.');
% memoryparams.K = 256;
% memoryparams.batchsize = 12000;
% memoryparams.iter = 100;
% memoryparams.lambda = 0.15;
% D = mexTrainDL_Memory(trainFeatures, memoryparams);
% save(sprintf('brodatz_experiments/brodatzGaussianDictionary%d%d_memory_norm%d_patch%d_large%d_%s',...
% 	class1, class2, useNorm, patchSize, largeExperiment, slideType),...
% 	'D', 'memoryparams');
% clear D
% clear memoryparams
% 
% disp('Started training kernel dictionary.');
% kernelgradientparams.initdict = NaN;
% kernelgradientparams.dictsize = 256;
% kernelgradientparams.iternum = 500;
% kernelgradientparams.iternum2 = 10;
% kernelgradientparams.blockratio = 0.03000;
% kernelgradientparams.codinglambda = 0.1500;
% kernelgradientparams.dictclear = 0;
% kernelgradientparams.kerneltype = 'G';
% kernelgradientparams.kernelparam1 = 8;
% kernelgradientparams.kernelparam2 = 1;
% kernelgradientparams.printinfo = 1;
% kernelgradientparams.errorinfo = 0;
% D = kerneldictgradient(trainFeatures, kernelgradientparams);
% save(sprintf('brodatz_experiments/brodatzKernelDictionary%d%d_iter500_norm%d_patch%d_large%d_%s',...
% 	class1, class2, useNorm, patchSize, largeExperiment, slideType),...
% 	'D', 'kernelgradientparams');
