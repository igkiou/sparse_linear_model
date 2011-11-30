% slideType = 'distinct';
% largeExperiment = 1;
% useNorm = 0;
% patchSize = 12;
% 
% system(sprintf('scp igkiou@140.247.62.217:MATLAB/sparse_linear_model/brodatz_experiments/brodatz_All_norm%d_patch%d_large%d_%s.mat brodatz_experiments/',...
% 	useNorm, patchSize, largeExperiment, slideType));
% load(sprintf('brodatz_experiments/brodatz_All_norm%d_patch%d_large%d_%s',...
% 	useNorm, patchSize, largeExperiment, slideType),...
% 	'trainFeatures', 'testFeatures', 'trainLabels', 'testLabels');
% 
% system(sprintf('scp igkiou@140.247.62.217:MATLAB/sparse_linear_model/brodatz_experiments/brodatzKernelDictionaryAll_iter500_norm%d_patch%d_large%d_%s.mat brodatz_experiments/',...
% 	useNorm, patchSize, largeExperiment, slideType));
% load(sprintf('brodatz_experiments/brodatzKernelDictionaryAll_iter500_norm%d_patch%d_large%d_%s',...
% 	useNorm, patchSize, largeExperiment, slideType), 'D');

load ~/MATLAB/datasets_all/MNIST/MNIST_28x28.mat
trainFeatures = imnorm(fea_Train', [-1 1]);
clear fea_Train fea_Test gnd_Train gnd_Test
gamma = 0.00728932024638;

kernelgradientparams.initdict = [];
kernelgradientparams.dictsize = 2048;
kernelgradientparams.iternum = 300;
kernelgradientparams.iternum2 = 10;
kernelgradientparams.blockratio = 3000 / size(trainFeatures, 2);
kernelgradientparams.codinglambda = 0.100;
kernelgradientparams.dictclear = 0;
kernelgradientparams.kerneltype = 'G';
kernelgradientparams.kernelparam1 = sqrt(0.5 / gamma);
kernelgradientparams.kernelparam2 = 1;
kernelgradientparams.printinfo = 1;
kernelgradientparams.errorinfo = 0;
kernelgradientparams.savepath = [];
D = kerneldictgradient(trainFeatures, kernelgradientparams);

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
