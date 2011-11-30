function [Vec options] = trainKernelSRLDA(trainFeatures, trainLabels)
%% use SR-LDA

options = [];
options.gnd = trainLabels;
options.ReguAlpha = 0.01;
options.KernelType = 'Gaussian';
options.t = 5;
options.ReguType = 'Ridge';
Vec = KSR_caller(options, trainFeatures);
