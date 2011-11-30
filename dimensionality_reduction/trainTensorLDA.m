function Vec = trainTensorLDA(trainFeatures, trainLabels)
%% use TensorLDA
options = [];
options.Regu = 1;				% use regularization
options.Reg = 1;
options.ReguAlpha = 0.01;
options.ReguType = 'Custom';	% use custom regularization
load('TensorR_32x32.mat');		% load tensor regularizer
options.regularizerR = regularizerR;	% set tensor regularizer
Vec = LDA(trainLabels, options, trainFeatures);
