function Vec = trainRLDA(trainFeatures, trainLabels)
%% use RLDA
options = [];
options.Regu = 1;				% use regularization
options.ReguAlpha = 0.01;
options.ReguType = 'Ridge';
Vec = LDA(trainLabels, options, trainFeatures);
