function Vec = trainSRLDA(trainFeatures, trainLabels)
%% use SR-LDA

options = [];
options.gnd = trainLabels;
options.ReguAlpha = 0.01;
options.ReguType = 'Ridge';
Vec = SR_caller(options, trainFeatures);
