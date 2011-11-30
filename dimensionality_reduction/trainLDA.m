function Vec = trainLDA(trainFeatures, trainLabels)
%% use LDA
options = [];
options.PCARatio = 1;			% how much of PCA to use
options.Regu = 0;				% do not use regularization
Vec = LDA(trainLabels, options, trainFeatures);
