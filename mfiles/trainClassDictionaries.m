params = setParameters;
params.dictionarySize = 16;
% disp('Running dimension 10');
% params.compressNumSamples = 10;
% [D Phi] = dictionary_learning(fea_Train', [], [], params);
% save dictionaryPhi10origNormExact D Phi params
% disp('Running dimension 20');
% params.compressNumSamples = 20;
% [D Phi] = dictionary_learning(fea_Train', [], [], params);
% save dictionaryPhi20origNormExact D Phi params
% disp('Running dimension 30');
% params.compressNumSamples = 30;
% [D Phi] = dictionary_learning(fea_Train', [], [], params);
% save dictionaryPhi30origNormExact D Phi params
% disp('Running dimension 40');
% params.compressNumSamples = 40;
% [D Phi] = dictionary_learning(fea_Train', [], [], params);
% save dictionaryPhi40origNormExact D Phi params
% disp('Running dimension 50');
% params.compressNumSamples = 50;
% [D Phi] = dictionary_learning(fea_Train', [], [], params);
% save dictionaryPhi50origNormExact D Phi params
% disp('Running dimension 60');
% params.compressNumSamples = 60;
% [D Phi] = dictionary_learning(fea_Train', [], [], params);
% save dictionaryPhi60origNormExact D Phi params
% disp('Running dimension 70');
% params.compressNumSamples = 70;
% [D Phi] = dictionary_learning(fea_Train', [], [], params);
% save dictionaryPhi70origNormExact D Phi params
% disp('Running dimension 80');
% params.compressNumSamples = 80;
% [D Phi] = dictionary_learning(fea_Train', [], [], params);
% save dictionaryPhi80origNormExact D Phi params
% disp('Running dimension 90');
% params.compressNumSamples = 90;
% [D Phi] = dictionary_learning(fea_Train', [], [], params);
% save dictionaryPhi90origNormExact D Phi params
% params.dictionaryMethod = 'gradient';
% classes = unique(gnd_Train);
% numClasses = length(classes);
% D = zeros(size(fea_Train, 2), numClasses * params.dictionarySize);
% for iterClass = 1:numClasses,
% 	D(:, ((iterClass - 1) * params.dictionarySize + 1):(iterClass * params.dictionarySize)) =...
% 		dictionary_learning(fea_Train(gnd_Train == classes(iterClass), :)', [], [], params);
% end;	
blockInds = 1:params.dictionarySize:(numClasses * params.dictionarySize + 1);
Phi = learn_sensing_block_exact(D, blockInds, 100, 0.8);

% dimensions = 10:10:100;
% accuraciesDict = zeros(size(dimensions));
% for iter = 1:length(dimensions),
% 	disp(sprintf('Running dimension %d', dimensions(iter)));
% 	eval(sprintf('load ./dictionaryPhi%dorigNormExact', dimensions(iter)));
% 	Phi = learn_sensing_exact(D, dimensions(iter));
% 	Vec = Phi';
% 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test);
% 	[results accuraciesDict(iter)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 4, gnd_Test, gnd_Train);
% end;
% 
% accuraciesDict_1NN = zeros(size(dimensions));
% for iter = 1:length(dimensions),
% 	disp(sprintf('Running dimension %d', dimensions(iter)));
% 	eval(sprintf('load ./dictionaryPhi%dorigNormExact', dimensions(iter)));
% 	Phi = learn_sensing_exact(D, dimensions(iter));
% 	Vec = Phi';
% 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test);
% 	[results accuraciesDict_1NN(iter)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
% end;
