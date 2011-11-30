params = setParameters;
params.dictionarySize = 1024;
disp('Running dimension 10');
params.compressNumSamples = 10;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi10origNormExact D Phi params
disp('Running dimension 20');
params.compressNumSamples = 20;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi20origNormExact D Phi params
disp('Running dimension 30');
params.compressNumSamples = 30;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi30origNormExact D Phi params
disp('Running dimension 40');
params.compressNumSamples = 40;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi40origNormExact D Phi params
disp('Running dimension 50');
params.compressNumSamples = 50;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi50origNormExact D Phi params
disp('Running dimension 60');
params.compressNumSamples = 60;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi60origNormExact D Phi params
disp('Running dimension 70');
params.compressNumSamples = 70;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi70origNormExact D Phi params
disp('Running dimension 80');
params.compressNumSamples = 80;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi80origNormExact D Phi params
disp('Running dimension 90');
params.compressNumSamples = 90;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi90origNormExact D Phi params
disp('Running dimension 100');
params.compressNumSamples = 100;
[D Phi] = dictionary_learning(fea_Train', [], [], params);
save dictionaryPhi100origNormExact D Phi params

% dimensions = 10:10:100;
% accuraciesDict = zeros(size(dimensions));
% for iter = 1:length(dimensions),
% 	disp(sprintf('Running dimension %d', dimensions(iter)));
% 	Phi = learn_sensing_exact(D, dimensions(iter));
% 	Vec = Phi';
% 	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test);
% 	[results accuraciesDict(iter)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 4, gnd_Test, gnd_Train);
% end;

accuraciesDict = zeros(size(dimensions));
for iter = 1:length(dimensions),
	disp(sprintf('Running dimension %d', dimensions(iter)));
% 	Phi = learn_sensing_exact(D, dimensions(iter));
% 	Vec = Phi';
	Vec = trainEigenFaces(fea_Train, gnd_Train, dimensions(iter));
	[fea_Train_Reduced, fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test);
	[results accuraciesDict_1NN(iter)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 4, gnd_Test, gnd_Train);
end;
