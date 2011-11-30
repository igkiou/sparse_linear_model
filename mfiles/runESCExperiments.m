dims = 10:10:100;
numTrainPerClass = 400;
split = 1;
load(sprintf('~/MATLAB/sparse_linear_model/text_experiments/%dTrain/%d', numTrainPerClass, split));
dictParams = setParameters;
dictParams.dictionaryNumIters = 100;
params = setParameters;

% % reu+tng, reu+tng
%% DONE
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'mutinfo';
% params.textNumWords = 100;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('reu+tng, reu+tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% DONE
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'mutinfo';
% params.textNumWords = 150;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('reu+tng, reu+tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% DONE
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'mutinfo';
% params.textNumWords = 300;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('reu+tng, reu+tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');


% % tng, reu+tng
% params.textDataSet = 'tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'mutinfo';
% params.textNumWords = 100;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, reu+tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');
% 
% params.textDataSet = 'tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'mutinfo';
% params.textNumWords = 150;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, reu+tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

% params.textDataSet = 'tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'mutinfo';
% params.textNumWords = 300;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, reu+tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

% % tng, tng
%% DONE
% params.textDataSet = 'tng';
% params.textVocabulary = 'tng';
% params.textSelectionMethod = 'mutinfo';
% params.textNumWords = 100;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% DONE
% params.textDataSet = 'tng';
% params.textVocabulary = 'tng';
% params.textSelectionMethod = 'mutinfo';
% params.textNumWords = 150;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% DONE
% params.textDataSet = 'tng';
% params.textVocabulary = 'tng';
% params.textSelectionMethod = 'mutinfo';
% params.textNumWords = 300;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

% % tng, tng
%% DONE
% params.textDataSet = 'tng';
% params.textVocabulary = 'tng';
% params.textSelectionMethod = 'counts';
% params.textNumWords = 100;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% DONE
% params.textDataSet = 'tng';
% params.textVocabulary = 'tng';
% params.textSelectionMethod = 'counts';
% params.textNumWords = 150;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% NOT DONE
% params.textDataSet = 'tng';
% params.textVocabulary = 'tng';
% params.textSelectionMethod = 'counts';
% params.textNumWords = 300;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

% % tng, tng
%% DONE
% params.textDataSet = 'tng';
% params.textVocabulary = 'tng';
% params.textSelectionMethod = 'occurrences';
% params.textNumWords = 100;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% DONE
% params.textDataSet = 'tng';
% params.textVocabulary = 'tng';
% params.textSelectionMethod = 'occurrences';
% params.textNumWords = 150;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');


%%

%% DONE
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'counts';
% params.textNumWords = 100;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% DONE
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'counts';
% params.textNumWords = 150;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% NOT DONE
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'counts';
% params.textNumWords = 300;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

% % tng, tng
%% NOT DONE
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'occurrences';
% params.textNumWords = 100;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% NOT DONE
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'occurrences';
% params.textNumWords = 150;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');

%% NOT DONE
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textSelectionMethod = 'counts';
% params.textNumWords = 300;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, tng, numwords %d\n', params.textNumWords);
% 
% dictParams.dictionaryBlockRatio = 2000 / size(trainFeatures, 2);
% D = dictionary_learning(trainFeatures, [], [], dictParams);
% save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_numwords%d_numpc%d_split%d_stochastic.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');


%% rcv1

%% NOT DONE
params.textDataSet = 'rcv1+tng';
params.textVocabulary = 'rcv1+tng';
params.textSelectionMethod = 'occurrences';
params.textNumWords = 100;
[tngData reuData vocabData tngLabels] = getTextData(params);
trainFeatures = [tngData(:, trainIdx) reuData];
trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
testFeatures = tngData(:, testIdx);
testLabels = tngLabels(testIdx);
fprintf('tng, tng, numwords %d\n', params.textNumWords);

trainFeaturesRCV1 = trainFeatures(:, 8001:end);
dictParams.dictionaryBlockRatio = 2000 / size(trainFeaturesRCV1, 2);
dictParams.dictionaryNumIters = 300;
dictParams.dictionaryClearing = 0;
D = dictionary_learning(trainFeaturesRCV1, [], [], dictParams);
save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/esc/data%s_vocab%s_select%s_numwords%d_numpc%d_split%d_stochastic.mat',...
		params.textDataSet, params.textVocabulary, params.textSelectionMethod, params.textNumWords, numTrainPerClass, split),...
		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'D', 'dictParams');
