dims = 10:10:100;
numTrainPerClass = 400;
split = 1;
load(sprintf('~/MATLAB/sparse_linear_model/text_experiments/%dTrain/%d', numTrainPerClass, split));


% % reu+tng, reu+tng
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textNumWords = 100;
% [tngData reuData vocabData tngLabels] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('reu+tng, reu+tng, numwords %d\n', params.textNumWords);
% 
%  for iterDim = dims,
% 	fprintf('dim %d, ', iterDim);
% 	d = data(sparse(trainFeatures)', trainLabels');
% 	a = epca({'distr=''poisson''', sprintf('dim=%d', iterDim)});
% 	% lower the learning rates, the parameters are not so stable as
% 	% exp(A*V) is taken during computation
% 	a.etaW = 0.0005;
% 	a.etaH = 0.0005;
% 	a.eps = .005;
% 	a.verbosity = 2;
% 	a.optimizer = 'gd2';
% 	[dout, e] = train(a, d);
% 	dtest = test(e, data(sparse(testFeatures)', testLabels'));
% 	save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/epca/data%s_vocab%s_numwords%d_dim%d_numpc%d_split%d.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, iterDim, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'a', 'e', 'dout', 'dtest');
% end;
% fprintf('\n');
% 
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textNumWords = 150;
% [tngData reuData vocabData] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('reu+tng, reu+tng, numwords %d\n', params.textNumWords);
% 
%  for iterDim = dims, 	
% 	 fprintf('dim %d, ', iterDim);
% 	d = data(sparse(trainFeatures)', trainLabels');
% 	a = epca({'distr=''poisson''', sprintf('dim=%d', iterDim)});
% 	% lower the learning rates, the parameters are not so stable as
% 	% exp(A*V) is taken during computation
% 	a.etaW = 0.0005;
% 	a.etaH = 0.0005;
% 	a.eps = .005;
% 	a.verbosity = 2;
% 	a.optimizer = 'gd2';
% 	[dout, e] = train(a, d);
% 	dtest = test(e, data(sparse(testFeatures)', testLabels'));
% 	save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/epca/data%s_vocab%s_numwords%d_dim%d_numpc%d_split%d.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, iterDim, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'a', 'e', 'dout', 'dtest');
% end;
% fprintf('\n');
% 
% params.textDataSet = 'reu+tng';
% params.textVocabulary = 'reu+tng';
% params.textNumWords = 300;
% [tngData reuData vocabData] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('reu+tng, reu+tng, numwords %d\n', params.textNumWords);
% 
%  for iterDim = dims, 	
% 	fprintf('dim %d, ', iterDim);
% 	d = data(sparse(trainFeatures)', trainLabels');
% 	a = epca({'distr=''poisson''', sprintf('dim=%d', iterDim)});
% 	% lower the learning rates, the parameters are not so stable as
% 	% exp(A*V) is taken during computation
% 	a.etaW = 0.0005;
% 	a.etaH = 0.0005;
% 	a.eps = .005;
% 	a.verbosity = 2;
% 	a.optimizer = 'gd2';
% 	[dout, e] = train(a, d);
% 	dtest = test(e, data(sparse(testFeatures)', testLabels'));
% 	save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/epca/data%s_vocab%s_numwords%d_dim%d_numpc%d_split%d.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, iterDim, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'a', 'e', 'dout', 'dtest');
% end;
% fprintf('\n');
% 
% 
% % tng, reu+tng
% params.textDataSet = 'tng';
% params.textVocabulary = 'reu+tng';
% params.textNumWords = 100;
% [tngData reuData vocabData] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, reu+tng, numwords %d\n', params.textNumWords);
% 
%  for iterDim = dims, 	
% 	fprintf('dim %d, ', iterDim);
% 	d = data(sparse(trainFeatures)', trainLabels');
% 	a = epca({'distr=''poisson''', sprintf('dim=%d', iterDim)});
% 	% lower the learning rates, the parameters are not so stable as
% 	% exp(A*V) is taken during computation
% 	a.etaW = 0.0005;
% 	a.etaH = 0.0005;
% 	a.eps = .005;
% 	a.verbosity = 2;
% 	a.optimizer = 'gd2';
% 	[dout, e] = train(a, d);
% 	dtest = test(e, data(sparse(testFeatures)', testLabels'));
% 	save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/epca/data%s_vocab%s_numwords%d_dim%d_numpc%d_split%d.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, iterDim, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'a', 'e', 'dout', 'dtest');
% end;
% fprintf('\n');
% 
% params.textDataSet = 'tng';
% params.textVocabulary = 'reu+tng';
% params.textNumWords = 150;
% [tngData reuData vocabData] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, reu+tng, numwords %d\n', params.textNumWords);
% 
%  for iterDim = dims, 	
% 	fprintf('dim %d, ', iterDim);
% 	d = data(sparse(trainFeatures)', trainLabels');
% 	a = epca({'distr=''poisson''', sprintf('dim=%d', iterDim)});
% 	% lower the learning rates, the parameters are not so stable as
% 	% exp(A*V) is taken during computation
% 	a.etaW = 0.0005;
% 	a.etaH = 0.0005;
% 	a.eps = .005;
% 	a.verbosity = 2;
% 	a.optimizer = 'gd2';
% 	[dout, e] = train(a, d);
% 	dtest = test(e, data(sparse(testFeatures)', testLabels'));
% 	save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/epca/data%s_vocab%s_numwords%d_dim%d_numpc%d_split%d.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, iterDim, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'a', 'e', 'dout', 'dtest');
% end;
% fprintf('\n');
% 
% params.textDataSet = 'tng';
% params.textVocabulary = 'reu+tng';
% params.textNumWords = 300;
% [tngData reuData vocabData] = getTextData(params);
% trainFeatures = [tngData(:, trainIdx) reuData];
% trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
% testFeatures = tngData(:, testIdx);
% testLabels = tngLabels(testIdx);
% fprintf('tng, reu+tng, numwords %d\n', params.textNumWords);
% 
%  for iterDim = dims, 	
% 	fprintf('dim %d, ', iterDim);
% 	d = data(sparse(trainFeatures)', trainLabels');
% 	a = epca({'distr=''poisson''', sprintf('dim=%d', iterDim)});
% 	% lower the learning rates, the parameters are not so stable as
% 	% exp(A*V) is taken during computation
% 	a.etaW = 0.0005;
% 	a.etaH = 0.0005;
% 	a.eps = .005;
% 	a.verbosity = 2;
% 	a.optimizer = 'gd2';
% 	[dout, e] = train(a, d);
% 	dtest = test(e, data(sparse(testFeatures)', testLabels'));
% 	save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/epca/data%s_vocab%s_numwords%d_dim%d_numpc%d_split%d.mat',...
% 		params.textDataSet, params.textVocabulary, params.textNumWords, iterDim, numTrainPerClass, split),...
% 		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'a', 'e', 'dout', 'dtest');
% end;
% fprintf('\n');


% tng, tng
params.textDataSet = 'tng';
params.textVocabulary = 'tng';
params.textNumWords = 100;
[tngData reuData vocabData] = getTextData(params);
trainFeatures = [tngData(:, trainIdx) reuData];
trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
testFeatures = tngData(:, testIdx);
testLabels = tngLabels(testIdx);
fprintf('tng, tng, numwords %d\n', params.textNumWords);

 for iterDim = dims, 	
	fprintf('dim %d, ', iterDim);
	d = data(sparse(trainFeatures)', trainLabels');
	a = epca({'distr=''poisson''', sprintf('dim=%d', iterDim)});
	% lower the learning rates, the parameters are not so stable as
	% exp(A*V) is taken during computation
	a.etaW = 0.0005;
	a.etaH = 0.0005;
	a.eps = .005;
	a.verbosity = 2;
	a.optimizer = 'gd2';
	[dout, e] = train(a, d);
	dtest = test(e, data(sparse(testFeatures)', testLabels'));
	save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/epca/data%s_vocab%s_numwords%d_dim%d_numpc%d_split%d.mat',...
		params.textDataSet, params.textVocabulary, params.textNumWords, iterDim, numTrainPerClass, split),...
		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'a', 'e', 'dout', 'dtest');
end;
fprintf('\n');

params.textDataSet = 'tng';
params.textVocabulary = 'tng';
params.textNumWords = 150;
[tngData reuData vocabData] = getTextData(params);
trainFeatures = [tngData(:, trainIdx) reuData];
trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
testFeatures = tngData(:, testIdx);
testLabels = tngLabels(testIdx);
fprintf('tng, tng, numwords %d\n', params.textNumWords);

 for iterDim = dims, 	
	fprintf('dim %d, ', iterDim);
	d = data(sparse(trainFeatures)', trainLabels');
	a = epca({'distr=''poisson''', sprintf('dim=%d', iterDim)});
	% lower the learning rates, the parameters are not so stable as
	% exp(A*V) is taken during computation
	a.etaW = 0.0005;
	a.etaH = 0.0005;
	a.eps = .005;
	a.verbosity = 2;
	a.optimizer = 'gd2';
	[dout, e] = train(a, d);
	dtest = test(e, data(sparse(testFeatures)', testLabels'));
	save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/epca/data%s_vocab%s_numwords%d_dim%d_numpc%d_split%d.mat',...
		params.textDataSet, params.textVocabulary, params.textNumWords, iterDim, numTrainPerClass, split),...
		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'a', 'e', 'dout', 'dtest');
end;
fprintf('\n');

params.textDataSet = 'tng';
params.textVocabulary = 'tng';
params.textNumWords = 300;
[tngData reuData vocabData] = getTextData(params);
trainFeatures = [tngData(:, trainIdx) reuData];
trainLabels = [tngLabels(trainIdx) zeros(1, size(reuData, 2))];
testFeatures = tngData(:, testIdx);
testLabels = tngLabels(testIdx);
fprintf('tng, tng, numwords %d\n', params.textNumWords);

 for iterDim = dims, 	
	fprintf('dim %d, ', iterDim);
	d = data(sparse(trainFeatures)', trainLabels');
	a = epca({'distr=''poisson''', sprintf('dim=%d', iterDim)});
	% lower the learning rates, the parameters are not so stable as
	% exp(A*V) is taken during computation
	a.etaW = 0.0005;
	a.etaH = 0.0005;
	a.eps = .005;
	a.verbosity = 2;
	a.optimizer = 'gd2';
	[dout, e] = train(a, d);
	dtest = test(e, data(sparse(testFeatures)', testLabels'));
	save(sprintf('~/MATLAB/sparse_linear_model/text_experiments/epca/data%s_vocab%s_numwords%d_dim%d_numpc%d_split%d.mat',...
		params.textDataSet, params.textVocabulary, params.textNumWords, iterDim, numTrainPerClass, split),...
		'params', 'trainFeatures', 'trainLabels', 'testFeatures', 'testLabels', 'a', 'e', 'dout', 'dtest');
end;
fprintf('\n');
