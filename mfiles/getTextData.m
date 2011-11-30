function [tngData reuData vocabData tngLabels] = getTextData(params)

dataSet = params.textDataSet;
vocabulary = params.textVocabulary;
selectionMethod = params.textSelectionMethod;
numWords = params.textNumWords;

if (~strcmp(dataSet, 'tng') && ~strcmp(dataSet, 'reu') && ...
		~strcmp(dataSet, 'tng+reu') && ~strcmp(dataSet, 'reu+tng') && ...
		~strcmp(dataSet, 'tng+rcv1') && ~strcmp(dataSet, 'rcv1+tng')),
	error('Unspecified dataset.'); %#ok
end;

if (~strcmp(vocabulary, 'tng') && ~strcmp(vocabulary, 'reu+tng') && ...
		~strcmp(vocabulary, 'rcv1+tng')),
	error('Unspecified vocabulary.'); %#ok
end;

if (~strcmp(selectionMethod, 'mutinfo') && ~strcmp(selectionMethod, 'counts') && ...
		~strcmp(selectionMethod, 'occurrences') && ~strcmp(selectionMethod, 'random')),
	error('Unspecified vocabulary selection method.'); %#ok
end;

if (((strcmp(dataSet, 'reu') || strcmp(dataSet, 'reu+tng') || strcmp(dataSet, 'tng+reu')) && ...
		~strcmp(vocabulary, 'reu+tnh')) || ...
	((strcmp(dataSet, 'rcv1') || strcmp(dataSet, 'rcv1+tng') || strcmp(dataSet, 'tng+rcv1')) && ...
		~strcmp(vocabulary, 'rcv1+tng'))),		
	error('Dataset %s cannot be used with vocabulary %s.', dataSet, vocabulary); %#ok
end;

if (strcmp(vocabulary, 'tng')),
	load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_dict.mat
	if (strcmp(selectionMethod, 'mutinfo')),
		tngInds = 1:numWords;
	elseif (strcmp(selectionMethod, 'counts')),
		load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_counts.mat
		sums = sum(counts, 2); %#ok
		[sortedCounts sortedInds] = sort(sums, 1, 'descend'); %#ok
		clear counts
		tngInds = sortedInds(1:numWords);
	elseif (strcmp(selectionMethod, 'occurrences')),
		load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_counts.mat
		sums = sum(sign(counts), 2); %#ok
		[sortedCounts sortedInds] = sort(sums, 1, 'descend'); %#ok
		clear counts
		tngInds = sortedInds(1:numWords);
	elseif (strcmp(selectionMethod, 'random')),
		tngInds = randperm(10000);
		tngInds = tngInds(1:numWords);
	end;
	vocabData = word(tngInds);
elseif (strcmp(vocabulary, 'reu+tng')),
	load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_dict.mat
	tngVocab = word;
	load ~/MATLAB/datasets_all/Reuters-21578/ModApte_dict.mat
	reuVocab = word;
	if (strcmp(selectionMethod, 'mutinfo')),
		[commonWords tngWordInds reuWordInds] = intersect(tngVocab(1:701), reuVocab); %#ok
		[foo sortedTngWordInds] = sort(tngWordInds); %#ok
		tngInds = tngWordInds(sortedTngWordInds(1:numWords));
		reuInds = reuWordInds(sortedTngWordInds(1:numWords));
		vocabData = tngVocab(tngInds(1:numWords));
	elseif (strcmp(selectionMethod, 'counts')),
		[commonWords tngWordInds reuWordInds] = intersect(tngVocab, reuVocab); %#ok
		load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_counts.mat
		tngCounts = counts; %#ok
		clear counts
		load ~/MATLAB/datasets_all/Reuters-21578/ModApte_counts.mat
		load ~/MATLAB/datasets_all/Reuters-21578/ModApte_test_counts.mat
		reuCounts = [counts counts_test]; %#ok
		clear counts
		clear coutns_test
		sums = sum(tngCounts(tngWordInds, :), 2) + sum(reuCounts(reuWordInds, :), 2);
		[sortedCounts sortedInds] = sort(sums, 1, 'descend'); %#ok
		tngInds = tngWordInds(sortedInds(1:numWords));
		reuInds = reuWordInds(sortedInds(1:numWords));
		vocabData = tngVocab(tngInds(1:numWords));
	elseif (strcmp(selectionMethod, 'occurrences')),
		[commonWords tngWordInds reuWordInds] = intersect(tngVocab, reuVocab); %#ok
		load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_counts.mat
		tngCounts = counts; %#ok
		clear counts
		load ~/MATLAB/datasets_all/Reuters-21578/ModApte_counts.mat
		load ~/MATLAB/datasets_all/Reuters-21578/ModApte_test_counts.mat
		reuCounts = [counts counts_test]; %#ok
		clear counts
		clear coutns_test
		sums = sum(sign(tngCounts(tngWordInds, :)), 2) + sum(sign(reuCounts(reuWordInds, :)), 2);
		[sortedCounts sortedInds] = sort(sums, 1, 'descend'); %#ok
		tngInds = tngWordInds(sortedInds(1:numWords));
		reuInds = reuWordInds(sortedInds(1:numWords));
		vocabData = tngVocab(tngInds(1:numWords));
	end;
elseif (strcmp(vocabulary, 'rcv1+tng')),
	load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_dict.mat
	tngVocab = word;
	load ~/MATLAB/datasets_all/rcv1_v2/rcv1_v2_dictionary.mat
	rcv1Vocab = word;
	if (strcmp(selectionMethod, 'mutinfo')),
		[commonWords tngWordInds rcv1WordInds] = intersect(tngVocab(1:701), rcv1Vocab); %#ok
		[foo sortedTngWordInds] = sort(tngWordInds); %#ok
		tngInds = tngWordInds(sortedTngWordInds(1:numWords));
		rcv1Inds = rcv1WordInds(sortedTngWordInds(1:numWords));
		vocabData = tngVocab(tngInds(1:numWords));
	elseif (strcmp(selectionMethod, 'counts')),
		[commonWords tngWordInds rcv1WordInds] = intersect(tngVocab, rcv1Vocab); %#ok
		load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_counts.mat
		tngCounts = counts; %#ok
		clear counts
		load ~/MATLAB/datasets_all/rcv1_v2/rcv1_v2_random_counts
		rcv1Counts = counts;
		clear counts
		sums = sum(tngCounts(tngWordInds, :), 2) + sum(rcv1Counts(rcv1WordInds, :), 2);
		[sortedCounts sortedInds] = sort(sums, 1, 'descend'); %#ok
		tngInds = tngWordInds(sortedInds(1:numWords));
		rcv1Inds = rcv1WordInds(sortedInds(1:numWords));
		vocabData = tngVocab(tngInds(1:numWords));
	elseif (strcmp(selectionMethod, 'occurrences')),
		[commonWords tngWordInds rcv1WordInds] = intersect(tngVocab, rcv1Vocab); %#ok
		load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_counts.mat
		tngCounts = counts; %#ok
		clear counts
		load ~/MATLAB/datasets_all/rcv1_v2/rcv1_v2_random_counts
		rcv1Counts = counts;
		clear counts
		sums = sum(sign(tngCounts(tngWordInds, :)), 2) + sum(sign(rcv1Counts(rcv1WordInds, :)), 2);
		[sortedCounts sortedInds] = sort(sums, 1, 'descend'); %#ok
		tngInds = tngWordInds(sortedInds(1:numWords));
		rcv1Inds = rcv1WordInds(sortedInds(1:numWords));
		vocabData = tngVocab(tngInds(1:numWords));
	end;
end;

if (strcmp(dataSet, 'tng')),
	load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_counts.mat
	tngData = counts(tngInds, :); 
	tngLabels = labels_ll;
	reuData = [];
elseif (strcmp(dataSet, 'tng+reu') || strcmp(dataSet, 'reu+tng')),
	load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_counts.mat
	tngData = counts(tngInds, :);
	tngLabels = labels_ll;
	
	load ~/MATLAB/datasets_all/Reuters-21578/ModApte_counts.mat
	load ~/MATLAB/datasets_all/Reuters-21578/ModApte_test_counts.mat
	reuData = [counts(reuInds, :) counts_test(reuInds, :)];
elseif (strcmp(dataSet, 'tng+rcv1') || strcmp(dataSet, 'rcv1+tng')),
	load ~/MATLAB/datasets_all/20_newsgroups/20news_w10000_counts.mat
	tngData = counts(tngInds, :);
	tngLabels = labels_ll;
	
	load ~/MATLAB/datasets_all/rcv1_v2/rcv1_v2_random_counts
	reuData = counts(rcv1Inds, :);
elseif (strcmp(dataSet, 'reu')),
	tngData = [];
	tngLabels = [];
	
	load ~/MATLAB/datasets_all/Reuters-21578/ModApte_counts.mat
	load ~/MATLAB/datasets_all/Reuters-21578/ModApte_test_counts.mat
	reuData = [counts(reuInds, :) counts_test(reuInds, :)]; 
elseif (strcmp(dataSet, 'rcv1')),
	tngData = [];
	tngLabels = [];
	
	load ~/MATLAB/datasets_all/rcv1_v2/rcv1_v2_random_counts
	reuData = counts(rcv1Inds, :);
end;
tngData = full(tngData);
reuData = full(reuData);
