classPairs = nchoose2(1:20);
numPairs = size(classPairs, 1);
permPairs = randperm(numPairs);
svmparams.allVerboseMode = - 1;
[foo trainDual] = link_func_dual(trainFeatures, 'P');
[foo testDual] = link_func_dual(testFeatures, 'P');
trainDual(isinf(trainDual(:))) = -eps;
testDual(isinf(testDual(:))) = -eps;

Phi50 = learn_sensing_exact(D, 50);
Phi100 = learn_sensing_exact(D, 100);

load text_experiments/epca/datatng_vocabtng_selectoccurrences_numwords100_dim100_numpc400_split1
trainPCA100 = get_x(dout)';
testPCA100 = get_x(dtest)';

load text_experiments/epca/datatng_vocabtng_selectoccurrences_numwords100_dim50_numpc400_split1
trainPCA50 = get_x(dout)';
testPCA50 = get_x(dtest)';

a1 = 0;
a2 = 0;
a3 = 0;
a4 = 0;
a5 = 0;
a6 = 0;
a7 = 0;
for iter = 1:100
	disp(sprintf('Now running iter %d.', iter));
	trainClass1 = trainLabels == classPairs(permPairs(iter), 1);
	testClass1 = testLabels == classPairs(permPairs(iter), 1);
	trainClass2 = trainLabels == classPairs(permPairs(iter), 2);
	testClass2 = testLabels == classPairs(permPairs(iter), 2);
	
	trainFeatures1 = trainFeatures(:, trainClass1);
	testFeatures1 = testFeatures(:, testClass1);
	trainFeatures2 = trainFeatures(:, trainClass2);
	testFeatures2 = testFeatures(:, testClass2);
	
	trainDual1 = trainDual(:, trainClass1);
	testDual1 = testDual(:, testClass1);
	trainDual2 = trainDual(:, trainClass2);
	testDual2 = testDual(:, testClass2);
	
	trainPCA1001 = trainPCA100(:, trainClass1);
	testPCA1001 = testPCA100(:, testClass1);
	trainPCA1002 = trainPCA100(:, trainClass2);
	testPCA1002 = testPCA100(:, testClass2);
	
	trainPCA501 = trainPCA50(:, trainClass1);
	testPCA501 = testPCA50(:, testClass1);
	trainPCA502 = trainPCA50(:, trainClass2);
	testPCA502 = testPCA50(:, testClass2);
	
	trainLabels1 = trainLabels(:, trainClass1);
	testLabels1 = testLabels(:, testClass1);
	trainLabels2 = trainLabels(:, trainClass2);
	testLabels2 = testLabels(:, testClass2);
	
	trainA1 = trainA(:, trainClass1);
	testA1 = testA(:, testClass1);
	trainA2 = trainA(:, trainClass2);
	testA2 = testA(:, testClass2);
	
	a1 = a1 + run_svm([trainFeatures1 trainFeatures2], [trainLabels1 trainLabels2],...
				[testFeatures1 testFeatures2], [testLabels1 testLabels2],...
				[], [], svmparams);
	
	a2 = a2 + run_svm([trainA1 trainA2], [trainLabels1 trainLabels2],...
				[testA1 testA2], [testLabels1 testLabels2],...
				[], [], svmparams);
	
	a3 = a3 + run_svm([trainDual1 trainDual2], [trainLabels1 trainLabels2],...
				[testDual1 testDual2], [testLabels1 testLabels2],...
				[], [], svmparams);
	
	a4 = a4 + run_svm(Phi50 * [trainDual1 trainDual2], [trainLabels1 trainLabels2],...
				Phi50 * [testDual1 testDual2], [testLabels1 testLabels2],...
				[], [], svmparams);
	
	a5 = a5 + run_svm(Phi100 * [trainDual1 trainDual2], [trainLabels1 trainLabels2],...
				Phi100 * [testDual1 testDual2], [testLabels1 testLabels2],...
				[], [], svmparams);
	
	a6 = a6 + run_svm([trainPCA1001 trainPCA1002], [trainLabels1 trainLabels2],...
				[testPCA1001 testPCA1002], [testLabels1 testLabels2],...
				[], [], svmparams);
	
	a7 = a7 + run_svm([trainPCA501 trainPCA502], [trainLabels1 trainLabels2],...
				[testPCA501 testPCA502], [testLabels1 testLabels2],...
				[], [], svmparams);
end;
a1 = a1 / 100;
a2 = a2 / 100;
