X = [];
Y = [];
load /home/igkiou/MATLAB/sparse_linear_model/results/Graz_experiments/sift_features/cars_database
for iter = 1:2:300,
load(database.path{iter})
X = [X feaSet.feaArr(:, feaSet.feaLabels == 0)];
Y = [Y feaSet.feaArr(:, feaSet.feaLabels == 1)];
disp(sprintf('Done iter %d', iter));
end;
trainSIFTFeaturesForegroundAll = X;
trainSIFTFeaturesBackgroundAll = Y;
clear X Y
trainBackgroundInds = randperm(size(trainSIFTFeaturesBackgroundAll, 2));
trainForegroundInds = randperm(size(trainSIFTFeaturesForegroundAll, 2));
trainSIFTFeaturesForegroundSample = trainSIFTFeaturesForegroundAll(:, trainForegroundInds(1:150000));
trainSIFTFeaturesBackgroundSample = trainSIFTFeaturesBackgroundAll(:, trainBackgroundInds(1:150000));
trainSIFTFeatures = [trainSIFTFeaturesBackgroundSample trainSIFTFeaturesForegroundSample];
trainLabels = [ones(1, 150000) zeros(1, 150000)];
save /home/igkiou/MATLAB/sparse_linear_model/results/Graz_experiments/sift_features/carsTrainSIFTFeatures trainSIFTFeaturesBackgroundAll trainSIFTFeaturesForegroundAll trainSIFTFeaturesBackgroundSample trainSIFTFeaturesForegroundSample trainBackgroundInds trainForegroundInds trainSIFTFeatures trainLabels -v7.3

clear all
X = [];
Y = [];
load /home/igkiou/MATLAB/sparse_linear_model/results/Graz_experiments/sift_features/cars_database
for iter = 2:2:300,
load(database.path{iter})
X = [X feaSet.feaArr(:, feaSet.feaLabels == 0)];
Y = [Y feaSet.feaArr(:, feaSet.feaLabels == 1)];
disp(sprintf('Done iter %d', iter));
end;
testSIFTFeaturesForegroundAll = X;
testSIFTFeaturesBackgroundAll = Y;
clear X Y
testBackgroundInds = randperm(size(testSIFTFeaturesBackgroundAll, 2));
testForegroundInds = randperm(size(testSIFTFeaturesForegroundAll, 2));
testSIFTFeaturesForegroundSample = testSIFTFeaturesForegroundAll(:, testForegroundInds(1:150000));
testSIFTFeaturesBackgroundSample = testSIFTFeaturesBackgroundAll(:, testBackgroundInds(1:150000));
testSIFTFeatures = [testSIFTFeaturesBackgroundSample testSIFTFeaturesForegroundSample];
testLabels = [ones(1, 150000) zeros(1, 150000)];
save /home/igkiou/MATLAB/sparse_linear_model/results/Graz_experiments/sift_features/carsTestSIFTFeatures testSIFTFeaturesBackgroundAll testSIFTFeaturesForegroundAll testSIFTFeaturesBackgroundSample testSIFTFeaturesForegroundSample testBackgroundInds testForegroundInds testSIFTFeatures testLabels -v7.3
