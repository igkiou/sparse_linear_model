params = setParameters;
params.dictionaryNumIters = 100;
params.dictionaryLambda = 0.01;
params.codingLambda = 0.01;
params.dictionaryNoiseModel = 0;
params.dictionarySize = 3000;
load data_split trainData
D = dictionary_learning_special(trainData, params);
