%% parameter setting

% sift descriptor extraction
params.className = 'person';
params.numPics = 300;
params.gridSpacing = 4;
params.patchSize = 32;
params.maxImSize = 640;
params.nrml_threshold = 1;                 % low contrast region normalization threshold (descriptor length)

database = calculateSiftFeatures(params.className, params.numPics, params.gridSpacing,...
					params.patchSize, params.maxImSize, params.nrml_threshold);
rt_data_dir = '/home/igkiou/MATLAB/sparse_linear_model/results/Graz_experiments/sift_features/';
save(sprintf('%s%s_database.mat', rt_data_dir, params.className), 'database', 'params');
