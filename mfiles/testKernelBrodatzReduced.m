load brodatz_experiments/brodatzDataAllUnnorm.mat
svmparams = setParameters;
svmparams.allVerboseMode = -1;
reducedDim = [2 5 10:10:60 64];
numDims = length(reducedDim);

folder_contents = ls('brodatz_experiments/brodatzKernelDictionaryAll*');
tempNames = regexp(folder_contents, '\s\s*', 'split');
numDicts = length(tempNames);
fid = fopen('brodatz_runs_reduced_kernel.txt', 'wt');
for iterDict = 1:(numDicts - 1),
	load(sprintf('%s', tempNames{iterDict}));
	fprintf(fid, '%s\n', tempNames{iterDict});
	D_kernel = D;
% 	gamma = 1/128;
% 	sigma = sqrt(0.5/gamma);
	if (exist('params', 'var')),
		sigma = params.kernelParam1;
	elseif (exist('kernelgradientparams', 'var')),
		sigma = kernelgradientparams.kernelparam1;
	else
		error('Cannot find dictionary learning parameter file.');
	end;
	accuracy_D_kernel_hinge = zeros(1, numDims);
	accuracy_D_kernel_huber = zeros(1, numDims);
	accuracy_D_kernel_knn = zeros(1, numDims);
	
	disp('gramDD.');
	gramDD = kernel_gram_mex(D_kernel, [], 'g', sigma);
	disp('gramTrainD.');
	gramTrainD = kernel_gram_mex(trainFeatures, D_kernel, 'g', sigma);
	disp('gramTestD.');
	gramTestD = kernel_gram_mex(testFeatures, D_kernel, 'g', sigma);
	Vec = learn_sensing_exact_kernel(D_kernel, reducedDim(end), [], gramDD, 'g', sigma)';
	[fea_Train_Reduced_Large fea_Test_Reduced_Large] = reduceDimensionKernel(Vec, trainFeatures', testFeatures',...
			D_kernel, gramTrainD, gramTestD, 'g', sigma);

	for iterDim = 1:numDims,
		fprintf('Dimension %d our of %d. ', iterDim, numDims);
		fea_Train_Reduced = fea_Train_Reduced_Large(:, 1:reducedDim(iterDim));
		fea_Test_Reduced = fea_Test_Reduced_Large(:, 1:reducedDim(iterDim));
% 		svmparams.svmLossFunction = 'hinge';
% 		fprintf('D kernel hinge. ');
% 		accuracy_D_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
% 		svmparams.svmLossFunction = 'huber';
% 		fprintf('D kernel huber. ');
% 		accuracy_D_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
		fprintf('D kernel KNN. ');
		[foo accuracy_D_kernel_knn(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testLabels', trainLabels');
		fprintf('\n');
	end;
	fprintf(fid, '%g ', accuracy_D_kernel_hinge);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_D_kernel_huber);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_D_kernel_knn);
	fprintf(fid, '\n');
	clear D params params kernelgradientparams
end;
fclose(fid);
