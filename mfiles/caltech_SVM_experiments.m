% experiments with "manual" ova libsvm
% classes = unique(train_labels_numeric);
% numClasses = length(classes);
% for iterClass = 1:numClasses,
% tmpY = double(train_labels_numeric == classes(iterClass));
% tmpY(tmpY == 0) = -1;
% % tmpYFirst(iterClass) = tmpY(1);
% model(iterClass) = libsvm_train(tmpY, [(1:1530)', kernel_traintrain], '-t 4 -c 10000  -e 0.0000001 -p 0.00001 -s 0');
% end;
% 
% out = zeros(1530, numClasses);
% for iterClass = 1:numClasses,
% a = zeros(size(kernel_traintrain, 1), 1);
% a(model(iterClass).SVs) = model(iterClass).sv_coef;
% b = model(iterClass).rho;
% % a = a * tmpYFirst(iterClass);
% % b = b * tmpYFirst(iterClass);
% out(:, iterClass) = kernel_traintest' * a + b;
% end;
% [foo, Ypred_test] = max(out, [], 2);
% sum(Ypred_test==test_labels_numeric) / numel(test_labels_numeric) * 100

% [train_labels_numeric test_labels_numeric label_legend] = extractCaltechLabels(train_labels, test_labels);
% % trkernel = trace(kernel_traintrain); % do diagonal normalization, supposedly used by Pinto
% % kernel_traintrain_proc = kernel_traintrain / trkernel; 
% % kernel_traintest_proc = kernel_traintest / trkernel;
% kernel_traintrain_proc = kernel_traintrain;
% kernel_traintest_proc = kernel_traintest;
% [Y class_name] = oneofc(train_labels_numeric); %#ok
% [a b] = libsvm(train_labels_numeric, kernel_traintrain_proc, 1000);
% C = li2nsvm_multiclass_fwd(kernel_traintest_proc, a, b, class_name);
% acc = sum(C==test_labels_numeric')/numel(C)*100;

%%
% combination = [318:321,322:325,326:329,666:669,670:673,674:677,66:68,689:691,681,680,679,678,692,69,630,632,688];
% K_train_train = zeros(3060);
% K_test_train = zeros([2995 3060]);
% for iter = 1:length(combination),
% 	load(sprintf('/home/igkiou/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_feat%d', combination(iter)));
% 	K_train_train = K_train_train + K / length(combination);
% 	load(sprintf('/home/igkiou/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_test_feat%d', combination(iter)));
% 	K_test_train = K_test_train + K / length(combination);
% end;
% load /home/igkiou/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_labels
% K_train_train = double(K_train_train);
% K_test_train = double(K_test_train);

%%
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_feataverage39.mat
K_train_train = K;
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_test_feataverage39.mat
K_test_train = K;
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_labels
K_train_train = double(K_train_train);
K_test_train = double(K_test_train);
KXX = K_train_train;

%%
% [a b] = libsvm(tr_label, K_train_train, 1000);
% [Y class_name] = oneofc(tr_label);
% C = li2nsvm_multiclass_fwd(K_test_train', a, b, class_name);
% acc = sum(C==te_label')/numel(C)*100;
