% load ~/MATLAB/sparse_linear_model/mexfiles/dict.mat
% dbmex on
% % [a2 b2] = group_lasso_proximal_mex(A2, 7, indexSets2);
% % A2 = group_lasso_ista_mex(X, D, 50, indexSets2);
% B = matrix_dictionary_learning_lowrank_apg_mex(D, X, A, numRows, mu, [], [], [], 100);

%%
% numRows = 64;
% numColumns = 31;
% numSamples = 30000;
% numAtoms = 3000;

%%
numRows = 5;
numColumns = 2;
numSamples = 30;
numAtoms = 20;

%%
% numRows = 15;
% numColumns = 10;
% numSamples = 300;
% numAtoms = 200;

%%
X = randn(numRows * numColumns, numSamples);
A = randn(numAtoms, numSamples);
B = randn(numRows * numColumns, numAtoms);
Y = randn(numColumns, numColumns);
W = eye(numColumns);
YtWsqY = Y'*Y;
XY = zeros(numRows * numColumns, numSamples);
for iter = 1:numSamples,
	XY(:,iter) = vec(reshape(X(:,iter), [numRows numColumns]) * Y);
end;
XAt = X * A';
XYAt = XY * A';
AAt = A * A';
mu = 0.01;
kappa = 0.01;
numIters = 100;

%%
% B2 = operator_dictionary_learning_lowrank_weighted_apg_old(B,X,A,Y,W,mu,kappa,[],1,10,1);

%%
B3 = operator_dictionary_learning_lowrank_weighted_apg(B,XYAt,AAt,YtWsqY,numRows,numSamples,mu,kappa,[],1,numIters,1);

%%
B4 = operator_dictionary_learning_lowrank_weighted_apg_mex(B,XYAt,AAt,YtWsqY,numRows,numSamples,mu,kappa,[],1,numIters,1);
% B41 = operator_dictionary_learning_lowrank_weighted_apg_mex_parallel(B,XYAt,AAt,YtWsqY,numRows,numSamples,mu,kappa,[],1,10,1);

%%
% B5 = matrix_dictionary_learning_lowrank_apg_old(B,X,A,numRows,mu,kappa,[],1,10,1);

%%
B6 = matrix_dictionary_learning_lowrank_apg(B,XAt,AAt,numRows,numSamples,mu,kappa,[],1,numIters,1);

%%
B7 = matrix_dictionary_learning_lowrank_apg_mex(B,XAt,AAt,numRows,numSamples,mu,kappa,[],1,numIters,1);
% B71 = matrix_dictionary_learning_lowrank_apg_mex_gesdd(B,XAt,AAt,numRows,numSamples,mu,kappa,[],1,100,1);
% B71 = matrix_dictionary_learning_lowrank_apg_mex16(B,XAt,AAt,numRows,numSamples,mu,kappa,[],1,100,1);
% B71 = matrix_dictionary_learning_lowrank_apg_mex_parallel(B,XAt,AAt,numRows,numSamples,mu,kappa,[],1,10,1);
