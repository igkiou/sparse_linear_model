% X = randn(64,300);
% D = randn(64,256);
% lambda = 0.01;
% dbmex on
% A = l1qp_featuresign_mex(X,D,lambda);

X = randn(64,300);
%dbmex on
Xn = convex_projection_mex(X,64,300);
