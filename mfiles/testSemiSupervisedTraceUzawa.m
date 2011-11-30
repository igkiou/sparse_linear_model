load cifar_32x32_SIFT
clear fea_Train fea_Test
load ~/MATLAB/datasets_all/cifar-10-batches-mat/cifar_32x32.mat gnd_Train gnd_Test          
clear fea_Test_SIFT fea_Train_SIFT
X = fea_Train_SIFT_Norm';
y = gnd_Train';
load tiny_experiments/tinySIFTGaussianDictionaryLarge_custom
Xt = fea_Test_SIFT_Norm';
yt = gnd_Test';
[l u j1 j2] = getDistanceExtremes(X, 5, 95, [], 6800);
[C deltaX i1 i2] = getConstraintSample(y, 6800, l, u, X);
b = C(:,4)';
yind = C(:,3)';
% [Ml1 z] = supervised_trace_learning_uzawa(deltaX, b, yind, 10);
[Ml1 z] = supervised_trace_learning_uzawa(deltaX, b, yind, 0.01, [], X, Xt, y, yt);
% [Ml1 Ml1t z] = semisupervised_trace_learning_uzawa_eig(deltaX, b, yind,
% D, 0.01, Phi'*Phi, X, Xt, y, yt);
