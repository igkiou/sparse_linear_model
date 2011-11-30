M = 100;
N = 50;
numEl = 1000;

inds = randperm(M * N);
inds = inds(1:numEl);
[a b] = ind2sub([M N], inds);
observationMat(:, 1) = a;
observationMat(:, 2) = b;
observationMat(:, 3) = randn(numEl ,1);
Y = randn(N, N);
profile on; profile clear; 
[B iter] = operator_completion_apg(M, N, observationMat, Y);
B1 = operator_completion_apg_mex(M, N, observationMat, Y);
norm(B-B1,'fro')
profile viewer
