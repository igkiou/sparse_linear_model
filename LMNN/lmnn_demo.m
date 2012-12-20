fprintf('Loading data ...\n');
load('data/isolet_small.mat');                        
fprintf('Running single metric LMNN  ...\n');
[L,Det]=lmnn2(xTr,yTr,'quiet',1,'maxiter',1000,'validation',0.3);
enerr=energyclassify(L,xTr,yTr,xTe,yTe,3);
knnerrL=knnclassify(L,xTr,yTr,xTe,yTe,3);
knnerrI=knnclassify(eye(size(L)),xTr,yTr,xTe,yTe,3);

fprintf('50-dim Isolet data set:\n');
fprintf('3-NN Euclidean training error: %2.2f\n',knnerrI(1)*100);
fprintf('3-NN Euclidean testing error: %2.2f\n',knnerrI(2)*100);
fprintf('3-NN Malhalanobis training error: %2.2f\n',knnerrL(1)*100);
fprintf('3-NN Malhalanobis testing error: %2.2f\n',knnerrL(2)*100);
fprintf('Energy classification error: %2.2f\n',enerr*100);
fprintf('Training time: %2.2fs\n (As a reality check: My laptop needs 118s)\n\n',Det.time);


% fprintf('\n\nRunning multiple-metrics LMNN  ...\n');
% [Ls,Dets]=MMlmnn(xTr,yTr,3,'initl',L,'verbose',1,'maxiter',300,'validation',0.3,'noatlas',1);
% fprintf('Using metric for classification ...\n')
% enerrLs=MMenergyclassify(Ls,xTr,yTr,xTe,yTe,3);
% knnerrLs=MMknnclassify(Ls,xTr,yTr,xTe,yTe,3);
% 
% clc;
% fprintf('50-dim Isolet data set:\n');
% fprintf('3-NN classification:\n')
% fprintf('Training:\tEuclidean=%2.2f\t1-Metric=%2.2f\tMultipleMetrics=%2.2f\n',knnerrI(1)*100,knnerrL(1)*100,knnerrLs(1)*100)
% fprintf('Testing:\tEuclidean=%2.2f\t1-Metric=%2.2f\tMultipleMetrics=%2.2f\n',knnerrI(2)*100,knnerrL(2)*100,knnerrLs(2)*100)
% fprintf('\n')
% fprintf('Energy classification:\n')
% fprintf('Testing:\t1-Metric=%2.2f\tMultipleMetrics=%2.2f\n',enerr*100,enerrLs*100);


       
