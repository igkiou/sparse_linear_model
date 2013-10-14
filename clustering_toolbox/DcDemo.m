%Example script for David Corney's Clustering Toolbox

k=5;ndim=10;stdev=3;nk=50;
fprintf('Creating data with %d clusters\n',k);
[data,labels]=MakeGaussData(k,ndim,stdev,nk);
%PCAGraph(data,2,labels);



reps=5;
fprintf('\nk-means results:\n')
[rit,ris,c]=doclustering('kmeans',data,k,labels,reps);
fprintf('Rand-to-truth:    \t%1.4f\n', mean(rit));
fprintf('Rand consistency: \t%1.4f\n',mean(mean(ris)));

fprintf('\nFuzzy results:\n')
[rit,ris,c]=doclustering('fuzzy',data,k,labels,reps);
fprintf('Rand-to-truth:    \t%1.4f\n', mean(rit));
fprintf('Rand consistency: \t%1.4f\n',mean(mean(ris)));

fprintf('\nMixture model results:\n')
[rit,ris,c]=doclustering('EM_spherical',data,k,labels,reps);
fprintf('Rand-to-truth:    \t%1.4f\n', mean(rit));
fprintf('Rand consistency: \t%1.4f\n',mean(mean(ris)));

fprintf('\nFinding "too few" clusters: (k-means):\n')
[rit,ris,c]=doclustering('kmeans',data,2,labels,reps);
fprintf('Rand-to-truth:    \t%1.4f\n', mean(rit));
fprintf('Rand consistency: \t%1.4f\n',mean(mean(ris)));
