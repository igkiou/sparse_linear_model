function [C deltaX inds1 inds2] = getConstraintSample(labels, numConstraintSamples, l, u, X)
% C = getConstraintSample(labels, numConstraintSamples, l, u, trainingSamples)
%
% Get ITML constraint matrix from true labels.  See ItmlAlg.m for
% description of the constraint matrix format

numSamples = length(labels);
C = zeros(numConstraintSamples, 4);
inds1 = randperm(numSamples);
inds2 = randperm(numSamples);
C(1:numConstraintSamples, 1) = inds1(1:numConstraintSamples);
C(1:numConstraintSamples, 2) = inds2(1:numConstraintSamples);
sameClass = (labels(inds1(1:numConstraintSamples)) == labels(inds2(1:numConstraintSamples)));
C(sameClass, 3) = 1;
C(sameClass, 4) = l;
C(~sameClass, 3) = -1;
C(~sameClass, 4) = u;
deltaX = X(:, inds1) - X(:, inds2);
