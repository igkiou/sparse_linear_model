function M = MeanMean(x)
%MEANMEAN Mean of all elements in an n-dimension array
% S = MEANMEAN(X)
%

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

M=x;
for i = 1:ndims(x)
   M=mean(M);
end
