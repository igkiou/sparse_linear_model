function S = SumSum(x)
%SUMSUM Sums all elements in an n-dimension array
% S = SUMSUM(X)
%

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

S=x;
for i = 1:ndims(x)
   S=sum(S);
end
