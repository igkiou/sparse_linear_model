function M = MinMin(x)
%MINMIN Minimum of all elements in an n-dimension array
% S = MINMIN(X)
%

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

M=x;
for i = 1:ndims(x)
   M=min(M);
end
