function M = MaxMax(x)
%MAXMAX Maximum of all elements in an n-dimension array
% S = MAXMAX(X)
%

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

M=x;
for i = 1:ndims(x)
   M=max(M);
end
