function k=neighborsU(X,K,maxk);
% function k=neighborsU(X,K,maxk);
%
  [D N]=size(X);

if(nargin<3)
  maxk=N;
end;


if(nargin<2)
    K=2;
end;

k=K;
while(k<=maxk & (1-connectedU(X,k)))
    k=k+1;
    fprintf('Trying %i Neighbors\n',k);
end;

