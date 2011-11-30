function k=neighbors(X,K);
[D N]=size(X);

if(nargin<2)
    K=2;
end;

k=K;
while(k<N & (1-connected(X,k)))
    k=k+1;
    fprintf('Trying %i Neighbors\n',k);
end;
    
    


function result=connected(X,K);

% result = connecteddfs (X,K)
%
% X input vector
% K number of neighbors
%
% Returns: result = 1 if connected 0 if not connected

if(nargin<2)
    fprintf('Number of Neighbors not specified!\nSetting K=4\n');
    K=4;    
end;
neighbors=getneighbors(X,K);

[K N]=size(neighbors);

oldchecked=[];
checked=[1];


i=1;
while((size(checked)<N) & (length(oldchecked)~=length(checked)))  
 next=neighbors(:,checked);
 next=transpose(sort(next(:)));
 next2=[next(2:end) 0];
 k=find(next-next2~=0);
 next=next(k);

 
 oldchecked=checked; 
 checked=neighbors(:,next(:));
 checked=transpose(sort(checked(:)));
 checked2=[checked(2:end) 0];
 k=find(checked-checked2~=0);
 checked=checked(k);
 
%  if(length(oldchecked)==length(checked))
%      prod(double(checked==oldchecked));     
%  end;
end;


result=(length(checked)==N);



function neighbors=getneighbors(X,K);
% PAIRWISE DISTANCES
[D,N] = size(X);
X2 = sum(X.^2);
dotProd = X'*X;
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*dotProd;

% NEIGHBORS
[sorted,index] = sort(distance);
neighbors = index(2:(1+K),:);