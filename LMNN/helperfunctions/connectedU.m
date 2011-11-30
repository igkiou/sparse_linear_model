function result=connectedU(X,K);

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

[D N]=size(X);

ne=getneighborsU(X,K);
maxSize=0;
for i=1:N
    if(length(ne{i})>maxSize) maxSize=length(ne{i});end;
end;
neighbors=ones(maxSize,N);
for i=1:N
    neighbors(1:length(ne{i}),i)=ne{i};    
end;


oldchecked=[];
checked=[1];

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

