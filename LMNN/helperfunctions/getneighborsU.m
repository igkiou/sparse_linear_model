function neighbors=getneighborsU(X,K);

ne=getneighbors(X,K);
[D N]=size(X);

for i=1:N
    neighbors{i}=[];
end;

for i=1:N
 neighbors{i}=merge(sort(neighbors{i}),sort(ne(:,i)));
 for j=1:K
    neighbors{ne(j,i)}=merge(neighbors{ne(j,i)}, i);
 end;
end;




