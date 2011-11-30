function [err,yy,Value]=energyclassify(L,x,y,xTest,yTest,Kg,varargin);
% function [err,yy,Value]=energyclassify(L,x,y,xTest,yTest,Kg,varargin);
%

% checks
D=length(L);
x=x(1:D,:);
xTest=xTest(1:D,:);
if(size(x,1)>length(L)) error('x and L must have matching dimensions!\n');end;

% set parameters
pars.alpha=1e-09;
pars.tempid=0;
pars.save=0;
pars.speed=10;
pars.skip=0;
pars.factor=1;
pars.correction=15;
pars.prod=0;
pars.thresh=1e-16;
pars.ifraction=1;
pars.scale=0;
pars.obj=0;
pars.union=1;
pars.tabularasa=Inf;
pars.blocksize=500;
pars=extractpars(varargin,pars);


pars


tempname=sprintf('temp%i.mat',pars.tempid);
% Initializationip
[D,N]=size(x);
[gen,NN]=getGenLS(x,y,Kg,pars);



if(pars.scale)
 fprintf('Scaling input vectors!\n');
 sc=sqrt(mean(sum( ((x-x(:,NN(end,:)))).^2)));
 x=x./sc;
 xTest=xTest./sc;
end;


Lx=L*x;
Lx2=sum(Lx.^2);
LxT=L*xTest;


for inn=1:Kg
 Ni(inn,:)=sum((Lx-Lx(:,NN(inn,:))).^2)+1;
end;

MM=min(y);
y=y-MM+1;
un=unique(y);
Value=zeros(length(un),length(yTest));

B=pars.blocksize;
if(size(x,2)>50000) B=250;end;
NTe=size(xTest,2);
for n=1:B:NTe
  fprintf('%2.2f%%: ',n/NTe*100);
  nn=n:n+min(B-1,NTe-n);
  DD=distance(Lx,LxT(:,nn));  
 for i=1:length(un)
 % Main Loopfor iter=1:maxiter 
  testlabel=un(i);
  fprintf('%i.',testlabel+MM-1);
  
  enemy=find(y~=testlabel);
  friend=find(y==testlabel);

  Df=mink(DD(friend,:),Kg);
  Value(i,nn)=sumiflessv2(DD,Ni(:,enemy),enemy)+sumiflessh2(DD,Df,enemy);
  
 end;
 fprintf('\n');
end;

 fprintf('\n');
 [temp,yy]=min(Value);

 yy=un(yy)+MM-1;
err=sum(yy~=yTest)./length(yTest);
fprintf('Energy error:%2.2f%%\n',err*100);



function dis=computeDistances(ind1,ind2,x2,x);
  if(isempty(ind1)) dis=[];return;end;
  dim=size(x,1);
  N=size(ind1,2);
  dis=zeros(1,N);
  B=floor(50000000/dim);
  for i=1:B:N
   BB=min(B-1,N-i);
   dis(i:i+BB)=x2(ind2(i:i+BB))+x2(ind1(i:i+BB))-2.*sum(x(:,ind1(i: ...
						  i+BB)).*x(:,ind2(i:i+BB)));
   if(i>1) fprintf('>');end;
  end;





function imp=getImpLS(x,y,Kg,Ki,pars);
[D,N]=size(x);
if(pars.skip) load('.LSKInn.mat');
else

un=unique(y);
Inn=zeros(Ki,N);
for c=un
  fprintf('%i nearest imposture neighbors for class %i :',Ki,c);
  i=find(y==c);
  j=find(y~=c);
  nn=LSKnn(x(:,j),x(:,i),1:Ki);
  Inn(:,i)=j(nn);
  fprintf('\n');
end;

end;

imp=[vec(Inn(1:Ki,:)')'; vec(repmat(1:N,Ki,1)')'];
imp=unique(imp','rows')'; % Delete dublicates
if(pars.save)
save('.LSKInn.mat','Inn');
end; 



function [gen,NN]=getGenLS(x,y,Kg,pars);
fprintf('Computing nearest neighbors ...\n');
[D,N]=size(x);
if(pars.skip) load('.LSKGnn.mat');
else
un=unique(y);
Gnn=zeros(Kg,N);
for c=un
fprintf('%i nearest genuine neighbors for class %i:',Kg,c);
i=find(y==c);
nn=LSKnn(x(:,i),x(:,i),2:Kg+1);
Gnn(:,i)=i(nn);
fprintf('\n');
end;

end;
NN=Gnn;
gen1=vec(Gnn(1:Kg,:)')';
gen2=vec(repmat(1:N,Kg,1)')';

gen=[gen1;gen2];

if(pars.save)
save('.LSKGnn.mat','Gnn');
end; 




function imp=checkup(L,x,y,Ki,NN,pars);
fprintf('Computing nearest neighbors ...\n');
[D,N]=size(x);

Lx=L*x;
Ni=sum((Lx-Lx(:,NN)).^2)+1;
un=unique(y);
imp=[];

for c=un
fprintf('All nearest imposture neighbors for class %i :',c);
i=find(y==c);
j=find(y~=c);
[limp1,limp2]=LSImps(Lx(:,j),Lx(:,i),Ni(i),pars);
imp=[imp [i(limp1);j(limp2)]];
fprintf('\n');
end;
imp=unique(sort(imp)','rows')';


function [limp1,limp2]=LSImps(X1,X2,Thresh,pars);
B=5000;
[D,N2]=size(X2);
N1=size(X1,2);

limp1=[];limp2=[];
sx1=sum(X1.^2);
sx2=sum(X2.^2);
for i=1:B:N2
  BB=min(B-1,N2-i);
  fprintf('.');
%  Dist=distance(X1,X2(:,i:i+BB));
  Dist=addh(addv(-2*X1'*X2(:,i:i+BB),sx1),sx2(i:i+BB));
 
  fprintf('.');

  imp=find(Dist<repmat(Thresh(i:i+BB),N1,1))';
  [a,b]=ind2sub([N1,N2],imp);
  limp1=[limp1 b];
  limp2=[limp2 a]; 
  fprintf('.'); 
 
  clear('nn','dist');
  fprintf('(%i%%) ',round((i+BB)/N2*100)); 
end;

fprintf(' [%i] ',length(limp1));

return;


function NN=LSKnn(X1,X2,ks,pars);
B=2000;
[D,N]=size(X2);
NN=zeros(length(ks),N);
DD=zeros(length(ks),N);

for i=1:B:N
  BB=min(B,N-i);
  fprintf('.');
  Dist=distance(X1,X2(:,i:i+BB));
  fprintf('.');
%  [dist,nn]=sort(Dist);
  [dist,nn]=mink(Dist,max(ks));
  clear('Dist');
  fprintf('.'); 
%  keyboard;
  NN(:,i:i+BB)=nn(ks,:);
  clear('nn','dist');
  fprintf('(%i%%) ',round((i+BB)/N*100)); 
end;


function I=updateOuterProduct(x,vals,active1,active2,a1,a2);


  


function op=SOP(x,a,b);

[D,N]=size(x);
B=500000;
op=zeros(D^2,1);
for i=1:B:length(a)
  BB=min(B,length(a)-i);
  Xa=x(:,a(i:i+BB));
  Xb=x(:,b(i:i+BB));
  XaXb=Xa*Xb';
  op=op+vec(Xa*Xa'+Xb*Xb'-XaXb-XaXb');
 
  if(i>1)   fprintf('.');end;
end;


function v=vec(M);
% vectorizes a matrix

v=M(:);

