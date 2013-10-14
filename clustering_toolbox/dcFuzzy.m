function [class,U,centres,error] = dcFuzzy(X,c,m,InitCentres)
%DCFUZZY Performs fuzzy c-means clustering
% [Class,U] = DCFUZZY(Data,c,m,InitCentres) where
% Data is matrix with variables as columns and records as rows,
% c is number of clusters to find,
% m is degree of fuzziness (default = 1.25),
% InitCentres is initial centres of clusters (default - randomly 
% chosen from data). 
%
% "Class" is a list of most likely cluster labels, U is the fuzzy membership matrix.
% "centres" is the centre of each cluster, and "error" is the final error value.

% Copyright (C) David Corney 2000		D.Corney@cs.ucl.ac.uk


[R,ndim]=size(X);

if exist('m')~=1
   m=1.5;
end
mpower=2/(m-1);	%saves time later...

if exist('InitCentres')~=1
   InitCentres= ChooseInitialCentres(X,c);
end

U=zeros(c,R);				%partition (membership) matrix: [0..1]

for i=1:c			
   ThisCentre=repmat(InitCentres(i,:),R,1);
   U(i,:)=sum((ThisCentre-X).^2,2)';
end
U=U./(maxmax(U));

OldU=U;
MaxIter=250;
v=zeros(c,ndim);

for r=1:MaxIter
   U=U.^m;
   for i = 1:c
      sumUi=sum(U(i,:));
      for j=1:ndim
			v(i,j)=sum(U(i,:).*X(:,j)')/sumUi;
      end
   end
   
   %Next block is faster code, inspired by Peter Acklam, Oslo, www.math.uio.no/~jacklam
   Xt=permute(X, [1 3 2]);
   vt=permute(v, [3 1 2]);
   dik=sqrt(sum(abs(Xt(:,ones(1,c),:) - vt(ones(1,R),:,:)).^2,3));
   %[zi,zj]=find(dik==0);dik(zi,zi)=1;
   for i = 1:c
      td(i,:)=sum((repmat(dik(:,i),1,c)./dik).^mpower,2)';
   end
   %td(zi,:)=zeros(1,c);td(zi,zj)=1;   
   U=1./td;
   
   % Older, slower method:
   %   for i = 1:c
   %      for k=1:R
   %         dik=EuclDist(X(k,:),v(i,:));
   %         for j=1:c
   %            djk=sqrt(sum((X(k,:)-v(j,:)).^2));
   %            U(i,k)=U(i,k)+(dik/djk).^(mpower);
   %         end
   %      end
   %   end
   %   U=1./U;
	
   Change=max(sqrt(sum((OldU-U).^2,2)));
   if Change<1e-5
      break		%convergence reached
   end
   OldU=U;
end

for i = 1:c
   sumUi=sum(U(i,:));
   for j=1:ndim
      v(i,j)=sum(U(i,:).*X(:,j)')/sumUi;
   end
end

[strength,class]=max(U);
centres=v;
error=sum((U.^m).*(dik.^2)',1);