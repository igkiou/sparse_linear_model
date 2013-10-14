function [Classes,Centres,FinalDistance]=dcKMeans(Data,k,InitCentres,MaxIters)
%DCKMEANS Performs k-means clustering
% Classes=DCKMEANS(Data,k,InitCentres) where Data is a matrix with
% variables in columns and records in rows, k is the number of 
% clusters to search for and InitCentres is a list of initial centres.
% Classes=DCKMEANS(Data,k) randomly chooses initial centres from data set.
% [Classes,Centres,FinalDistance]=DCKMEANS(Data,k,InitCentres) also returns
% the centres and distances of each point to its nearest centre.

% Copyright (C) David Corney 2000		D.Corney@cs.ucl.ac.uk

if nargin < 3
	InitCentres = ChooseInitialCentres(Data,k);		%randomly choose starting point (where needed)
end
Centres=InitCentres;
OldCentres=Centres;
if nargin<4
   MaxIters=500;
end

[R,C]=size(Data);

DataSq=repmat(sum(Data.^2,2),1,k);	%sum squared data - save re-calculating repeatedly later
%Do we need DataSq? It's constant, and we're minimsing things...


for i = 1:MaxIters
   
   Dist = DataSq + repmat(sum((Centres.^2)',1),R,1) - 2.*(Data*(Centres'));   %i.e. d^2 = (x-c)^2 = x^2 + c^2 -2xc

   [D,Centre]=min(Dist,[],2);		%label of nearest centre for each point

   for j=1:k
      idx=find(Centre==j);
      if length(idx)>0
         Centres(j,:)=mean(Data(idx,:));
      end
   end
   
   Change=sum(sum(abs(OldCentres-Centres)));
   if Change < 1e-10	%Have we converged yet?
      break
   end
   OldCentres=Centres;
   
end


[FinalDistance,Classes]=min(Dist,[],2);		%label points one last time
