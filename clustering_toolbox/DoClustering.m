function [RITrue,RISelf,EstIndex]=DoClustering(Algorithm,Data,k,TrueID,Repetitions)
%DOCLUSTERING Performs cluster analysis of specified data using specified
% algorithm.
% [RITrue,RISelf,Index]=DOCLUSTERING(Algorithm,Data,k,TrueID,Repetitions)
% where Algorithm is the name of the clustering algorithm (string),
% Data is 2-d matrix, with variables as columns and records as rows,
% k is number of clusters to search for,
% TrueID is a vector of class labels, one for each row in "Data", ([]->not known)
% and Repetitions is the number of times to repeat the algorithm (default 1).
% RITrue is the (adjusted) Rand Index comparing the solution to the true labels
% (where known). RISelf is the (adjusted) Rand Index between pairs of solutions
% (when Repetitions > 1). 
% Index is the estimated labels of each record.
%
% AlgorithmList=DOCLUSTERING with no parameters returns list of all 
% the aglorithms available

% Copyright (C) David Corney 2000		D.Corney@cs.ucl.ac.uk

%If no inputs, then return list of all aglorithms used (arbitrary order).
if nargin == 0
   RITrue={'kmeans','EM_spherical','EM_elliptical', ...
         'hier_centroid','hier_complete','hier_single', ...
         'fuzzy','random'};
   return 
end

if nargin < 3
   error('DOCLUSTERING requires at least 3 arguments');
   help(mfilename)
   return
end
if nargin < 4, TrueID=[]; end			%no known class labels
if nargin < 5, Repetitions = 1;end	%only do it once
if nargin < 6, options = [];end

[R,C]=size(Data);
if length(TrueID)>0 & length(TrueID)~=R
   error('DOCLUSTERING number of labels doesn''t match number of records');
   return
end

if k == 1			%Only want one cluster? Well, ...OK
   RITrue=0;RISelf=0;
   EstIndex=ones(R,1);
   return
end

%Hierarchical clustering - calculate distance/dissimilarity matrix
if strncmp(Algorithm,'hier',4) == 1		
   if Repetitions > 1, warning('Only performing one hierarchical clustering execution');end	 %Hier. is deterministic; no point in repeating it
   Repetitions = 1;
   %Next line does fast distance calculations as given by Peter Acklam, Oslo, www.math.uio.no/~jacklam
	distance=sqrt(sum(abs(repmat(permute(Data, [1 3 2]), [1 R 1]) - repmat(permute(Data, [3 1 2]), [R 1 1])).^2,3));
   distance=distance+diag(ones(1,R).*Inf);
end

%Do the clustering
for i =1:Repetitions
   InitCentres = ChooseInitialCentres(Data,k);		%randomly choose starting point (where needed)
   switch Algorithm
   case 'kmeans'
      EstIndex(i,:) = dcKMeans(Data,k,InitCentres)';
   case 'EM_spherical'
      [mix, index,likelihood]=dcEMGMM(Data, k, 'spherical', 1);
      EstIndex(i,:)=index';
   case 'EM_elliptical'
      [mix, index,likelihood]=dcEMGMM(Data, k, 'full', 1);
      EstIndex(i,:)=index';
   case {'hier_complete'}
      cluster = dcAgg(distance, 'complete', k);
      for j = 1:k
         EstIndex(i,cluster{j})=j;
      end
   case {'hier_single'}
      cluster= dcAgg(distance, 'single', k);
      for j = 1:k
         EstIndex(i,cluster{j})=j;
      end
   case {'hier_centroid'}
      cluster = dcAgg(distance, 'centroid', k);
      for j = 1:k
         EstIndex(i,cluster{j})=j;
      end
   case 'fuzzy'
      m=1.5;
      EstIndex(i,:) = dcFuzzy(Data,k,m,InitCentres);
   case 'random'
      EstIndex(i,:) = floor(rand(1,R)*k+1);		%randomly assign labels
   otherwise
      error(sprintf('DOCLUSTERING - unsupported algorithm "%s"',Algorithm))
   end %of switch/case
   
   %return adjusted Rand index (accuracy) if true labels are given
   if length(TrueID)>0
      RITrue(i)=RandIndex(EstIndex(i,:), TrueID);   
   else
      RITrue(i)=-1;
   end
   
end %of repetitions loop

%Calculate adjusted Rand index between each pair of solutions found
if Repetitions > 1			   
   l=0;
   for i=1:Repetitions-1
      for j=i+1:Repetitions
         l=l+1;
         RISelf(l)=RandIndex(EstIndex(i,:)',EstIndex(j,:)');
      end
   end
else
   RISelf=0;
end

