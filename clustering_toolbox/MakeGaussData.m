function [X,ClusterID,centres] = MakeGaussData(NCentres,NDims,StdDev,nk)
%MAKEGAUSSDATA - Creates spherical data clouds
% X = MakeGaussData(NCentres,NDims,StdDev,NPointsPerCluster), where
% NCentres = number of clusters, NDims = no. of dimensions, StdDev = standard
% deviation (width) of each cluster, NPointsPerCluster = no. of points per cluster
% (scalar or vector).
% [X,ClusterID] = MakeGaussData(NCentres,NDims,StdDev,NPointsPerCluster) also
% returns integer cluster label for each data point
% [X,ClusterID,centres] = MakeGaussData(NCentres,NDims,StdDev,NPointsPerCluster)
% returns matrix listing the centre of each cluster
% The centres are drawn uniformly between -5 and +5.
%
% e.g. 
%     k=5;ndim=20;stdev=.5;nk=150;
%     [data,labels]=MakeGaussData(k,ndim,stdev,nk);
%  	PCAGraph(data,2,labels);
%

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

if nargin < 4
   help(mfilename);
   error('MAKEGAUSSDATA requires 4 arguments')
   return
end

if length(nk)==1 & NCentres > 1
   nk=ones(1,NCentres)*nk;
end

if length(nk) ~= NCentres
   error('MAKEGAUSSDATA: number of centres doesn''t match vector of cluster sizes')
   return
end

centres = rand(NCentres,NDims)*11-6;
distance=sqrt(sum(abs(repmat(permute(centres, [1 3 2]), [1 NCentres 1]) - repmat(permute(centres, [3 1 2]), [NCentres 1 1])).^2,3));
distance=distance+diag(ones(1,NCentres).*Inf);
while min(min(distance))<2*StdDev
	centres = rand(NCentres,NDims)*11-6;
	distance=sqrt(sum(abs(repmat(permute(centres, [1 3 2]), [1 NCentres 1]) - repmat(permute(centres, [3 1 2]), [NCentres 1 1])).^2,3));
	distance=distance+diag(ones(1,NCentres).*Inf);
end  

[K,C]=size(centres);
startp=1;endp=1;

for i = 1:K 
   set=randn(nk(i),C)*StdDev+ones(nk(i),1)*centres(i,:);
   endp=startp+size(set,1)-1;
   X(startp:endp,:)=set;
   ClusterID(startp:endp,:)=repmat(i,nk(i),1);
   startp=endp+1;
end
