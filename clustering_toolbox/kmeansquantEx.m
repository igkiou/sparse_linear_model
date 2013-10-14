% function [idx,netsim,dpsim,expref] = kmeansquantEx(x, S, K, EMruns)
%
% Initializes k-centers with the quantized means (exemplars) found by the
% k-means algorithm.
%
% Input:
%    x is a d-by-N matrix containing N training cases of d-dimensional data
%    S is an N-by-N similarity matrix
%    K is the number of clusters
%    EMruns is the number of EM restarts to use
%
function [idx,netsim,dpsim,expref] = kmeansquantEx(x, S, K, KMruns)

KMits = 100;
KCits = 100;
I = length(S);
if size(S,1)~=size(S,2), error('Similarity matrix S must be square'); end;
if size(x,2)~=size(S,2), x=x'; warning('Using x'' instead of x'); end;
if size(x,2)~=size(S,2), error('Invalid matrix dimensions; X must contain data vectors for S'); end;
idx=zeros(I,KMruns,'uint32'); netsim=zeros(1,KMruns); dpsim=zeros(1,KMruns); expref=zeros(1,KMruns);
for tt=1:KMruns,
   	init=[]; while length(unique(init))~=K,	
        [junk,mu]=kmeans(x,K,ceil(I*rand));
		init=zeros(1,K); for k=1:K, [junk,init(k)]=min(sum((x-mu(:,k(ones(1,size(x,2))))).^2)); end;
	end;
	[idx(:,tt),netsim(tt),dpsim(tt),expref(tt)] = kcenters(S,K,'init',init);
end;
[netsim,tt]=sort(netsim); idx=idx(:,tt); dpsim=dpsim(tt); expref=expref(tt);
return