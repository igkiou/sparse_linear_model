% function [idx,netsim,dpsim,expref] = migEMquant(x, S, K, EMruns)
%
% Initializes k-centers with the clustering partition found by a
% an isotropic diagonal mixture of Gaussians.
%
% Input:
%    x is a d-by-N matrix containing N training cases of d-dimensional data
%    S is an N-by-N similarity matrix
%    K is the number of clusters
%    EMruns is the number of EM restarts to use
%
function [idx,netsim,dpsim,expref] = migEMquant(x, S, K, EMruns)

EMits = 100;
KCits = 100;
I = length(S);
if size(S,1)~=size(S,2), error('Similarity matrix S must be square'); end;
if size(x,2)~=size(S,2), x=x'; warning('Using x'' instead of x'); end;
if size(x,2)~=size(S,2), error('Invalid matrix dimensions; X must contain data vectors for S'); end;
idx=zeros(I,EMruns,'uint32'); netsim=zeros(1,EMruns); dpsim=zeros(1,EMruns); expref=zeros(1,EMruns);
for tt=1:EMruns,
	init=[]; while length(unique(init))~=K,	
		[p,mu,phi,lPxtr,logPcx] = migEM(x,K,EMits,(std(x(:))/K/10)^2);
		[junk,init] = max(logPcx);
	end;
	[idx(:,tt),netsim(tt),dpsim(tt),expref(tt)] = kcenters(S,K,'init',init);
end;
[netsim,tt]=sort(netsim); idx=idx(:,tt); dpsim=dpsim(tt); expref=expref(tt);
return