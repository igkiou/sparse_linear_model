% function [idx,netsim,dpsim,expref] = kmnKlogKquant(x, S, K, EMruns)
%
% Initializes k-centers with the clustering partition found by the
% k-log(k) heuristic on k-means.
%
% Input:
%    x is a d-by-N matrix containing N training cases of d-dimensional data
%    S is an N-by-N similarity matrix
%    K is the number of clusters
%    EMruns is the number of EM restarts to use
%
function [idx,netsim,dpsim,expref] = kmnKlogKquant(x, S, K, EMruns),

EMits = 100;
KCits = 100;
I = length(S);
if size(S,1)~=size(S,2), error('Similarity matrix S must be square'); end;
if size(x,2)~=size(S,2), x=x'; warning('Using x'' instead of x'); end;
if size(x,2)~=size(S,2), error('Invalid matrix dimensions; X must contain data vectors for S'); end;
idx=zeros(I,EMruns,'uint32'); netsim=zeros(1,EMruns); dpsim=zeros(1,EMruns); expref=zeros(1,EMruns);
for tt=1:EMruns,
	init=[]; while length(unique(init))~=K,	
		[c,mu] = kmnKlogK(x,K);
		init = c;
	end;
	[idx(:,tt),netsim(tt),dpsim(tt),expref(tt)] = kcenters(S,K,'init',init);
end;
[netsim,tt]=sort(netsim); idx=idx(:,tt); dpsim=dpsim(tt); expref=expref(tt);
return



function [c,mu] = kmnKlogK(x,K)

% load('v:\matlab\affinity_propagation\application_olivetti\data.mat'); x=x'; K=6;
Ktrue = K;
[N,M] = size(x); % N=dataset dimensionality, M=dataset size
KlogK = min(ceil(K*log(K)),M);
K = KlogK;

% initialization
m=randperm(M); m=m(1:K); mu=x(:,m);

% first round of *E*M
d = zeros(K,M);
c = zeros(1,M);
for m=1:M,
	for k=1:K,
		d(k,m) = sum((x(:,m)-mu(:,k)).^2);
	end;
	[junk,c(m)] = min(d(:,m));
end;


% first round of E*M*
mu(:)=0;
for k=1:K, for n=1:N,
	mu(n,k) = mean(x(n,find(c==k)));
end; end;

m = hist(c,1:K);
prune = find(m/M<(1/2/KlogK+2/M)); k=setdiff(1:K,prune);
mu=mu(:,k); K=length(k);

K_ = ceil(K*rand); % select first new cluster centre
dists = +Inf(K,K);
while length(K_)<Ktrue,
	for k=1:K, for k_=K_,
		dists(k,k_) = sum((mu(:,k)-mu(:,k_)).^2);
	end; end;
	[junk, k_] = max(min(dists,[],2)); K_ = [K_ k_];
end;
k=K_; mu=mu(:,k); K=length(k);

% second round of *E*M
d = zeros(K,M);
c = zeros(1,M);
for m=1:M,
	for k=1:K,
		d(k,m) = sum((x(:,m)-mu(:,k)).^2);
	end;
	[junk,c(m)] = min(d(:,m));
end;


% second round of E*M*
mu(:)=0;
for k=1:K,
    m=find(c==k); if isempty(m), continue; end;
    for n=1:N,
    	mu(n,k) = mean(x(n,m));
    end;
end;

return;