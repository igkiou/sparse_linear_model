function [idx,netsim,dpsim,expref] = klogk(S,K,runs),

	idx=zeros(length(S),runs,'uint32'); netsim=zeros(1,runs); dpsim=zeros(1,runs); expref=zeros(1,runs);
	for tt=1:runs,
		[idx(:,tt),netsim(tt),dpsim(tt),expref(tt)] = kcenters(S,K,'init',unique(kccprune(S,K))');
	end;
	[netsim,tt]=sort(netsim); idx=idx(:,tt); dpsim=dpsim(tt); expref=expref(tt);
return

% [idx,en]=kccprune(S,K,nruns)
%
% Identifies K exemplars (data centers) using a two-round EM-style
% algorithm starting with K log K clusters and pruning away all but K 
% clusters after the first round of EM (c.f. Dasgupta et al).
%
% For n data points, the input is an n x n
% similarity matrix, S. S(i,k) is the similarity that data point i has 
% to data point k (its potential cluster center). The identified exemplars 
% and the assignments of other data points to these exemplars are 
% returned, along with the energy or cost of the solution (negative net 
% similarity). The diagonal of S contains preferences, indicating for 
% each data point how preferable it is as an exemplar. Without other 
% available information, a good choice is to set all diagonal elements 
% equal to the median of the non-diagnonal elements of the similarity 
% matrix. Similarities are assumed to be additive (so, if they are 
% computed from a probability model, use log-probabilities.) 
% 
% Input:
%
%   S         Similarity matrix (see above)
%   K         Number of clusters to find
%   nruns     Number of runs to try, where each run is initialized
%             randomly (default 1)
%
% Ouput:
%
%   idx(i)    Index of the data point that data point i is assigned
%             to. idx(i)=i indicates that point i is an exemplar
%   en        Energy or cost achieved
%
% Copyright Brendan J. Frey, Dec 2006. This software may be freely used
% and distributed for non-commercial purposes.
%

function [idx,en]=kccprune(S,K,nruns);

if nargin<2 error('kccprune:1','Too few input arguments'); elseif nargin==2 nruns=1; elseif nargin>3 error('kccprune:2','Too many input arguments'); end;

n=size(S,1); en=zeros(1,nruns); idx=zeros(n,nruns); for rep=1:nruns
    % First round of EM
    KlogK=min(n,ceil(K*log(K))); % Maybe divide this by 2 if not much data
    tmp=randperm(n)'; mu=tmp(1:KlogK); % Pick klogk exemplars at random
    [tmp cl]=max(S(:,mu),[],2); % Find class assignments
    cl(mu)=1:KlogK; % Set assignments of exemplars to themselves
    mp=zeros(KlogK,1); % Mixing proportions
    for j=1:KlogK % For each class, find new exemplar and mixing proportion
        I=find(cl==j);
        [Scl ii]=max(sum(S(I,I),1));
        mu(j)=I(ii(1));
        mp(j)=length(I)/n;
    end;

    % Prune small clusters and perform farthest-first traversal
    wT=2/(KlogK)+2/n; % Using 1/KlogK+2/n works better sometimes
    ii=find(mp>=wT);
    nii=find(mp<wT); [yy jj]=sort(mp(nii),'descend');
    mu=mu([ii;nii(jj(1:K-length(ii)))]); nn=length(mu);
    SS=S(mu,mu)+S(mu,mu)'; for i=1:nn SS(i,i)=Inf; end;
    mumu=zeros(K,1); mumu(1)=ceil(rand*nn); mxS=SS(:,mumu(1));
    for k=2:K
        [yy ii]=min(mxS);
        mumu(k)=ii(1);
        mxS=max(mxS,SS(:,mumu(k)));
    end;
    mu=mu(mumu);

    % Second round of EM
    [tmp cl]=max(S(:,mu),[],2); % Find class assignments
    cl(mu)=1:K; % Set assignments of exemplars to themselves
    idx=zeros(n,1);
    for j=1:K % For each class, find new exemplar
        I=find(cl==j);
        [Scl ii]=max(sum(S(I,I),1)); if numel(Scl)==0 keyboard; end;
        en(1,rep)=en(1,rep)-Scl(1);
        mu(j)=I(ii(1));
        idx(I)=mu(j);
    end;
end;

