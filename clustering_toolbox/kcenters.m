function [idx,netsim,dpsim,expref,varargout] = kcenters(S, K, varargin),
%KCENTERS k-centers clustering
%   [idx,netsim,dpsim,expref] = kcenters(S,K,...) performs K-centers clustering
%   on data represented by the square similarity matrix, S, where S(i,k)
%   is related to the probability of training case i belonging to a cluster with
%   exemplar k.  K is the number of clusters to partition the data into.
%
%  Options include:
%   'runs',#runs ... to specify a number of random restarts (default 1000)
%   'plot' ... to show a CDF of the net similarity for the runs
%   'maxits', #its ... the maximum # iterations for convergence (default 100)
%   'online' ... returns additional matrix of comparisons used

%
%   Delbert Dueck, Brendan Frey (www.psi.toronto.edu)
%     15 Nov. 2005  rev. 20 Apr. 2007

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% input parameter checking
if nargin<2 error('Too few input arguments'); end;
[I1,I2,I3]=size(S); if (I1~=I2 | I3~=1 | ~isreal(S) | ~isnumeric(S)) error('Similarity matrix S must be square'); else I=I1; clear('I1','I2','I3'); end;
if (numel(K)~=1 | ~isnumeric(K) | ~isreal(K)) error('K must be positive scalar'); end;
Sdiag = diag(S);
% if strcmp(class(S),'single'), warning('Now performing k-centers clustering on SINGLE-precision data'); end;

maxits=100; runs=1000; plt=0; details=0; online=0;
i=1; while i<=length(varargin)
	if i==1 & isnumeric(varargin{i}),
		if numel(varargin{1})==I, init=varargin{1}; elseif numel(varargin{1})==1, runs=varargin{1}; end;
		i=i+1; continue;
	end;
	if strcmp(varargin{i},'plot'),
		plt=1; tic; i=i+1;
	elseif strcmp(varargin{i},'details'),
		details=1; i=i+1;
	elseif strcmp(varargin{i},'maxits'),
		maxits=varargin{i+1}; i=i+2;
		if maxits<=0 error('maxits must be a positive integer'); end;
	elseif (strcmp(varargin{i},'restarts') | strcmp(varargin{i},'runs')),
		runs=varargin{i+1}; i=i+2;
		if runs<=0 error('runs must be a positive integer'); end;
	elseif (strcmp(varargin{i},'init') | strcmp(varargin{i},'initialize')),
		init=varargin{i+1}; i=i+2;
	elseif strcmp(varargin{i},'online'),
		online=1; i=i+1;
	else i=i+1;
	end;
end;
mu=[]; c=[];
if exist('init')
	if numel(init)==K, mu=reshape(init,[K 1]); runs=1;
	elseif numel(init)==I && length(unique(init))==K, c=reshape(init,[I 1]); runs=1;
	else clear('init'); end;
end;

% 	if TT==0, [junk,perm]=sort(-sum(L)); mu=perm(1:K); end; % initialize deterministically
if I<2^8, idx=zeros(I,runs,'uint8');
elseif I<2^16, idx=zeros(I,runs,'uint16');
else idx=zeros(I,runs,'uint32'); end;
netsim=zeros(1,runs); dpsim=zeros(1,runs); expref=zeros(1,runs);
if online, ref=cell(1,1); end;

for tt=1:runs,
	
	% INITIALIZATION
% 	if ~exist('mu') mu=randperm(I)'; mu=mu(1:K); end; % random initial exemplars
% 	if ~exist('c'), [junk,c]=max(S(:,mu),[],2); c(mu)=(1:K)'; end; % initial E-step
	if isempty(mu), mu=randperm(I)'; mu=mu(1:K); end; % random initial exemplars
	if isempty(c), [junk,c]=max(S(:,mu),[],2); c(mu)=(1:K)'; end; % initial E-step
	if online, ref{tt}=zeros(I,I,'uint8'); end;
	
	% E-M iterations
	for t = 1:maxits,

		% M-step (update cluster centers)
		muprev = mu;
		for k = 1:K,
			locs = find(c==k);
% 			mu(k) = locs(argmax(sum(S(locs,locs),1)));
			[junk, i] = max(sum(S(locs,locs),1)); mu(k)=locs(i);
			if online, ref{tt}(locs,locs)=ref{tt}(locs,locs)+1; end;
		end;

		mu = sort(mu);
		% E-step (update where everyone's assigned)
		[junk,c] = max(S(:,mu),[],2);
		if online, ref{tt}(:,mu)=ref{tt}(:,mu)+1; end;
		c(mu) = (1:K)'; % exemplars must be in their own clusters

		if all(muprev==mu) break; end;
	end;

	idx(:,tt) = mu(c);
	netsim(tt) = sum(S(sub2ind([I I],(1:I)',mu(c))));
	expref(tt) = sum(Sdiag(mu));
	dpsim(tt) = netsim(tt)-expref(tt);
	if plt,
		if toc>0.1, tic;
			figure(1000);
			plot(sort(netsim(1:tt)),'r-');
			axis tight;
			title('K-CENTERS RESTARTS VISUALIZATION');
			xlabel('Initializations (sorted)');
			ylabel('Fitness (net similarity) of solution');
			drawnow;
		end;
	end;
	mu=[]; c=[];
end;
% PERFORM FINAL SORT
[netsim,tt]=sort(netsim); idx=idx(:,tt); dpsim=dpsim(tt); expref=expref(tt);
if online, varargout{1}=ref; end;
return;

for tt=1:100, imagesc(ref{tt}); title(sprintf('k-centers restart #%d',tt)); pause; end;