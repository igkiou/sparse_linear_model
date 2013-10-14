% function [idx,netsim,dpsim,expref] = mdgEMquantEx(x, S, K, EMruns)
%
% Initializes k-centers with the quantized means (exemplars) found by a
% a diagonal mixture of Gaussians.
%
% Input:
%    x is a d-by-N matrix containing N training cases of d-dimensional data
%    S is an N-by-N similarity matrix
%    K is the number of clusters
%    EMruns is the number of EM restarts to use
%
function [idx,netsim,dpsim,expref] = mdgEMquantEx(x, S, K, EMruns)

EMits = 100;
KCits = 100;
I = length(S);
if size(S,1)~=size(S,2), error('Similarity matrix S must be square'); end;
if size(x,2)~=size(S,2), x=x'; warning('Using x'' instead of x'); end;
if size(x,2)~=size(S,2), error('Invalid matrix dimensions; X must contain data vectors for S'); end;
idx=zeros(I,EMruns,'uint32'); netsim=zeros(1,EMruns); dpsim=zeros(1,EMruns); expref=zeros(1,EMruns);
for tt=1:EMruns,
	init=[]; while length(unique(init))~=K,	
		[p,mu,phi,lPxtr] = mdgEM(x,K,EMits,(std(x(:))/K/10)^2);
		init=zeros(1,K); for k=1:K, [junk,init(k)]=min(sum((x-mu(:,k(ones(1,size(x,2))))).^2)); end;
	end;
	[idx(:,tt),netsim(tt),dpsim(tt),expref(tt)] = kcenters(S,K,'init',init);
end;
[netsim,tt]=sort(netsim); idx=idx(:,tt); dpsim=dpsim(tt); expref=expref(tt);
return

% function [p,mu,phi,lPxtr] = migEM(x,K,its,minphi)
%
% Performs EM for a mixture of K axis-aligned (diagonal covariance
% matrix) Gaussians. its iterations are used and the input variances are
% not allowed to fall below minphi (if minphi is not given, its default
% value is 0). The parameters are randomly initialized using the mean
% and variance of each input.
%
% Input:
%
%   x(:,t) = the N-dimensional training vector for the tth training case
%   K = number of Gaussians to use
%   its = number of iterations of EM to apply
%   minphi = minimum variance of sensor noise (default: 0)
%  
% Output:
%
%   p = probabilities of clusters
%   mu(:,c) = mean of the cth cluster
%   phi(:,c) = variances for the cth cluster
%   lPxtr(i) = log-probability of data after i-1 iterations
%
% Copyright 2001 Brendan J. Frey
%

function [p,mu,phi2,lPxtr,logPcx] = mdgEM(x,K,its,minphi)

if nargin==3, minphi = 0; end;
N = size(x,1); T = size(x,2);

% Initialize the parameters
p = 10+rand(K,1); p = p/sum(p);
mu = mean(x,2)*ones(1,K) + std(x(:))*randn(N,K);
phi2 = var(x(:))*ones(1,K)*2; phi2(phi2<minphi) = minphi;

% Do its iterations of EM
lPxtr = zeros(its,1);
for i=1:its
	% Do the E step
	r = zeros(K,1); rx = zeros(N,K); rDxm2 = zeros(N,K); lPx = zeros(1,T);
	iphi2 = 1./phi2;
	logNorm = log(p)-1/2*N*log(2*pi)-1/2*log(phi2');
	logPcx = zeros(K,T);
	for k=1:K, logPcx(k,:) = logNorm(k) - 0.5*sum((iphi2(k)*ones(N,T)).*(x-mu(:,k)*ones(1,T)).^2,1); end;
	mx = max(logPcx,[],1); Pcx = exp(logPcx-ones(K,1)*mx); norm = sum(Pcx,1);
	PcGx = Pcx./(ones(K,1)*norm); lPx = log(norm) + mx;
	lPxtr(i) = sum(lPx);
%   plot([0:i-1],lPxtr(1:i),'r-');
%   title('Log-probability of data versus # iterations of EM');
%   xlabel('Iterations of EM');
%   ylabel('log P(D)');
%   drawnow;
% 	plotfig(x,mu,phi2);
	r = mean(PcGx,2);
	rx = zeros(N,K);
	for k=1:K, rx(:,k) = mean(x.*(ones(N,1)*PcGx(k,:)),2); end;

	% Do the M step
	p = r;
	mu = rx./(ones(N,1)*r');
	for k=1:K, phi2(k) = PcGx(k,:)*mean((x-mu(:,k(ones(1,T)))).^2)'/sum(PcGx(k,:)); end;
	phi2(phi2<minphi) = minphi;
	if i>1, if abs((lPxtr(i)-lPxtr(i-1))/(.5*lPxtr(i)+.5*lPxtr(i-1)))<1e-5, break; end; end;
end;
lPxtr(i+1:end)=[];
return