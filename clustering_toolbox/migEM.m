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

function [p,mu,phi2,lPxtr,logPcx] = migEM(x,K,its,minphi)

if nargin==3 minphi = 0; end;
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


function plotfig(x,M,phi)
% 	cmap = hsv2rgb([+zeros(256,1) linspace(0,1,256)' ones(256,1)]);
% 	[x_,y_] = meshgrid(-10:.1:+10,-10:.1:+10);
% 	figure; plot3(x(1,:),x(2,:),0*x(2,:),'go', 'MarkerFaceColor','g', 'LineWidth',1.5);
% 	xlabel('x'); ylabel('y'); zlabel('z');
% 	for k=1:size(M,2),
% 		surf(x_,y_,    exp(-(x_-mu(1,k)).^2/2/phi(1,k)-(y_-mu(2,k)).^2/2/phi(1,k))-1, 'LineStyle','none'); hold on;
% 	end;
% 	colormap(cmap)
	figure(1); clf; plot(x(1,:),x(2,:),'go', 'MarkerFaceColor','g', 'LineWidth',1.5); hold on; plot(M(1,:),M(2,:),'rx','MarkerSize',12, 'LineWidth',2);
	w = 2.15; h = 2;
	for k=1:size(M,2),
		rectangle('Position',[M(1,k) M(2,k) 0 0]+[-phi(k) -phi(k) +2*phi(k) +2*phi(k)], 'Curvature',[1 1], 'EdgeColor','r', 'LineWidth',2);
	end;
	xlim([-10 +10]); ylim([-10 +10]);
% 	xlim([floor(min(x(1,:))) ceil(max(x(1,:)))]);
% 	ylim([floor(min(x(2,:))) ceil(max(x(2,:)))]);
	print('-dpng','-r0','deleteme.png'); img=imread('deleteme.png'); delete('deleteme.png');
	[img,map] = rgb2ind(img,256);
	if exist('deleteme.gif'), imwrite(img,map,'deleteme.gif','DelayTime',.1,'WriteMode','append');
	else imwrite(img,map,'deleteme.gif','DelayTime',.1,'LoopCount',Inf,'WriteMode','overwrite'); end;
return
