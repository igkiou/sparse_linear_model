function neighbors=getneighbors(X,K);
% % % PAIRWISE DISTANCES
% % [D,N] = size(X);
% %  X2 = sum(X.^2,1);
% %  dotProd = X'*X;
% % distance = repmat(X2,N,1)+repmat(X2',1,N)-2*dotProd;

% NEIGHBORS
[sorted,index] = sort(distance(X));
neighbors = index(2:(1+K),:);
