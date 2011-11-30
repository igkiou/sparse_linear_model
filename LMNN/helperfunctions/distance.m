function dist=distance(X,x)
% dist=distance(X,x)
%
% computes the pairwise squared distance matrix between any column vectors in X and
% in x
%
% INPUT:
%
% X     dxN matrix consisting of N column vectors
% x     dxn matrix consisting of n column vectors
%
% OUTPUT:
%
% dist  Nxn matrix 
%
% Example:
% Dist=distance(X,X);
% is equivalent to
% Dist=distance(X);
%
% $Id: distance.m 1223 2007-12-02 21:41:10Z kilianw $

[D,N] = size(X);


  
 if(nargin>=2)
  [d,n] = size(x);
  if(D~=d)
   error('Both sets of vectors must have same dimensionality!\n');
  end;
  X2 = sum(X.^2,1);
  x2 = sum(x.^2,1);
%   if(exist('addchv') & isreal(X))
%    dist=addchv(X.'*x,-2,x2,X2); 
%   else
%    dist = repmat(x2,N,1)+repmat(X2.',1,n)-2*X.'*x; 
	dist = bsxfun(@plus, x2, bsxfun(@minus, X2.', 2*X.'*x)); 
%   end;
  
 else
  [D,N] = size(X);
%  if(exist('pdistBOGO','file'))
%   dist=squareform(pdist(X')).^2;   
%  else      
%    if(exist('addchv') & isreal(X))
%     X2 = sum(X.^2,1);
%     dist=addchv(X.'*X,-2,X2,X2);
%    else
    X2 = repmat(sum(X.^2,1),N,1);
    dist = X2+X2.'-2*X.'*X;   
%    end;
%  end;
 end;


