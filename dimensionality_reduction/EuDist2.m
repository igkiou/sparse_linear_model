function D = EuDist2(fea_a,fea_b,bSqrt)
% Euclidean Distance matrix
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b


% if ~exist('bSqrt','var')
%     bSqrt = 1;
% end
% 
% 
% if (~exist('fea_b','var')) | isempty(fea_b)
%     [nSmp, nFea] = size(fea_a);
% 
%     aa = sum(fea_a.*fea_a,2);
%     ab = fea_a*fea_a';
%     
%     aa = full(aa);
%     ab = full(ab);
% 
%     if bSqrt
%         D = sqrt(repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab);
%         D = real(D);
%     else
%         D = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
%     end
%     
%     D = max(D,D');
%     D = D - diag(diag(D));
%     D = abs(D);
% else
%     [nSmp_a, nFea] = size(fea_a);
%     [nSmp_b, nFea] = size(fea_b);
%     
%     aa = sum(fea_a.*fea_a,2);
%     bb = sum(fea_b.*fea_b,2);
%     ab = fea_a*fea_b';
% 
%     aa = full(aa);
%     bb = full(bb);
%     ab = full(ab);
% 
%     if bSqrt
%         D = sqrt(repmat(aa, 1, nSmp_b) + repmat(bb', nSmp_a, 1) - 2*ab);
%         D = real(D);
%     else
%         D = repmat(aa, 1, nSmp_b) + repmat(bb', nSmp_a, 1) - 2*ab;
%     end
%     
%     D = abs(D);
% end

if nargin < 1
   error('Not enough input arguments');
end

if exist('fea_b','var') && ~isempty(fea_b) && (size(fea_a, 2) ~= size(fea_b, 2)),
	error('fea_a and fea_b should be of same dimensionality');
end

if ~isreal(fea_a) || (exist('fea_b','var') && ~isreal(fea_b))
	warning('Computing distance table using imaginary inputs. Results may be off.'); 
end

if ~exist('bSqrt','var')
    bSqrt = 1;
end

% Padd zeros if necessray

if (~exist('fea_b','var')) || isempty(fea_b),
	if (size(fea_a, 2) == 1),
		fea_a = [fea_a, zeros(size(fea_a, 1), 1)]; 
	end;
	aa = sum(fea_a .* fea_a, 2);
	if bSqrt,
		D = sqrt(bsxfun(@plus, aa, bsxfun(@minus, aa', 2 * fea_a * fea_a')));
        D = real(D);
    else
        D = bsxfun(@plus, aa, bsxfun(@minus, aa', 2 * fea_a * fea_a'));
	end;
	D = max(D,D');
	D = D - diag(diag(D));
	D = abs(D);

else
	if (size(fea_a, 2) == 1),
		fea_a = [fea_a, zeros(size(fea_a, 1), 1)]; 
		fea_b = [fea_b; zeros(size(fea_b, 1), 1)]; 
	end;
	if bSqrt,
		D = sqrt(bsxfun(@plus, sum(fea_a .* fea_a, 2), bsxfun(@minus, sum(fea_b .* fea_b, 2)', 2 * fea_a * fea_b')));
        D = real(D);
    else
        D = bsxfun(@plus, sum(fea_a .* fea_a, 2), bsxfun(@minus, sum(fea_b .* fea_b, 2)', 2 * fea_a * fea_b'));
	end;
	D = abs(D);
end;

% Compute distance table


% Make sure result is real


