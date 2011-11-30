function [D, A] = multidictgradient(X, labels, params)
%DICTGRADIENT Coupled K-SVD dictionary and sensing matrix training.
%  [D,PHI,ALPHA] = DICTGRADIENT(PARAMS) runs the coupled K-SVD training
%  algorithm on the specified set of signals, returning the simultaneously
%  trained dictionary D and sensing matrix PHI, as well as the signal
%  representation matrix ALPHA.  
%
%  DICTGRADIENT alternately optimizes D and PHI, with the other matrix being
%  considered fixed. For the optimization of PHI given a fixed D, the
%  optimization problem solved by DICTGRADIENT is given by
%
%      min  |L - L*Gamma'*Gamma*L'|_F^2
%     Gamma
%
%	where [V L] = eig(D) and Gamma = PHI * V. For the optimization of D
%	given a fixed PHI, the optimization problem solved by DICTGRADIENT is
%	given by
%
%   min alpha*|X-D*ALPHA|_F^2+|Y-PHI*D*ALPHA|_F^2  s.t.  |A_i|_0 <= T
%   D,ALPHA
%
%  where X is the set of training signals, Y = PHI*X + H, H is additive
%  Gaussian noise, A_i is the i-th column of ALPHA, T is the target
%  sparsity, and alpha is a regularization parameter. 
%
%  Required fields in PARAMS:
%  --------------------------
%
%    'codinglambda' - Sparse coding regularization parameter.
%
%    'initdict' / 'dictsize' - Initial dictionary / no. of atoms to train.
%      At least one of these two should be present in PARAMS.
%
%      'dictsize' specifies the number of dictionary atoms to train. If it
%      is specified without the parameter 'initdict', the dictionary is
%      initialized with dictsize randomly selected training signals.
%
%      'initdict' specifies the initial dictionary for the training. It
%      should be either a matrix of size NxL, where N=size(X,1), or an
%      index vector of length L, specifying the indices of the examples to
%      use as initial atoms. If 'dictsize' and 'initdict' are both present,
%      L must be >= dictsize, and in this case the dictionary is
%      initialized using the first dictsize columns from initdict. If only
%      'initdict' is specified, dictsize is set to L.
%
%    'initsens' / 'senssize' - Initial sensing matrix / no. of compressed
%      samples. At least one of these two should be present in PARAMS.
%
%      'senssize' specifies the number of rows in the sensing matrix PHI.
%      If it is specified without the parameter 'initsens', the sensing
%      matrix is initialized as a random Gaussian matrix.
%
%      'initsens' specifies the initial sensing matrix for the training. It
%      should be a matrix of size MxN, where N=size(X,1). If 'senssize'
%      and 'initsens' are both present, M must be >= senssize, and in this
%      case the dictionary is initialized using the first senssize rows
%      from sensdict. If only 'initsens' is specified, senssize is set to
%      M. 
%
%	  'noisestd' specifies the standard deviation of the additive Gaussian
%	  noise added to the projected X, as a percentage of the amplitude
%	  of each coefficient.
%
%	  'alpha' specifies the trade-off between the two types of
%	  reconstruction error.
%
%
%  Optional fields in PARAMS:
%  --------------------------
%
%    'iternum' - Number of training iterations.
%      Specifies the number of iterations to perform. If not
%      specified, the default is 10.
%
%
%  Optional fields in PARAMS - advanced:
%  -------------------------------------
%
%    'muthresh' - Mutual incoherence threshold.
%      This parameter can be used to control the mutual incoherence of the
%      trained dictionary, and is typically between 0.9 and 1. At the end
%      of each iteration, the trained dictionary is "cleaned" by discarding
%      atoms with correlation > muthresh. The default value for muthresh is
%      0.99. Specifying a value of 1 or higher cancels this type of
%      cleaning completely. Note: the trained dictionary is not guaranteed
%      to have a mutual incoherence less than muthresh. However, a method
%      to track this is using the VERBOSE parameter to print the number of
%      replaced atoms each iteration; when this number drops near zero, it
%      is more likely that the mutual incoherence of the dictionary is
%      below muthresh.
%
%
%   Summary of all fields in PARAMS:
%   --------------------------------
%
%   Required:
%     'codinglambda'		   sparse coding regularization
%     'initdict' / 'dictsize'  initial dictionary / dictionary size
%
%   Optional (default values in parentheses):
%     'iternum'                number of training iterations (10)
%	  'blockratio'			   portion of samples in block (0)
%     'muthresh'               mutual incoherence threshold (0.99)
%	  'printinfo'			   print progress messages (0)
%	  'errorinfo'			   print objective function value progress (0)
%
%
%  References:
%  [1] M. Aharon, M. Elad, and A.M. Bruckstein, "The K-SVD: An Algorithm
%      for Designing of Overcomplete Dictionaries for Sparse
%      Representation", the IEEE Trans. On Signal Processing, Vol. 54, no.
%      11, pp. 4311-4322, November 2006.
%  [2] M. Elad, R. Rubinstein, and M. Zibulevsky, "Efficient Implementation
%      of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit",
%      Technical Report - CS, Technion, April 2008.
%  [3] J. Duarte-Carvajalino, and G. Sapiro, "Learning to Sense Sparse
%	   Signals: Simultaneous Sensing Matrix and Sparsifying Dictionary
%	   Optimization", the IEEE Trans. On Image Processing, Vol. 18, no. 7,
%      pp. 1395-1408, July 2009.
%
%  See also LEARN_SENSING, RANDOM_SENSING, OMP, OMP2.

%%%%% parse input parameters %%%%%

% coding lambda %

if (isfield(params,'codinglambda')),
  codinglambda = params.codinglambda;
else
  error('Sparse coding regularization parameter not specified'); %#ok
end;

% incoherence lambda %

if (isfield(params,'incoherencelambda')),
  incoherencelambda = params.incoherencelambda;
else
  error('Block incoherence regularization parameter not specified'); %#ok
end;

% iteration count %

if (isfield(params,'iternum')),
  iternum = params.iternum;
else
  iternum = 10;
end;

% inner iteration count %

if (isfield(params,'iternum2')),
  iternum2 = params.iternum2;
else
  iternum2 = 10;
end;

% block ratio %

if (isfield(params,'blockratio')),
  blockratio = params.blockratio;
else
  blockratio = 0;
end;

% status messages %

if (isfield(params,'printinfo')),
  printinfo = params.printinfo;
else
  printinfo = 0;
end;

if (isfield(params,'errorinfo')),
  errorinfo = params.errorinfo;
else
  errorinfo = 0;
end;

% mutual incoherence limit %

if (isfield(params,'muthresh')),
  muthresh = params.muthresh;
else
  muthresh = 0.99;
end;

if (muthresh < 0),
  error('invalid muthresh value, must be non-negative'); %#ok
end;

% determine class information %

if(size(labels, 2) ~= size(X, 2)),
  error('numbers of data samples and labels differ'); %#ok
end;
classes = unique(labels);
numclasses = length(classes);

% determine dictionary size %

if (isfield(params,'initdict')),
  if (any(size(params.initdict)==1) && all(iswhole(params.initdict(:)))),
    dictsize = ceil(length(params.initdict) / numClasses);
	if (dictsize * numClasses ~= length(params.initdict)),
	  error('number of dictionary atoms not a multiple of number of classes'); %#ok
	end;
  else
    dictsize = size(params.initdict,2);
  end;
end;

if (isfield(params,'dictsize')),    % this superceedes the size determined by initdict
  dictsize = params.dictsize;
end;

if (size(X, 2) < dictsize),
  warning('Number of training signals is smaller than number of atoms to train'); %#ok
end;

% initialize the dictionary %

if (isfield(params,'initdict') && ~any(isnan(params.initdict(:)))),
  if (any(size(params.initdict)==1) && all(iswhole(params.initdict(:)))),
	for iterclass = 1:numclasses,
	  Xclass = X(:, labels == classes(iterclass));
	  D(:, ((iterclass - 1) * dictsize + 1):(iterclass * dictsize)) = Xclass(:,params.initdict(1:dictsize));
	end;
  else
    if (size(params.initdict,1)~=size(X,1) || size(params.initdict,2)<dictsize),
      error('Invalid initial dictionary'); %#ok
    end;
    D = params.initdict(:,1:(dictsize * numclasses));
  end;
else
  for iterclass = 1:numclasses,
    Xclass = X(:, labels == classes(iterclass));
	X_ids = find(colnorms_squared(Xclass) > 1e-6);   % ensure no zero X elements are chosen
	perm = randperm(length(X_ids));
    D(:, ((iterclass - 1) * dictsize + 1):(iterclass * dictsize)) = Xclass(:,X_ids(perm(1:dictsize)));
  end;
end;

% normalize the dictionary %

D = normcols(D);

% precalculate useful quantities %

[signalsize numsamples] = size(X); %#ok
if ((blockratio > 0) && (blockratio < 1)),
  blocksamples = ceil(blockratio * numsamples);
else
  blocksamples = numsamples;
end;
replaced_atoms = zeros(1, dictsize); 
unused_sigs = 1:numsamples;

%%%%%%%%%%%%%%%%%  main loop  %%%%%%%%%%%%%%%%%

for iter = 1:iternum
	
  if (printinfo),
	  disp(sprintf('Iteration %d / %d begins\n', iter, iternum)); %#ok
  end;
 
  %%%%%%%%  perform dictionary optimization  %%%%%%%% 
  
  %%%%%  block creation %%%%%
  
  if (printinfo),
	  disp(sprintf('Creating blocks\n')); %#ok
  end;
  
  %TODO: Write multi-blocking
%   if ((blockratio > 0) && (blockratio < 1)),
% 	  inds = randperm(numsamples);
% 	  blockinds = inds(1:blocksamples);
% 	  Xblock = X(:, blockinds);
% 	  Lblock = labels(:, blockinds);
%   else
	  Xblock = X;
	  Lblock = labels;
%   end;
  
  %%%%%  sparse coding  %%%%%
  
  if (printinfo),
	  disp(sprintf('Sparse coding\n')); %#ok
  end;

  for iterclass = 1:numclasses,
	Dclass = D(:, ((iterclass - 1) * dictsize + 1):(iterclass * dictsize));
	A(:, Lblock == iterclass) = l1qp_featuresign_mex(Xblock(:, Lblock == iterclass), Dclass, codinglambda);
  end;
	
  %%%%%  dictionary update  %%%%%
  
  if (printinfo),
	  disp(sprintf('Updating dictionary\n')); %#ok
  end;
  
%   D = l2ls_learn_basis_dual(X, A, 1, D);
%   D = l2ls_learn_basis_multi_atom(Xblock, Lblock, A, D, incoherencelambda, 1, iternum2);
  D = l2ls_learn_basis_multi(Xblock, Lblock, A, D, incoherencelambda, 1, iternum2);
  
  %%%%%  compute error  %%%%%
  
%   if (errorinfo)
%     lassoObj = compute_err(D, A, Xblock, codinglambda);
%   end;
   
  %%%%%  clear dictionary  %%%%%
  
  if (printinfo),
	  disp(sprintf('Clearing dictionary\n')); %#ok
  end;
  
  % TODO: Write cleardict_multi
%   D = cleardict_multi(D, A, Xblock, Lblock, codinglambda, muthresh, unused_sigs, replaced_atoms);
%   [D unused_sigs replaced_atoms] = cleardict_zeronorm(D, X, labels, unused_sigs, replaced_atoms);
  
  %%%%%  print info  %%%%%
  
  if (printinfo),
% 	  if (errorinfo),
% 		  disp(sprintf('eigObj = %g, lassoObj = %g\n', eigObj, lassoObj)); %#ok
% 	  end;
	  disp(sprintf('Iteration %d / %d complete\n', iter, iternum)); %#ok
  end;
  
end;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             compute_err              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% function err = compute_err(D, A, X, codinglambda)
%   
% err = norm(X - D * A, 'fro') ^ 2 + codinglambda * sum(sum(abs(A)));
% 
% % TODO: Use block version
% % err = 0;
% % blocks = [1:3000:size(X, 2) size(X, 2) + 1];
% % for i = 1:length(blocks) - 1
% %   err = err...
% % 	  + alpha * sum((X(:, blocks(i):blocks(i + 1) - 1) - D * A(:, blocks(i):blocks(i + 1) - 1)) .^ 2)...
% % 	  + (1 - alpha) * sum((Y(:, blocks(i):blocks(i + 1) - 1) - Phi * D * A(:, blocks(i):blocks(i + 1) - 1)) .^ 2)...
% % 	  + codinglambda * sum(sum(abs(A(:, blocks(i):blocks(i + 1) - 1))));
% % end
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           cleardict                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [D unused_sigs replaced_atoms] = cleardict_zeronorm(D, X, L, unused_sigs, replaced_atoms)

classes = unique(L);
numclasses = length(classes);
dictsize = size(D,2) / numclasses;
Xunused = X(:, unused_sigs);
Lunused = L(:, unused_sigs);

for iterclass = 1:numclasses,
	Dclass = D(:, ((iterclass - 1) * dictsize + 1):(iterclass * dictsize));
	classIds = find(Lunused == classes(iterclass));
	Xclass = Xunused(:, classIds);		
	nonzeroIds = find(colnorms_squared(Xclass) > 1e-6);
	nonzeroPerm = randperm(length(nonzeroIds));
	iterIds = 1;
	for iteratom = 1:min(dictsize, length(nonzeroIds)),
		if (norm(Dclass(:, iteratom)) ^ 2 < 1e-6),
			Dclass(:, iteratom) = Xclass(:, nonzeroPerm(iterIds)) / norm(Xclass(:, nonzeroPerm(iterIds)));
			iterIds = iterIds + 1;
			unused_sigs(unused_sigs == classIds(nonzeroPerm(iterIds))) = [];
		end;
	end;
	D(:, ((iterclass - 1) * dictsize + 1):(iterclass * dictsize)) = Dclass;
end;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           cleardict                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% function D = cleardict_multi(D, A, X, L, codinglambda, muthresh, unused_sigs, replaced_atoms)
% 
% use_thresh = 4;  % at least this number of samples must use the atom to be kept
% 
% dictsize = size(A,1);
% numClasses = length(D);
% 
% % compute error in blocks to conserve memory
% for iterClass = 1:numClasses,
% err = zeros(1, size(X, 2));
% blocks = [1:3000:size(X, 2) size(X, 2) + 1];
% for i = 1:length(blocks) - 1
%   err(blocks(i):blocks(i + 1) - 1) = ...
% 	  sum((X(:, blocks(i):blocks(i + 1) - 1) - D * A(:, blocks(i):blocks(i + 1) - 1)) .^ 2)...
% 	  + codinglambda * sum(sum(abs(A(:, blocks(i):blocks(i + 1) - 1))));
% end
% 
% usecount = sum(abs(A) > 1e-7, 2);
% 
% for j = 1:dictsize
%   
%   % compute G(:,j)
%   Gj = D' * D(:, j);
%   Gj(j) = 0;
%   
%   % replace atom
%   if ((max(Gj .^ 2) > muthresh^2 || usecount(j) < use_thresh) && ~replaced_atoms(j))
%     [y, i] = max(err(unused_sigs)); %#ok
%     D(:, j) = X(:, unused_sigs(i)) / norm(X(:, unused_sigs(i)));
%     unused_sigs = unused_sigs([1:i - 1, i + 1:end]);
%   end
% end
% 
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                misc                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Y = colnorms_squared(X)

% compute in blocks to conserve memory
Y = zeros(1, size(X, 2));
blocksize = 2000;
for i = 1:blocksize:size(X, 2)
  blockids = i:min(i + blocksize - 1, size(X, 2));
  Y(blockids) = sum(X(:, blockids) .^ 2);
end

end

