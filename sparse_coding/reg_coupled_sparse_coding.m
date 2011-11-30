function [B, Phi, S, stat] = reg_coupled_sparse_coding(X, num_bases, num_samples, Sigma, beta, gamma, num_iters,...
													batch_size, kappa, noise_std, initB, fname_save)
%
% Regularized sparse coding
%
% Inputs
%       X           -data samples, column wise
%       num_bases   -number of bases
%		num_samples -number of measurements
%       Sigma       -smoothing matrix for regularization
%       beta        -smoothing regularization
%       gamma       -sparsity regularization
%       num_iters   -number of iterations 
%       batch_size  -batch size
%		kappa		-loss terms balance
%		noise_std	-noise std as proportion of amplitude
%       initB       -initial dictionary
%       fname_save  -file name to save dictionary
%
% Outputs
%       B           -learned dictionary
%		Phi			-learned projection matrix
%       S           -sparse codes
%       stat        -statistics about the training
%
% Written by Jianchao Yang @ IFP UIUC, Sep. 2009.

pars = struct;
pars.patch_size = size(X,1);
pars.num_patches = size(X,2);
pars.num_bases = num_bases;
pars.num_samples = num_samples;
pars.num_trials = num_iters;
pars.beta = beta;
pars.gamma = gamma;
pars.kappa = kappa;
pars.noise_std = noise_std;
pars.VAR_basis = 1; % maximum L2 norm of each dictionary atom

if ~isa(X, 'double'),
    X = cast(X, 'double');
end

if exist('batch_size', 'var') && ~isempty(batch_size)
    pars.batch_size = batch_size; 
else
    pars.batch_size = size(X, 2);
end

if exist('fname_save', 'var') && ~isempty(fname_save)
    pars.filename = fname_save;
else
    pars.filename = sprintf('Results/reg_sc_b%d_%s', num_bases, datestr(now, 30));	
end;

pars

% initialize basis
if ~exist('initB', 'var') || isempty(initB)
    B = rand(pars.patch_size, pars.num_bases)-0.5;
	B = B - repmat(mean(B,1), size(B,1),1);
    B = B*diag(1./sqrt(sum(B.*B)));
else
    disp('Using initial B...');
    B = initB;
end

Phi = random_sensing(B, pars.num_samples);

kI = kappa * eye(size(X, 1));

[L M]=size(B);

t=0;
% statistics variable
stat= [];
stat.fobj_avg = [];
stat.elapsed_time=0;

% optimization loop
while t < pars.num_trials
    t=t+1;
	disp(sprintf('Iter %d.',t));
    start_time= cputime;
    stat.fobj_total=0;  
	
	disp(sprintf('Optimize for Phi.'));
	Phi = learn_sensing(B, pars.num_samples, Phi);
	
	% create equivalent data and dictionary
	Y = Phi * X;
	Y = Y + pars.noise_std * Y .* randn(size(Y, 1), size(Y, 2));
	Xeq = [pars.kappa * X; (1 - pars.kappa) * Y];
	Beq = [kI; (1 - pars.kappa) * Phi] * B;
% 	normfactors = diag(1 ./ sqrt(sum(Deq .^ 2)));
% 	Beq = Beq * normfactors;
	Beq = normcols(Beq);

    % Take a random permutation of the samples
    indperm = randperm(size(X,2));
    
    sparsity = [];
    
	disp(sprintf('Optimize for codes and dictionary.'));
	for batch=1:(size(X,2)/pars.batch_size),
        % This is data to use for this step
        batch_idx = indperm((1:pars.batch_size)+pars.batch_size*(batch-1));
        Xbeq = Xeq(:,batch_idx);

% 		% normalize equivalent dictionary and save norm factors
% 		normfactors = diag(1 ./ sqrt(sum(Deq .^ 2)));
% 		Beqtemp = Beq * normfactors;
		
        % learn coefficients (conjugate gradient)   
		S = L1QP_FeatureSign_Set(Xbeq, Beq, Sigma, pars.beta, pars.gamma);
%         S = L1QP_FeatureSign_Set(Xbeq, Beqtemp, Sigma, pars.beta, pars.gamma);
		
		% renormalize sparse codes
% 		S = normfactors * S;
        
        sparsity(end+1) = length(find(S(:) ~= 0))/length(S(:));
        
        % get objective
        [fobj] = getObjective_RegSc(Xeq, Beq, S, Sigma, pars.beta, pars.gamma);       
        stat.fobj_total = stat.fobj_total + fobj;
        % update basis
        Beq = l2ls_learn_basis_dual(Xbeq, S, pars.VAR_basis);
	end

	PhitPhi = (1 - pars.kappa) ^ 2 * Phi' * Phi;
	cholPhi = chol(pars.kappa ^ 2 * eye(size(PhitPhi)) + PhitPhi);
	B = (cholPhi \ (cholPhi' \ ([pars.kappa * eye(size(Phi, 2)) (1 - pars.kappa)* Phi'] * Beq)));
	B = normcols(B);

	% get statistics
    stat.fobj_avg(t)      = stat.fobj_total / pars.num_patches;
    stat.elapsed_time(t)  = cputime - start_time;
    
    fprintf(['epoch= %d, sparsity = %f, fobj= %f, took %0.2f ' ...
             'seconds\n'], t, mean(sparsity), stat.fobj_avg(t), stat.elapsed_time(t));
         
    % save results
    fprintf('saving results ...\n');
    experiment = [];
    experiment.matfname = sprintf('%s.mat', pars.filename);     
    save(experiment.matfname, 't', 'pars', 'B', 'stat');
    fprintf('saved as %s\n', experiment.matfname);
end

return

%% 

function retval = assert(expr)
retval = true;
if ~expr 
    error('Assertion failed');
    retval = false;
end
return
