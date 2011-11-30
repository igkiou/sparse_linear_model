params = SetDefaultParams(params);

[l, u] = ComputeDistanceExtremes(X, 5, 95, eye(size(X,2)));
k = length(unique(y));

num_constraints = params.const_factor * (k * (k-1));
C = GetConstraints(y, num_constraints, l, u);

% A = feval(metric_learn_alg, C, X, A0, params);

tol = params.thresh;
gamma = params.gamma;
max_iters = params.max_iters;

% check to make sure that no 2 constrained vectors are identical
valid = ones(size(C,1),1);
for (i=1:size(C,1)),
   i1 = C(i,1);
   i2 = C(i,2);
   v = X(i1,:)' - X(i2,:)'; 
   if (norm(v) < 10e-10),
      valid(i) = 0;
   end
end
C = C(valid>0,:);

i = 1;
iter = 0;
c = size(C, 1);
lambda = zeros(c,1);
bhat = C(:,4);
lambdaold = zeros(c,1);
conv = Inf;
A = A_0;

while (true),
    i1 = C(i,1);
    i2 = C(i,2);
    v = X(i1,:)' - X(i2,:)';
	Av = A * v;
	wtw = v'*Av;
%     wtw = v'*A*v;

    if (abs(bhat(i)) < 10e-10),
        error('bhat should never be 0!');
    end
    if (inf == gamma),
        gamma_proj = 1;
    else
        gamma_proj = gamma/(gamma+1);
    end
    
    if C(i,3) == 1
        alpha = min(lambda(i),gamma_proj*(1/(wtw) - 1/bhat(i)));
        lambda(i) = lambda(i) - alpha;
        beta = alpha/(1 - alpha*wtw);        
        bhat(i) = inv((1 / bhat(i)) + (alpha / gamma));        
    elseif C(i,3) == -1
        alpha = min(lambda(i),gamma_proj*(1/bhat(i) - 1/(wtw)));
        lambda(i) = lambda(i) - alpha;
        beta = -1*alpha/(1 + alpha*wtw); 
        bhat(i) = inv((1 / bhat(i)) - (alpha / gamma));
	end

    A = A + beta*(Av*Av');
%     A = A + (beta*A*v*v'*A);
% 	disp(sprintf('beta = %f', beta));

    if i == c
		normsum = norm(lambda) + norm(lambdaold);
		conv = norm(lambdaold - lambda, 1) / normsum;
        if (normsum == 0)
            break;
        else
            if (conv < tol || iter > max_iters),
                break;
            end
        end
        lambdaold = lambda;
    end
    i = mod(i,c) + 1;

    iter = iter + 1;
    if (mod(iter, 5000) == 0),       
       disp(sprintf('itml iter: %d, conv = %f', iter, conv));
    end
end
disp(sprintf('itml converged to tol: %f, iter: %d', conv, iter));
