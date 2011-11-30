function noiseVector = addBernoulliSignSupportNoise(numEl, valRange, p)

support = binornd(1, p, [numEl 1]);
numSupport = sum(support);
signs = binornd(1, 0.5, [numSupport 1]);
signs(~signs) = - 1;
vals = rand([numSupport 1]) * valRange;
noiseVector = zeros(numEl, 1);
noiseVector(logical(support)) = signs .* vals;
