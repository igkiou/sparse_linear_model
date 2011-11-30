%%
numFeatures = 100;
numSamples1 = 20;
numSamples2 = 50;
param1 = 0.1;
param2 = 3;

%%
X1 = randn(numFeatures, numSamples1);
X2 = randn(numFeatures, numSamples2);
W = randn(numFeatures, numFeatures);
W = W' * W;

%%
K = kernel_gram(X1, [], 'l');
G = kernel_gram_mex(X1, [], 'l');
fprintf('linear onearg %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, X2, 'l');
G = kernel_gram_mex(X1, X2, 'l');
fprintf('linear twoarg %g\n', norm(K - G, 'fro'));

%%
K = kernel_gram(X1, [], 'g');
G = kernel_gram_mex(X1, [], 'g');
fprintf('gaussian onearg %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, X2, 'g');
G = kernel_gram_mex(X1, X2, 'g');
fprintf('gaussian twoarg %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, [], 'g', param1);
G = kernel_gram_mex(X1, [], 'g', param1);
fprintf('gaussian onearg param %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, X2, 'g', param1);
G = kernel_gram_mex(X1, X2, 'g', param1);
fprintf('gaussian twoarg param %g\n', norm(K - G, 'fro'));

%%
K = kernel_gram(X1, [], 'p');
G = kernel_gram_mex(X1, [], 'p');
fprintf('poly onearg %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, X2, 'p');
G = kernel_gram_mex(X1, X2, 'p');
fprintf('poly twoarg %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, [], 'p', param1, param2);
G = kernel_gram_mex(X1, [], 'p', param1, param2);
fprintf('poly onearg param %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, X2, 'p', param1, param2);
G = kernel_gram_mex(X1, X2, 'p', param1, param2);
fprintf('poly twoarg param %g\n', norm(K - G, 'fro'));

%%
K = kernel_gram(X1, [], 'h');
G = kernel_gram_mex(X1, [], 'h');
fprintf('sobolev onearg %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, X2, 'h');
G = kernel_gram_mex(X1, X2, 'h');
fprintf('sobolev twoarg %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, [], 'h', param1);
G = kernel_gram_mex(X1, [], 'h', param1);
fprintf('sobolev onearg param %g\n', norm(K - G, 'fro'));

K = kernel_gram(X1, X2, 'h', param1);
G = kernel_gram_mex(X1, X2, 'h', param1);
fprintf('sobolev twoarg param %g\n', norm(K - G, 'fro'));

%%
D = l2_distance(X1, []);
F = l2_distance_mex(X1, []);
fprintf('l2dist onearg %g\n', norm(D - F, 'fro'));

D = l2_distance(X1, X2);
F = l2_distance_mex(X1, X2);
fprintf('l2dist twoarg %g\n', norm(D - F, 'fro'));

D = l1_distance(X1, []);
F = l1_distance_mex(X1, []);
fprintf('l1dist onearg %g\n', norm(D - F, 'fro'));

D = l1_distance(X1, X2);
F = l1_distance_mex(X1, X2);
fprintf('l1dist twoarg %g\n', norm(D - F, 'fro'));

KXX = X1' * X1;
D = kernel_distance(KXX);
F = kernel_distance_mex(KXX);
fprintf('kerneldist %g\n', norm(D - F, 'fro'));

%%
[P V] = l2_sphere_projection(X1);
[Q W] = l2_sphere_projection_mex(X1);
fprintf('l2sphere matdiff %g\n', norm(P - Q, 'fro'));
fprintf('l2sphere normdiff %g\n', norm(V - W, 'fro'));

[P V] = l2_sphere_projection(X1, param1);
[Q W] = l2_sphere_projection_mex(X1, param1);
fprintf('l2sphere param matdiff %g\n', norm(P - Q, 'fro'));
fprintf('l2sphere param normdiff %g\n', norm(V - W, 'fro'));

[P V] = l2_ball_projection(X1);
[Q W] = l2_ball_projection_mex(X1);
fprintf('l2ball matdiff %g\n', norm(P - Q, 'fro'));
fprintf('l2ball normdiff %g\n', norm(V - W, 'fro'));

[P V] = l2_ball_projection(X1, param1);
[Q W] = l2_ball_projection_mex(X1, param1);
fprintf('l2ball param matdiff %g\n', norm(P - Q, 'fro'));
fprintf('l2ball param normdiff %g\n', norm(V - W, 'fro'));

KXX = X1' * X1;
[P V] = kernel_sphere_projection(KXX);
[Q W] = kernel_sphere_projection_mex(KXX);
fprintf('kernelsphere matdiff %g\n', norm(P - Q, 'fro'));
fprintf('kernelsphere normdiff %g\n', norm(V - W, 'fro'));

[P V] = kernel_sphere_projection(KXX, param1);
[Q W] = kernel_sphere_projection_mex(KXX, param1);
fprintf('kernelsphere param matdiff %g\n', norm(P - Q, 'fro'));
fprintf('kernelsphere param normdiff %g\n', norm(V - W, 'fro'));

[P V] = kernel_ball_projection(KXX);
[Q W] = kernel_ball_projection_mex(KXX);
fprintf('kernelball matdiff %g\n', norm(P - Q, 'fro'));
fprintf('kernelball normdiff %g\n', norm(V - W, 'fro'));

[P V] = kernel_ball_projection(KXX, param1);
[Q W] = kernel_ball_projection_mex(KXX, param1);
fprintf('kernelball param matdiff %g\n', norm(P - Q, 'fro'));
fprintf('kernelball param normdiff %g\n', norm(V - W, 'fro'));
