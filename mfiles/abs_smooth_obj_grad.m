function [f df] = abs_smooth_obj_grad(x, r)

inds_small = x < - r;
inds_large = x > r;
inds_smooth = ~(inds_small | inds_large);

f = zeros(size(x));
f(inds_smooth) = x(inds_smooth) .^ 2 / 2 / r + r / 2;
f(inds_large) = x(inds_large);
f(inds_small) = - x(inds_small);

df = zeros(size(x));
df(inds_smooth) = x(inds_smooth) / r;
df(inds_large) = 1;
df(inds_small) = -1;
