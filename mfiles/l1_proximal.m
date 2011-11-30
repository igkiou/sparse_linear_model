function [Xr normValue] = l1_proximal(X, tau)

threshVec = max(abs(X) - tau, 0);
Xr = sign(X) .* threshVec;
if (nargout >= 2),
	normValue = sum(threshVec);
end;
