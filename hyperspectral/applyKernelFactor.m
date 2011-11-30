function cube = applyKernelFactor(operator, kernelFactor)

[M N O] = size(operator);
cube = reshape(reshape(operator, [M * N, O]) * kernelFactor', [M, N, O]);
