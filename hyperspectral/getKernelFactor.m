function Y = getKernelFactor(wavelengths, tau)

kernelMat = kernel_gram(wavelengths, [], 'h', tau);
[U S] = eig(kernelMat);
Y = U * sqrt(S);
