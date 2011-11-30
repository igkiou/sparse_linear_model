tic; C2 = cudaSampleLegacy(0, 0, 1, 0, A, B, C); toc
tic; C3 = mklSample(0, 0, 1, 0, A, B, C); toc
tic; C4 = cudaSampleCula(0, 0, 1, 0, A, B, C); toc
tic; C5 = cudaSampleCulaMalloc(0, 0, 1, 0, A, B, C); toc
tic; C6 = mklSampleOpenMP(0, 0, 1, 0, A, B, C); toc
