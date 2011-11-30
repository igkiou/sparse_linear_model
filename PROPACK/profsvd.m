A = randn(1024, 100);
opt.eta = eps;
opt.delta = eps;
for iter = 1:100,
	[U S V] = svd(A);
	[U1 S1 V1] = lansvd(A,100,'L',opt);
	[U2 S2 V2] = lansvd_modified_nocomplex(A,100,'L',opt);
end;
	
