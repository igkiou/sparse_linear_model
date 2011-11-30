G = transpose(Psi)*transpose(Phi)*Phi*Psi;
K = G - eye(size(G));
D = Phi*Psi;

% dLdPsi=4*transpose(Phi)*Phi*Psi*transpose(Psi)*transpose(Phi)*Phi*Psi-4*transpose(Phi)*Phi*Psi;
% dLdPhi = transpose(4*Psi*transpose(Psi)*transpose(Phi)*Phi*Psi*transpose(Psi)*transpose(Phi)-4*Psi*transpose(Psi)*transpose(Phi));

% dLdPsi1=4*transpose(Phi)*Phi*Psi*K;
% dLdPhi1=4*Phi*Psi*K*transpose(Psi);

dLdPsi2=4*transpose(Phi)*D*K;
dLdPhi2=4*D*K*transpose(Psi);


d2 = (X^2*A+X*A*X+A*X^2)*2*Psi;
