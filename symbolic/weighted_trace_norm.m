syms d11 d12 d13 d21 d22 d23 b11 b12 b13 b21 b22 b23
D = [d11 d12 d13;d21 d22 d23]
B = [b11 b12 b13;b21 b22 b23]
I = eye(3)
syms l1 l2
L = [l1 l2]
L*D
trace(((L*B).'*(L*B)-I).'*((L*B).'*(L*B)-I))
trace(((L*B).'*(L*B)-I).'*((L*B).'*(L*B)-I))+trace(((L*D).'*(L*B)).'*((L*D).'*(L*B)))+trace(((L*B).'*(L*D)).'*((L*B).'*(L*D)))
gk1 = trace(((L*B).'*(L*B)-I).'*((L*B).'*(L*B)-I))+trace(((L*D).'*(L*B)).'*((L*D).'*(L*B)))+trace(((L*B).'*(L*D)).'*((L*B).'*(L*D)));
Dl = [D B]
Il = eye(6)
gk2 = trace(((L*Dl).'*(L*Dl)-Il).'*((L*Dl).'*(L*Dl)-Il));
simplify(gk1-gk2)
