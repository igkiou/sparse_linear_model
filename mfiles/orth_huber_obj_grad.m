function [obj deriv] = orth_huber_obj_grad(X, data, labels, weights, bias, orthlambda, M, N)

[obj_orth deriv_orth] = orth_obj_grad_mex(X, M, N);
[obj_huber deriv_huber] = huber_obj_grad_mex(X, data, labels, weights, bias);

obj = orthlambda * obj_orth + obj_huber;
deriv = orthlambda * deriv_orth + deriv_huber;
