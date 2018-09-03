function c = init_c_dictionary(A,b,x)
% This function initializes c
% It is used in bp_ppa
%

Ax = A*x;
residual = b-Ax;
Du= A'*residual;
c=residual/max(max(abs(Du)));

return;
