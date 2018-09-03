function [dk,residual_norm,Ax] = compute_Z(A,b,x,y,alpha,tau)
% [dk,residual_norm,Ax] = compute_dk(P,b,x,c,alpha,tau)
%
% Computes the gradient of the function f_{\alpha,c} 
% grad_f is descent direction in the loop in k of the algorithm
% bp_ppa
%

Ax = A*x;
dk = 2 * alpha * y + b - Ax ;

norm_dk = sqrt(sum(dk.^2));
temp = zeros(size(norm_dk));
temp(norm_dk<tau) = 1;
temp(norm_dk>tau) = tau./norm_dk(norm_dk>tau);
%%%%% it may also can be replaced by 
%dk = dk * diag(norm_dk);
ddd = ones(size(A,1),1)*temp;
dk = dk.*ddd;

residual_norm = sqrt(mean( ( Ax(:) - b(:) ).^2 ));

return;

