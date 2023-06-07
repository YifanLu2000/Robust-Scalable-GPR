function [Knm, dist_nm, dist_nm2] = construct_kernel(Xn, Xm, lambda, beta)
%CONSTRUCT_KERNEL 此处显示有关此函数的摘要
%   此处显示详细说明
N = size(Xn,1); M = size(Xm, 1);
dist_nm = repmat(Xn,[1 1 M])-permute(repmat(Xm,[1 1 N]),[3 2 1]);
dist_nm = squeeze(dist_nm);
dist_nm2=dist_nm.^2;
Knm = lambda*exp(-beta*dist_nm2);
end

