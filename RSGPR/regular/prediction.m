function [testMean, testVariance] = prediction(testX, hyperParam)
Kmm = hyperParam.Kmm;
inducingX = hyperParam.inducingX;
lambda = hyperParam.lambda;
beta = hyperParam.beta;
mu = hyperParam.mu;
SigmaInv = hyperParam.SigmaInv;
sigma2 = hyperParam.sigma2;

[U, ~, ~] = construct_kernel(testX, inducingX, lambda, beta);
testMean = (U/Kmm)*mu;
testVariance = lambda - diag((U/Kmm)*U') + diag((U/SigmaInv)*U');
end
