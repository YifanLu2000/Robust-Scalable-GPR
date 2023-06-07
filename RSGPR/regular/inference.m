function hyperparam = inference(trainX, trainY, initHyperParam, useInducing, useSVI)
% FUNCTIONNAME - Robust and scalable inference of hyperparameters for Gaussian process regression
%
%   hyperparam = inference(trainX, trainY, initHyperParam, useInducing, useSVI)
%   Detailed description
%   of the function and its functionality. It may span multiple lines.
%
%   Inputs:
%       INPUT1 - Description of input 1
%       INPUT2 - Description of input 2
%
%   Outputs:
%       OUTPUT1 - Description of output 1
%       OUTPUT2 - Description of output 2
%
%   Examples:
%       Example usage or sample calls to the function
%
%   References:
%       References or citations if applicable
%
%   Notes:
%       Additional notes or important information
%
%   See also: OTHERFUNCTION1, OTHERFUNCTION2

% Author: Yifan Lu
% Email: lyf048@whu.edu.cn
% Date: 2023/6/7

% get data num
N = size(trainX,1);

% initialization
f = zeros(N,1);  % posterior GP mean
sigma2 = sum((trainY-f).^2)/N;  % noise
maxIter = initHyperParam.maxIter;  % max iteration of inference
Ba = initHyperParam.Ba; % hyperparam of Beta distribution
Bb = initHyperParam.Bb; 
Gamma = exp(psi(Ba)-psi(Ba+Bb));  % expection of gamma
minusGamma = exp(psi(Bb+N)-psi(Ba+Bb+N));  % expection of 1-gamma
variancef = zeros(N,1);  % variance of the posterior GP
momentum = 0.9;  % the momentum of gradient descent
gradLambda = 0; gradBeta = 0;  % initial gradient for lambda and beta
outlierA = initHyperParam.outlierA;  % outlier uniform distribution
minP = initHyperParam.minP;  % prevent numerical problem
beta = initHyperParam.beta; lambda = initHyperParam.lambda;
iter = 1;
% construct SE kernel
% construct inducing variables kernel matrix K
M = initHyperParam.M;
% random choose M inducing variables from trainX as initialization
uniqueX = unique(trainX, 'rows');
idx = randperm(size(uniqueX,1)); 
idx = idx(1:min(M,size(uniqueX,1))); % in case M > length(uniqueX)
inducingX = uniqueX(idx,:);
inducingX = linspace(-0.5,0.5,M)';
hyperparam.inducingXInit = inducingX;

[Kmm, dist_mm, dist_mm2] = construct_kernel(inducingX, inducingX, lambda, beta);
[Knm, dist_nm, dist_nm2] = construct_kernel(trainX, inducingX, lambda, beta);
% start main loop of inference
while iter < maxIter
    stepSize = 1/(iter + 1e2);  % step size for gradient descent
    % update q_3(Z)
    P = get_P(trainY, f, sigma2, Gamma, minusGamma, variancef, outlierA);
    P = max(P, minP);  % prevent numerical problem
    Sp  =sum(P);
    % update q_2(\gamma)
    Gamma = exp(psi(Ba+Sp)-psi(Ba+Bb+N));
    minusGamma = exp(psi(Bb+N-Sp)-psi(Ba+Bb+N));
    % update q_1(f,f_m)
    commonTerm = (Kmm+Knm'*(Knm.*P)/sigma2)\(Knm');
    fm = Kmm*commonTerm*(trainY.*P) / sigma2;
    f = (Knm/Kmm)*fm;
    variancef = (lambda+diag(Knm*commonTerm)-diag((Knm/Kmm)*Knm'));
    variancef = max(variancef,0);
    % update q_4(sigma^2, lambda, beta, Xm)
    sigma2 = sum((trainY - f).^2.*P+P.*variancef)/Sp;  % update sigma2
    K_y = sigma2*diag(1./P)+(Knm/Kmm)*Knm';
    C = K_y\trainY;
    % calculate the gradient for lambda
    dKy_dlambda = (Knm/Kmm)*Knm'/lambda;
    gradLambda = 0.5*momentum*(trace(C*C'*dKy_dlambda)-trace(K_y\dKy_dlambda)-Sp/sigma2+trace(diag(P)*dKy_dlambda)/sigma2)+(1-momentum)*gradLambda;
    gradLambda = max(min(gradLambda,1),-1);  % gradient clip
%     lambda = lambda + stepSize * gradLambda;
    lambda = max(lambda + stepSize * gradLambda,0.1);
    % calculate the gradient for lambda
    dKy_dbeta = -2*((Knm.*dist_nm2)/Kmm)*Knm' + (Knm/Kmm)*(Kmm.*dist_mm2)*(Knm/Kmm)';
    gradBeta = 0.5*momentum*(trace(C*C'*dKy_dbeta)-trace(K_y\dKy_dbeta)+trace(diag(P)*dKy_dbeta)/(sigma2))+(1-momentum)*gradBeta;
    gradBeta = max(min(gradBeta,1),-1);  % gradient clip
%     beta = beta + stepSize * gradBeta;
    beta = max(beta + stepSize * gradBeta,0.1);
    % update Xm
    gradXm = zeros(M,1);
    for gradIter = 1:M
        const_distnm = zeros(N,M);
        const_distnm(:,gradIter) = -dist_nm(:,gradIter);
        dKnmdXm = -2*beta*(Knm.*const_distnm);
        const_distmm = zeros(M,M);
        const_distmm(:,gradIter) = -dist_mm(:,gradIter);
        const_distmm(gradIter,:) = -dist_mm(gradIter,:);
        dKmmdXm = -2*beta*(Kmm.*const_distmm);
        dKydXm = 2*(dKnmdXm/Kmm)*Knm'-(Knm/Kmm)*dKmmdXm*(Knm/Kmm)';
        gradXm(gradIter) = 0.5*(trace((C*C'*dKydXm)-K_y\dKydXm)+trace(diag(P)*dKydXm)/sigma2); 
        gradXm(gradIter) = max(min(gradXm(gradIter),5),-5);
    end
    inducingX = inducingX + stepSize * gradXm;
    inducingX = min(max(inducingX,min(trainX)),max(trainX)); % limit inducing variable within trainX
    % update kernel
    [Kmm, dist_mm, dist_mm2] = construct_kernel(inducingX, inducingX, lambda, beta);
    [Knm, dist_nm, dist_nm2] = construct_kernel(trainX, inducingX, lambda, beta);
    iter = iter + 1;
end
% output hyperparamter of GP
SigmaInv = Kmm+Knm'*diag(P)*Knm/sigma2;
mu = (Kmm/(Kmm+Knm'*diag(P)*Knm/sigma2))*Knm'*diag(P)*trainY/sigma2;
hyperparam.lambda = lambda;
hyperparam.beta = beta;
hyperparam.sigma2 = sigma2;
hyperparam.P = P;
hyperparam.inducingX = inducingX;
hyperparam.mu = mu;
hyperparam.SigmaInv = SigmaInv;
hyperparam.f = f;
hyperparam.Kmm = Kmm;
end

function P = get_P(trainY, f, sigma2, Gamma, minusGamma, variancef, outlierA)
    inlier_term = Gamma * exp(-(trainY-f).^2/(2*sigma2)) .* exp(-variancef./(2*sigma2)) ./ sqrt(2*pi*sigma2);
    P = inlier_term ./ (minusGamma/outlierA + inlier_term);
end

































