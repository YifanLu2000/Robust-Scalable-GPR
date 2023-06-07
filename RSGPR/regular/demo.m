clc,clear all;
close all;
warning off;
%% generate data
outlierRT = 0.5;
N_inlier = 100; sigma_GT = 0.1; 
N_outlier = ceil(outlierRT/(1-outlierRT)*N_inlier);
train_x = (rand(N_inlier,1)-0.5)*5;
train_y = 0.3 + 0.4*train_x + 0.5*sin(2.7*train_x) + 1.1./(1+train_x.^2) + sigma_GT*randn(N_inlier,1);
outlier_x = (rand(N_outlier,1)-0.5)*8;
outlier_y = 10*(rand(N_outlier,1)-0.5);
train_x = [train_x;outlier_x];
train_y = [train_y;outlier_y];
iter = 150;
xs = linspace(-6, 6, 1000)';                  
figure;
plot(train_x(1:N_inlier),train_y(1:N_inlier),'k.'); hold on;
plot(train_x(1+N_inlier:end),train_y(1+N_inlier:end),'r.'); hold off;
linewidth = 1.5;
marksize = 2;
color_red = [200,36,35]/255;
color_darkgray = [89,89,89]/255;
%% Ours no SVI
param.M = 15;
param.beta = 1;
param.lambda = 1;
param.maxIter = 150;
param.outlierA = max(train_y)-min(train_y);
param.minP = 1e-8;
param.Ba = 10;
param.Bb = 10;
tic
hyperParam = inference(train_x,train_y,param);
toc
[fmu_RSGPR,fs2_RSGPR] = prediction(xs, hyperParam);

f_RSGPR_ys = [fmu_RSGPR+2*sqrt(fs2_RSGPR+hyperParam.sigma2); flipdim(fmu_RSGPR-2*sqrt(fs2_RSGPR+hyperParam.sigma2),1)];
figure;
fill([xs; flipdim(xs,1)], f_RSGPR_ys, [7 7 7]/8,'linewidth',linewidth,'edgecolor',color_darkgray)
hold on; 
plot(xs, fmu_RSGPR,'-','linewidth',linewidth,'color',color_darkgray); 
plot(train_x(1:N_inlier),train_y(1:N_inlier),'k.'); hold on;
plot(train_x(1+N_inlier:end),train_y(1+N_inlier:end),'r.'); 
plot(hyperParam.inducingX,-4.5,'^','color',color_darkgray);
plot(hyperParam.inducingXInit,4.5,'v','color',color_darkgray);
axis([-5 5 -5 5])
xticks([])
yticks([])
set(gca,'linewidth',1.5)
drawnow;




