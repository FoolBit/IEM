%%
clc; clear; close all;

%% load data
addpath utils/
load("../data/processed_data/SVMdatanew_cue_7.mat");

%% params
n_sub = size(acc_CV_SVM_all,3);
n_tpt = size(acc_CV_SVM_all, 1);

tpts = (1:n_tpt)*10;
base_start = 1;
base_end = 50;

%% smooth data
smoothed_acc_CV_all_trial = smoothdata(acc_CV_SVM_all, 1,'gaussian',16);
% smoothed_acc_CV_all_trial = acc_CV_SVM_all;

%% compute baseling, mean and std
baseline = mean(mean(smoothed_acc_CV_all_trial(base_start:base_end,:,:),3),1);
m = squeeze(mean(smoothed_acc_CV_all_trial, 3));
sd = squeeze(std(smoothed_acc_CV_all_trial, 0, 3)/sqrt(n_sub));

%% t-test
H = nan(n_tpt, 2);
centered_smoothed_acc_CV_all = smoothed_acc_CV_all_trial;
for ii = 1:2
    centered_smoothed_acc_CV_all(:,ii,:) = smoothed_acc_CV_all_trial(:,ii,:)-baseline(ii);
    for tt = 1:n_tpt
        H(tt, ii) = ttest(centered_smoothed_acc_CV_all(tt, ii, :),0,'alpha',0.05,'Tail','Right');
    end
end

%% plot!
% init
f = figure;
% subplot(1,2,2);
hold off
h = gca;

% params
colors = [58,95,205;178,35,35]/255;
sa = .5;
heights = [0.160, 0.161];
lines = [];
start_time = [500, 1000];
end_time = [1000, 1500];

% plot start!
for i = 1:2
    lines = [lines,plot(h, tpts,m(:,i), 'Color', changeSaturation(colors(i,:),sa),'LineWidth',2)];
    plotPatch(h, tpts, m(:,i)', sd(:,i)', changeSaturation(colors(i,:),sa));
    plotSig(f, tpts, H(:,i), heights(i), changeSaturation(colors(i,:),sa), 5);
    plot([1, n_tpt*10], [baseline(i), baseline(i)], 'Color',changeSaturation(colors(i,:),sa),'LineStyle','--');
end

% change ylim
tmp_ylim = get(h,'YLim');
miny = tmp_ylim(1)-0.001;
maxy = tmp_ylim(2)+0.001;
set(h,'YLim',[miny, maxy]);
clear tmp_ylim
% 
% plot time info
for i = 1:2
   plot([start_time(i), start_time(i)], [miny, maxy],  'Color','k','LineStyle','--','LineWidth',1.5); 
   plot([end_time(i), end_time(i)], [miny, maxy],  'Color','k','LineStyle','--','LineWidth',1.5); 
end
% 
% info
legend(lines,["刺激1","刺激2"],'Box','off');
xlabel("时间（ms）");
ylabel("正确率");
title("提示2");