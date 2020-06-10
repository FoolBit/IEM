clc;clear;close all;

%% add path
addpath('../utils/')

%% load data
load('accData_wmnew_cue.mat');

%% params
dt = 10;
n_tpt = 450;
tpts = (1:n_tpt)*dt;

colors = [58,95,205;178,35,35]/255;

start_time = [500, 2000,3500];
end_time = [1000, 2500,4000];

%%
acc_neuron = acc_neuron_cue(:,:,1);
acc_syn = acc_syn_cue(:,:,1);
%% plot
%f = figure;
subplot(2,2,1);
h = gca;
hold on;

plot(tpts, acc_neuron(:,1),'Color',changeSaturation(colors(1,:),.5),'LineWidth',2.2);
plot(tpts, acc_neuron(:,2),'Color',changeSaturation(colors(2,:),.5),'LineWidth',2.2);

% change ylim
tmp_ylim = get(h,'YLim');
miny = tmp_ylim(1)-0.01;
maxy = tmp_ylim(2)+0.01;
set(h,'YLim',[miny, maxy]);
clear tmp_ylim

% plot time info
for i = 1:3
   plot([start_time(i), start_time(i)], [miny, maxy],  'Color','k','LineStyle','--','LineWidth',1.2); 
   plot([end_time(i), end_time(i)], [miny, maxy],  'Color','k','LineStyle','--','LineWidth',1.2); 
end

xlim([0, 4500])
xlabel('时间(ms)');
ylabel('正确率');
legend(["刺激1","刺激2"],'Box','off');
title("(a)");

%% plot
subplot(2,2,2);
h = gca;
hold on;

plot(tpts, acc_syn(:,1),'Color',changeSaturation(colors(1,:),.5),'LineWidth',2.2);
plot(tpts, acc_syn(:,2),'Color',changeSaturation(colors(2,:),.5),'LineWidth',2.2);

% change ylim
tmp_ylim = get(h,'YLim');
miny = tmp_ylim(1)-0.01;
maxy = tmp_ylim(2)+0.01;
set(h,'YLim',[miny, maxy]);
clear tmp_ylim

% plot time info
for i = 1:3
   plot([start_time(i), start_time(i)], [miny, maxy],  'Color','k','LineStyle','--','LineWidth',1.2); 
   plot([end_time(i), end_time(i)], [miny, maxy],  'Color','k','LineStyle','--','LineWidth',1.2); 
end

xlim([0, 4500])
xlabel('时间(ms)');
ylabel('正确率');
legend(["刺激1","刺激2"],'Box','off');
title("(b)");