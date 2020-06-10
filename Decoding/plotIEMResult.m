%%
clc; clear; close all;

%% load data
addpath ../utils/
load("../data/processed_data/IEMdata_cue_7.mat");

%% params
n_chan = 6;
n_tpt = size(slope_all_all, 1);
n_sub = size(slope_all_all,3);
tpts = (1:n_tpt)*10;
base_start = 1;
base_end = 50;

colors = [58,95,205;178,35,35]/255;
sa = .5;
start_time = [500,1000];
end_time = [1500,1500];

x = tpts;
y = 1:180;
[xx, yy] = meshgrid(x,y);

%% plot channel response aligned && reconstruct response aligned
cnt = 0;
f = figure;
for ii = 1:2
%     % plot mean
%     subplot(2,2,1);
%     imagesc(squeeze(mean(chan_resp_aligned_all(:,:,:,ii),3)));
%     colorbar;
%     title(sprintf("Aligned channel response for angle %i", ii));
%     set(gca,'TickDir','out','FontSize',14);
%     ylabel('Channel');
%     xlabel('Time point');
%     hold on; 
%     for jj = 1:size(start_time, 2)
%         plot(start_time(jj)*[1 1],[0, 16], ...
%             'Color', changeSaturation(colors(jj,:),sa), 'LineWidth',.75, 'LineStyle','--');
%         plot(end_time(jj)*[1 1],[0, 16], ...
%             'Color', changeSaturation(colors(jj,:),sa), 'LineWidth',.75, 'LineStyle','--');
%     end
    cnt = cnt+1;
    subplot(3,2,cnt);
    imagesc([0,n_tpt*10],[0, 180],squeeze(mean(recons_aligned_all(:,:,:,ii),3)));
    colorbar;
    title(sprintf("重构刺激响应（刺激%i,线索2）", ii));
    set(gca,'TickDir','out','FontSize',10);
    ylabel('信息通道');
    xlabel('时间（ms）');
    axis xy;
    hold on;
    for jj = 1:size(start_time, 2)
        plot(start_time(jj)*[1 1],[175, 0], ...
            'Color','k','LineStyle','--','LineWidth',1); 
        plot(end_time(jj)*[1 1],[175, 0], ...
            'Color','k','LineStyle','--','LineWidth',1); 
    end
    
    % 3D plot
%     subplot(2,2,3);
%     surf(squeeze(mean(chan_resp_aligned_all(:,:,:,ii),3)),'EdgeColor','none','LineStyle','none','FaceLighting','phong');
%     shading interp
%     h=findobj('type','patch');
%     set(h,'linewidth',2)
%     hold on
%     set(gca, 'box','off')
%     set(gca,'color','none')
%     set(gca,'LineWidth',2,'TickDir','out');
%     set(gca,'FontSize',14)
%     view(3)
%     %axis([x(1) x(end) tpts_recon(1) em.time(end) lim]);
%     set(gca,'YTick',[0:45:180])
%     %set(gca,'XTick',[recon_tpt_range(1):500:recon_tpt_range(2)])
%     ylabel('Channel Offset (\circ)');
%     xlabel('Time (ms)');
%     set(get(gca,'xlabel'),'FontSize',14,'FontWeight','bold')
%     set(get(gca,'ylabel'),'FontSize',14,'FontWeight','bold')
%     zlabel({'Channel'; 'response (a.u.)'}); set(get(gca,'zlabel'),'FontSize',14,'FontWeight','bold')
%     %set(get(gca,'ylabel'),'rotation',90); %where angle is in degrees
%     grid off
%     title(sprintf("Aligned channel response for angle %i", ii));
    
    cnt = cnt+1;
    subplot(3,2,cnt);
    surf(xx,yy,squeeze(mean(recons_aligned_all(:,:,:,ii),3)),'EdgeColor','none','LineStyle','none','FaceLighting','phong');
    shading interp
    h=findobj('type','patch');
    set(h,'linewidth',2)
    hold on
    set(gca,'color','none')
    set(gca,'LineWidth',2,'TickDir','out');
    view(3)
    %axis([x(1) x(end) tpts_recon(1) em.time(end) lim]);
    set(gca,'YTick',[0:45:180])
    %set(gca,'XTick',[recon_tpt_range(1):500:recon_tpt_range(2)])
    ylabel('信息通道');
    xlabel('时间(ms)');
    set(get(gca,'xlabel'),'FontSize',14,'FontWeight','bold')
    set(get(gca,'ylabel'),'FontSize',14,'FontWeight','bold')
    zlabel("响应幅度"); set(get(gca,'zlabel'),'FontSize',14,'FontWeight','bold')
    %set(get(gca,'ylabel'),'rotation',90); %where angle is in degrees
    grid off
    title(sprintf("重构刺激响应（刺激%i,线索2）", ii));
    set(gca,'FontSize',10)
end

%% plot fidelity
figure;
hold on;
% smoothed_fidelity = smoothdata(smoothdata(all_fidelity_all));
smoothed_fidelity = smoothdata(all_fidelity_all, 'gaussian',16);
for ii = 1:2
   plot(squeeze(mean(smoothed_fidelity(:,:,ii), 2))); 
   title("Fidelity");
   xlabel("Time point");
end
legend(["angle 1", "angle 2"]);

%% slope
% smooth data
% smoothed_slope = smoothdata(smoothdata(slope_all_all));
smoothed_slope = smoothdata(slope_all_all,'gaussian',16);
% smoothed_slope = slope_all_all;

% compute baseling, mean and std
baseline = squeeze(mean(mean(smoothed_slope(base_start:base_end, :, :), 1), 2));
m = squeeze(mean(smoothed_slope, 2));
sd = squeeze(std(smoothed_slope, 0, 2))/sqrt(n_sub);

% t-test
H = nan(n_tpt, 2);
centered_slope = smoothed_slope;
for ii = 1:2
    centered_slope(:,:,ii) = smoothed_slope(:,:,ii)-baseline(ii);
    for tt = 1:n_tpt
        H(tt, ii) = ttest(centered_slope(tt, :, ii),0,'alpha',0.05,'Tail','Right');
    end
end
clear centered_slope

% plot
% init
subplot(3,2,[5,6]);
hold on;
h = gca;

% params
heights = [-0.008, -0.007];
lines = [];

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

% % plot time info
for i = 1:size(start_time, 2)
   plot([start_time(i), start_time(i)], [miny, maxy],  'Color','k','LineStyle','--','LineWidth',1.5); 
   plot([end_time(i), end_time(i)], [miny, maxy],  'Color','k','LineStyle','--','LineWidth',1.5);  
end

% info
legend(lines,["刺激1","刺激2"],'Box','off');
xlabel("时间（ms）");
ylabel("斜率");
set(gca,'Box','on')