%%
clc; clear; close all;

%% load data
addpath utils/
load("../data/processed_data/IEMdata_probe_1_0.mat");

%% params
n_chan = 6;
n_tpt = size(slope_all_all, 1);
tpts = 1:n_tpt;
base_start = 1;
base_end = 50;

colors = ['b';'r'];
start_time = [100];
end_time = [250];

%% plot channel response aligned && reconstruct response aligned
for ii = 1:2
    figure;
    % plot mean
    subplot(2,2,1);
    imagesc(squeeze(mean(chan_resp_aligned_all(:,:,:,ii),3)));
    colorbar;
    title(sprintf("Aligned channel response for angle %i", ii));
    set(gca,'TickDir','out','FontSize',14);
    ylabel('Channel');
    xlabel('Time point');
    hold on; 
    for jj = 1:size(start_time, 2)
        plot(start_time(jj)*[1 1],[0, 16], ...
            'Color', colors(jj), 'LineWidth',.75, 'LineStyle','--');
        plot(end_time(jj)*[1 1],[0, 16], ...
            'Color', colors(jj), 'LineWidth',.75, 'LineStyle','--');
    end
    
    subplot(2,2,2);
    imagesc(squeeze(mean(recons_aligned_all(:,:,:,ii),3)));
    colorbar;
    title(sprintf("Aligned Rsconstructed channel response for angle %i", ii));
    set(gca,'TickDir','out','FontSize',14);
    ylabel('Channel');
    xlabel('Time point');
    hold on;
    for jj = 1:size(start_time, 2)
        plot(start_time(jj)*[1 1],[0, 180], ...
            'Color', colors(jj), 'LineWidth',.75, 'LineStyle','--');
        plot(end_time(jj)*[1 1],[0, 180], ...
            'Color', colors(jj), 'LineWidth',.75, 'LineStyle','--');
    end
    
    % 3D plot
    subplot(2,2,3);
    surf(squeeze(mean(chan_resp_aligned_all(:,:,:,ii),3)),'EdgeColor','none','LineStyle','none','FaceLighting','phong');
    shading interp
    h=findobj('type','patch');
    set(h,'linewidth',2)
    hold on
    set(gca, 'box','off')
    set(gca,'color','none')
    set(gca,'LineWidth',2,'TickDir','out');
    set(gca,'FontSize',14)
    view(3)
    %axis([x(1) x(end) tpts_recon(1) em.time(end) lim]);
    set(gca,'YTick',[0:45:180])
    %set(gca,'XTick',[recon_tpt_range(1):500:recon_tpt_range(2)])
    ylabel('Channel Offset (\circ)');
    xlabel('Time (ms)');
    set(get(gca,'xlabel'),'FontSize',14,'FontWeight','bold')
    set(get(gca,'ylabel'),'FontSize',14,'FontWeight','bold')
    zlabel({'Channel'; 'response (a.u.)'}); set(get(gca,'zlabel'),'FontSize',14,'FontWeight','bold')
    %set(get(gca,'ylabel'),'rotation',90); %where angle is in degrees
    grid off
    title(sprintf("Aligned channel response for angle %i", ii));
    
    subplot(2,2,4);
    surf(squeeze(mean(recons_aligned_all(:,:,:,ii),3)),'EdgeColor','none','LineStyle','none','FaceLighting','phong');
    shading interp
    h=findobj('type','patch');
    set(h,'linewidth',2)
    hold on
    set(gca, 'box','off')
    set(gca,'color','none')
    set(gca,'LineWidth',2,'TickDir','out');
    set(gca,'FontSize',14)
    view(3)
    %axis([x(1) x(end) tpts_recon(1) em.time(end) lim]);
    set(gca,'YTick',[0:45:180])
    %set(gca,'XTick',[recon_tpt_range(1):500:recon_tpt_range(2)])
    ylabel('Channel Offset (\circ)');
    xlabel('Time (ms)');
    set(get(gca,'xlabel'),'FontSize',14,'FontWeight','bold')
    set(get(gca,'ylabel'),'FontSize',14,'FontWeight','bold')
    zlabel({'Channel'; 'response (a.u.)'}); set(get(gca,'zlabel'),'FontSize',14,'FontWeight','bold')
    %set(get(gca,'ylabel'),'rotation',90); %where angle is in degrees
    grid off
    title(sprintf("Aligned Rsconstructed channel response for angle %i", ii));
end

%% plot fidelity
figure;
hold on;
smoothed_fidelity = smoothdata(smoothdata(all_fidelity_all));
for ii = 1:2
   plot(squeeze(mean(smoothed_fidelity(:,:,ii), 2))); 
   title("Fidelity");
   xlabel("Time point");
end
legend(["angle 1", "angle 2"]);

%% slope
% smooth data
% smoothed_slope = smoothdata(smoothdata(slope_all_all));
smoothed_slope = slope_all_all;

% compute baseling, mean and std
baseline = squeeze(mean(mean(smoothed_slope(base_start:base_end, :, :), 1), 2));
m = squeeze(mean(smoothed_slope, 2));
sd = squeeze(std(smoothed_slope, 0, 2));

% t-test
H = nan(n_tpt, 2);
centered_slope = smoothed_slope;
for ii = 1:2
    centered_slope(:,:,ii) = smoothed_slope(:,:,ii)-baseline(ii);
    for tt = 1:n_tpt
        H(tt, ii) = ttest(centered_slope(tt, :, ii),0,'alpha',0.005);
    end
end
clear centered_slope

% plot
% init
f = figure;
hold on;
h = gca;

% params
heights = [0.015, 0.015];
lines = [];

% plot start!
for i = 1:2
    lines = [lines,plot(h, m(:,i), 'Color', colors(i))];
    plotPatch(h, tpts, m(:,i)', sd(:,i)', colors(i));
    plotSig(f, tpts, H(:,i), heights(i), colors(i), 1.2);
    plot([1, n_tpt], [baseline(i), baseline(i)], strcat(colors(i),'--'));
end

% change ylim
tmp_ylim = get(h,'YLim');
miny = tmp_ylim(1)-0.01;
maxy = tmp_ylim(2)+0.01;
set(h,'YLim',[miny, maxy]);
clear tmp_ylim

% plot time info
for i = 1:size(start_time, 2)
   plot([start_time(i), start_time(i)], [miny, maxy], strcat(colors(i),':')); 
   plot([end_time(i), end_time(i)], [miny, maxy], strcat(colors(i),':')); 
end

% info
legend(lines,["angle 1","angle 2"]);
xlabel("Time point");
ylabel("Slope");
title("Slope");