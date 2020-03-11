%%
clear;close all; clc;
%% load data
addpath utils/;
load(sprintf('./data/hfdata_encode_all31'));
load(sprintf('./data/angles.mat'));
% load data/eegdata.mat

%% chose one subject
which_subject = 1;
data_all = smoothdata(smoothdata(hfdata{which_subject}, 3), 3);
% data_all = rawdata_d{which_subject};
label_all = [ts1(which_subject,:)',ts2(which_subject,:)'];

%% params
sampling_rate = 1/.01;
train_epoch = 50;
train_group = 5;
time_window = 1;

chose_ang = 2;
tpts = 1:400;
twindow = [1,400];
delay_window = [120,120];

%% other params
labels = label_all(:,chose_ang);
conds = sort(unique(labels));
n_cond = length(conds);

pos_colors = hsv(length(conds));
n_elec = size(data_all,2);
n_tpt = size(data_all,3);
n_trial = size(data_all, 1);

DEBUG = 1;  % DEBUG=1 to plot lots of unuseful figures

%% choose trials based on diff
% threhold = 0;
% diff = min(mod(label_all(:,1) - label_all(:,2),180),mod(label_all(:,2) - label_all(:,1),180));
% idx = diff >= threhold;
% labels = labels(idx);
% data_all = data_all(idx,:,:);
% 
% n_elec = size(data_all,2);
% n_tpt = size(data_all,3);
% n_trial = size(data_all, 1);

%% random label
% indx = randperm(length(labels));
% labels = labels(indx);

%% plot labels
if DEBUG == 1
    figure;hold on;
    for tt = 1:n_cond
        idx = label_all(:,1) == conds(tt);
        scatter(conds(tt)*ones(sum(idx),1),label_all(idx, 2));
    end
    
    figure;hold on;
    bin_edges = [conds-1;180];
    bin_centers = (bin_edges(1:end-1)+bin_edges(2:end))/2;
    for tt = 1:n_cond
        subplot(2,3,tt);
        idx = labels == conds(tt);
        h = histogram(label_all(idx,2),[conds-1;180]);
        nn = h.BinCounts;
        for i = 1:length(nn)
            % 直方图上面数据对不齐，利用水平和垂直对齐 ，可以参考search ?Document 中的text函数
            text(bin_centers(i),nn(i)+0.5,num2str(nn(i)),'VerticalAlignment','middle','HorizontalAlignment','center');
        end
        ylim([0, 26])
    end
end
%% plot mean response across time for each condition
resp_across_time = nan(n_elec,n_tpt,n_cond);
for tt = 1:n_tpt
    for i = 1:n_cond
        idx = labels == conds(i);
        resp_across_time(:,tt,i) = mean(data_all(idx, :, tt), 1);
    end
end

figure;
for i = 1:n_cond
    subplot(3,2,i);
    imagesc(resp_across_time(:,:,i));
    colorbar;
    title(strcat(sprintf("Mean response for %i", conds(i)),"\circ"));
    set(gca,'TickDir','out','FontSize',14);
    ylabel('Electrode');
    xlabel('Time point');
end

%% Step 1: Build spatial IEM (same as in fundamentals tutorial!)

n_chan = 6; % # of channels, evenly spaced around the screen
% chan_centers = linspace(180/n_chan,180,n_chan);
chan_centers = conds;

chan_width = 90;

% evaluate basis set at these
angs = linspace(1,180,180)';

tuning_funcs = build_basis_polar_mat(angs,chan_centers,chan_width);

if DEBUG == 1
    % now let's look at the basis set:
    figure; plot(angs,tuning_funcs,'LineWidth',1.5);
    xlabel('Angle (\circ)');ylabel('Channel sensitivity'); title('Basis set (information channels)');
    xlim([0 180]);set(gca,'XTick',0:45:180,'TickDir','out','Box','off','FontSize',14);
    
    legend('10','40','70','100','130','160','Box','off');
end
%% Step 2a: Use basis set to compute channel responses

stim_mask = zeros(n_trial,length(angs));
for tt = 1:n_trial
    stim_mask(tt,angs==labels(tt)) = 1;
end

X_all = stim_mask * tuning_funcs;

if DEBUG == 1
    % let's check out the computed channel responses:
    figure;
    % first, a single trial
    whichtrial_C = 21;
    subplot(1,2,1);
    hold on;  chan_colors = lines(n_chan);
    plot(chan_centers,X_all(whichtrial_C,:),'k-','LineWidth',1.5);
    for cc = 1:n_chan
        plot(chan_centers(cc),X_all(whichtrial_C,cc),'ko','MarkerSize',10,'MarkerFaceColor',chan_colors(cc,:))
    end
    plot([1 1]*labels(whichtrial_C),[0 1],'k--','LineWidth',2);
    xlabel('Channel center (\circ)');
    ylabel('Predicted channel response');
    title(sprintf('Trial %i: %i\\circ',whichtrial_C,labels(whichtrial_C)));
    set(gca,'TickDir','out','XTick',0:45:180,'FontSize',14);
    
    
    % all trials
    subplot(1,2,2); hold on;
    imagesc(chan_centers,1:size(labels,1),X_all); axis ij;
    plot([0 180],whichtrial_C+[-0.5 0.5],'r-');
    plot(labels,1:n_trial,'r+','LineWidth',1.5,'MarkerSize',5);
    xlim([-1 160+15]);ylim([0 50]);
    xlabel('Channel center (\circ)');
    ylabel('Trial');
    
    % put the rank of the design matrix (X_all) in the title
    title(sprintf('All trials (rank = %i)',rank(X_all)));
    set(gca,'TickDir','out','XTick',0:45:180,'FontSize',14);
end

%% Step 2b & 3: Train/test IEM (full delay period) - leave-one-run-out
rng(3728291);% fill this in with estimated channel responses

chan_resp = nan(n_trial, n_chan, n_tpt);

for tt = 1:n_tpt
    % cat more timepoint
    tpt_idx = abs(tpts-tt)<=time_window;
    data = reshape(data_all(:,:,tpt_idx),n_trial,(sum(tpt_idx))*n_elec);
    chan_resp(:,:,tt) = train_IEM(data, X_all, train_epoch, train_group);
end

%%
if DEBUG == 1
    figure;
    for pp = 1:length(conds)
        subplot(3,2,pp);
        imagesc(squeeze( mean(chan_resp(labels==conds(pp),:,:),1)));
        colorbar;
        title(strcat(sprintf("Mean channel response for %i", conds(pp)),"\circ"));
        set(gca,'TickDir','out','FontSize',14);
        ylabel('Channel');
        xlabel('Time point');
        set(gca,'TickDir','out','XTick',0:50:n_tpt,'FontSize',14);
    end
end
%% Align them
targ_pos = 90;
[~,targ_idx] = min(abs(chan_centers-targ_pos));
rel_chan_centers = chan_centers - targ_pos; % x values for plotting - relative channel position compared to aligned

% create a new variable where we'll put the 'shifted' versions of channel response functions
chan_resp_aligned = nan(size(chan_resp));

% loop over all trials
for tt = 1:n_trial
    shift_by = targ_idx - find(conds==labels(tt)); % when targ_pos is higher, we need to shift right
    chan_resp_aligned(tt,:,:) = circshift(chan_resp(tt,:,:),shift_by);
end

% plot it - plot all trials aligned, sorted by bin in their bin color (liek above)
% then add the mean across all trials on top

if DEBUG == 1
    figure;
    
    for pp = 1:n_cond
        subplot(3,2,pp);
        idx = labels==conds(pp);
        imagesc(squeeze( mean(chan_resp_aligned(idx,:,:),1)));
        colorbar;
        title(strcat(sprintf("Aligned Mean channel response for %i", conds(pp)),"\circ"));
        set(gca,'TickDir','out','FontSize',14);
        ylabel('Channel');
        xlabel('Time point');
    end
    % plot mean
    figure;
    imagesc(squeeze(mean(chan_resp_aligned,1)));
    colorbar;
    title("Aligned Mean channel response for all conditions");
    set(gca,'TickDir','out','FontSize',14);
    ylabel('Channel');
    xlabel('Time point');
end
%% Convert channel responses to 'reconstructions'
recons_raw = nan(n_trial, size(angs,1), n_tpt);
for tt = 1:n_tpt
    recons_raw(:,:,tt) = chan_resp(:,:,tt) * tuning_funcs.';
end
recons_aligned = nan(size(recons_raw));

% loop over trials
for tt = 1:n_trial
    % we want to adjust so that each position is set to 0
    shift_by = labels(tt); % if this is +, shift left (so use -1*shift_by)
    recons_aligned(tt,:,:) = circshift(recons_raw(tt,:,:),targ_pos-shift_by);
end

% NOTE: the above assumes angs is a set of integers, spaced by 1. I'll
% leave it as an exercise to re-implement this procedure for an arbitrary
% feature space. there are hints about how to do this in Sprague et al,
% 2016 (and the code is available online)

%%
% plot aligned reconstructions: all indiv trials and average
% (we're only going to plot non-excluded trials here)
if DEBUG == 1
    figure;
    for pp = 1:n_cond
        subplot(3,2,pp);
        idx = labels==conds(pp);
        imagesc(squeeze( mean(recons_aligned(idx,:,:),1)));
        colorbar;
        title(strcat(sprintf("Rsconstructed Mean channel response for %i", conds(pp)),"\circ"));
        set(gca,'TickDir','out','FontSize',14);
        ylabel('Channel');
        xlabel('Time point');
    end
    
    % plot mean
    figure;
    imagesc(squeeze(mean(recons_aligned,1)));
    colorbar;
    title("Aligned Rsconstructed channel response for all conditions");
    set(gca,'TickDir','out','FontSize',14);
    ylabel('Channel');
    xlabel('Time point');
    
    % 3D plot
    figure;
    surf(squeeze(mean(recons_aligned,1)),'EdgeColor','none','LineStyle','none','FaceLighting','phong')
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
    
end
%% Quantifying reconstructions: 'fidelity' (Sprague et al, 2016)
if DEBUG == 1
    figure;
    ax1 = subplot(1,2,1,polaraxes);
    hold on;
    for pp = 1:n_cond
        thisidx = labels==conds(pp);
        
        % plot the reconstruction in polar coords
        polarplot(deg2rad(angs),mean(recons_raw(thisidx,:,120),1),'-','Color',pos_colors(pp,:),'LineWidth',1.5);
        clear thisidx;
    end
    title('Aligned reconstructions (binned)');
    
    % now let's focus on one bin (which_bin)
    ax2 = subplot(1,2,2,polaraxes);
    hold on;
    which_bin = 3;
    title(sprintf('Example bin (%i)',which_bin));
    
    
    % first, draw the same as above for 'background'
    
    % get the trials we're using:
    thisidx = labels==conds(which_bin);
    
    polarplot(deg2rad(angs),mean(recons_raw(thisidx,:,120),1),'-','Color',pos_colors(which_bin,:),'LineWidth',1.5);
    
    
    % and draw the mean of this position bin:
    polarplot([1 1]*deg2rad(conds(which_bin)),[0 1],':','Color',pos_colors(which_bin,:),'LineWidth',1);
    
    % we want to do the vector average; let's add some example vectors
    which_angs = (-135:45:180)-22; % (offsetting them from bin center)
    % (note: these are JUST FOR VISUALIZATION) - we'll compute the real vector
    % average below with all angles of the reconstruction. EXERCISE: is that
    % really necessary? how few angles can we get away with? what are the
    % constraints on which angles must be included?
    
    for aa = 1:length(which_angs)
        % figure out which bin of recons_raw to average for each plotted angle
        angidx = find(angs==which_angs(aa)); % note: could replace w/ round, etc, for non-matching angles....
        
    end
    
    % now, we compute a vector sum of ALL points on the reconstruction
    %
    % but remember, this is a CIRCULAR variable - we can't just take the mean
    % of recons_raw of these trials, so what do we do?
    
    % first, convert average reconstruction to X,Y
    [tmpx,tmpy] = pol2cart(deg2rad(angs),mean(recons_raw(thisidx,:,120),1)');
    
    % now, take the mean of X,Y
    mux = mean(tmpx); muy = mean(tmpy); clear tmpx tmpy;
    
    % and convert back to polar coordinates:
    [muth,mur] = cart2pol(mux,muy); % NOTE: muth is in RADIANS here
    
    % and draw:
    polarplot(muth*[1 1],mur*[0 1],'-','LineWidth',4,'Color',pos_colors(which_bin,:));
    
    %
    % Hopefully you can see that the average (thick colored line) aligns very
    % nicely with the 'true' polar angle (thin dashed line) of this bin. So
    % let's quantify this by projecting the vector mean (thick line) onto a
    % unit vector in the direction of the bin center. Projection is just the
    % dot product of the vector mean (mux,muy)<dot>(TrueX,TrueY), and TrueXY
    % are just cosd(th) and sind(th), where theta is the bin center:
    this_fidelity = dot([mux muy],  [cosd(conds(which_bin)) sind(conds(which_bin))]);
    text(-pi/2,2,sprintf('Fidelity = %.03f',this_fidelity),'HorizontalAlignment','center');
    
    
    
end
% EXERCISE: does the order of operations here matter? can you compute
% fidelity on each trial then average, or do you need to compute fidelity
% on the average?

%% simplifying the computation of 'fidelity'
% So - we know how to project the vector average of the reconstruction onto
% the unit vector in the 'correct' direction, which lets us quantify
% fidelity. But, we've also computed 'aligned' reconstructions (indeed, we
% almost always do this in most analyses). In that case, the projection
% onto the 'correct' position amounts to projecting onto the x axis - so we
% can simplify our calculation of fidelity when using aligned data:
% F = mean( r(theta) * cos(theta) )  where r(theta) is 0-centered
% reconstruction value

% let's make a function that does this:
compute_fidelity = @(rec) mean( rec .* cosd(angs') ,2);
% compute_fidelity = @(rec) mean( bsxfun(@times,rec,sind(angs')) ,2);

% compute fidelity for each trial
all_fidelity = nan(n_trial,n_tpt);
for tt = 1:n_tpt
    all_fidelity(:,tt) = compute_fidelity(squeeze(recons_aligned(:,:,tt)));
    
end
%
% plot distribution of fidelity across all trials
if DEBUG == 1
    figure;
    histogram(all_fidelity);
    xlabel('Fidelity');
    title('Fidelity across all trials');
    set(gca,'TickDir','out');
    
    %
    % plot average fidelity (+std dev) for each position bin
    % figure;
    % for pp = 1:n_cond
    %     subplot(3,2,pp);hold on;
    %     thisidx = labels==conds(pp);
    %     thismu  = mean(all_fidelity(thisidx,:),1);
    %     thiserr = std(all_fidelity(thisidx))./sqrt(sum(thisidx,1));
    %
    %     plot(tpts,thismu + thiserr,'-','LineWidth',1,'Color',pos_colors(pp,:));
    %     plot(tpts,thismu + thiserr*-1,'-','LineWidth',1,'Color',pos_colors(pp,:));
    %     plot(tpts, thismu, '*','LineWidth',1.2,'Color',pos_colors(pp,:))
    % end
    figure;
    % plot(smoothdata(smoothdata(mean(all_fidelity,1))));
    plot(mean(all_fidelity,1));
    xlabel('Timepoint');
    ylabel('Mean fidelity');
    title('Fidelity across time');
    tmpylim = get(gca,'YLim');
    xlim([0 400]);
    set(gca,'TickDir','out','XTick',0:50:400);
end
%% Quantifying reconstructions: 'slope' (Foster et al, 2016)
slope_all = nan(n_tpt, 1);

% flip
ang_stay = angs>=1 & angs <= 90;
ang_flip = angs>=91 & angs <= 180;
recons_aligned_flip = recons_aligned;
recons_aligned_flip(:,ang_stay,:) = recons_aligned_flip(:,ang_stay,:) + flip(recons_aligned_flip(:,ang_flip,:), 2);
mean_recons_aligned = squeeze(mean(recons_aligned_flip(:,ang_stay,:),1));

for tt = 1:n_tpt
    Y = mean_recons_aligned(:,tt);
    x = (1:90)';
    X = [ones(length(x),1) x];
    b = X\Y;
    slope_all(tt) = b(2);
end

if DEBUG == 1
    figure;
    
    plot(smoothdata(smoothdata(slope_all)));
    xlabel('Slope');
    ylabel('Timepoint');
    title('Computing slope');
    set(gca,'TickDir','out','XTick',0:50:400);
end
%% Using channel-based models to 'decode' stimulus features
% let's decode an example trial first:

decode_trial = 20; % 18 is good...
figure;subplot(1,2,1);hold on;

% plot the channel response profile for this trial
plot(chan_centers,chan_resp(decode_trial,:,200),'-','LineWidth',2);
plot([1 1]*mod(labels(decode_trial),180),[0 1],'r--','LineWidth',1.5);
xlabel('Modeled channel center (\circ)');
ylabel('Channel response profile');
title(sprintf('Trial %i',decode_trial));
set(gca,'XTick',chan_centers(2:2:end),'TickDir','out','FontSize',16);

% plot the vector average visually (in polar coords)

% first, the channel responses in polar coords
subplot(1,2,2);
polarplot(deg2rad(chan_centers),chan_resp(decode_trial,:,200),'k:','LineWidth',1.0);
hold on;

polarplot([1 1]*deg2rad(labels(decode_trial)),[0 1],'r--','LineWidth',.75);

for cc = 1:length(chan_centers)
    polarplot([1 1]*deg2rad(chan_centers(cc)),[0 chan_resp(decode_trial,cc,200)],'-','LineWidth',2);
end

% to compute the poulation vector, we just compute the sum of each
% population's preferred feature value (unit vector in each channel's tuned
% direction) weighted by its reconstructed response:

% this computes each channel's unit vector (given by [cosd(th) sind(th)]
% and multiplies it by the corresponding channel's response, and sums
% across channels
vec_sum_ex = mean(chan_resp(decode_trial,:, 200).*[cosd(chan_centers');sind(chan_centers')],2);

[vec_sum_ex_pol(1),vec_sum_ex_pol(2)] = cart2pol(vec_sum_ex(1),vec_sum_ex(2));
polarplot([1 1]*vec_sum_ex_pol(1),[0 vec_sum_ex_pol(2)],'k-','LineWidth',2.5);

%title(sprintf('Actual: %i \circ ; Decoded: %i \circ',mod(c_all(decode_trial,1),360),round(mod(rad2deg(vec_sum_ex_pol(1)),360))));

fprintf('Trial %i\nActual position:\t%f deg\n',decode_trial,mod(labels(decode_trial),180));
fprintf('Decoded position:\t%f deg\n',mod(rad2deg(vec_sum_ex_pol(1)),180));

% add the decoded value to the original channel response profile plot
subplot(1,2,1); hold on;
plot(mod(rad2deg(vec_sum_ex_pol(1)),360)*[1 1],[0 1],'k--','LineWidth',1.5);
legend({'Data','Actual','Decoded'},'Location','best');


% EXERCISE: look at other trials - not every one is perfect, but often they
% look quite good! (remember, this is *single-trial* decoding of a
% *continuous feature value*



%% Now, let's decode ALL trials!


vec_sum_all = chan_resp(:,:,200) * [cosd(chan_centers) sind(chan_centers)];

decoded_angs = atan2d(vec_sum_all(:,2),vec_sum_all(:,1));

% and plot them - compare each trial's decoded feature value to its actual value
figure; subplot(1,2,1); hold on;
scatter(labels,mod(decoded_angs,180),15,'k','filled','MarkerFaceAlpha',0.5);

%plot unity line
plot([0 180],[0 180],'k--');

xlabel('Actual position (\circ)');
ylabel('Decoded position (\circ)');
title('Channel-based vector average');

xlim([0 180]);ylim([0 180]);

axis equal square;

set(gca,'TickDir','out','FontSize',14,'XTick',0:90:360,'YTick',0:90:360);

% and let's look at a histogram of decoding errors
tmperr = abs(decoded_angs - labels);
tmperr = min(tmperr, abs(360-tmperr));
decoding_err = min(tmperr, abs(180-tmperr));
clear tmperr;

subplot(1,2,2);hold on;
histogram(decoding_err); xlim([0 90]);
xlabel('Decoding error (\circ)');
title('Single-trial decoding error');
set(gca,'TickDir','out','FontSize',14,'XTick',-180:90:180,'YTick',[]);