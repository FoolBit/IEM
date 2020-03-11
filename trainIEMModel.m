%%
clear;close all; clc;
%% load data
addpath utils/;
load(sprintf('./data/hfdata_encode_all31'));
load(sprintf('./data/angles.mat'));
% load data/eegdata.mat

%% data data
chan_resp_aligned_all = nan(6, 400, 31, 2);
recons_aligned_all = nan(180, 400, 31, 2);
all_fidelity_all = nan(400, 31, 2);
slope_all_all = nan(400, 31, 2);
decode_errs_all = nan(400, 31, 2);


%% params
sampling_rate = 1/.01;
train_epoch = 50;
train_group = 5;
time_window = 1;

tpts = 1:400;
twindow = [1,400];

%% train!
for which_subject = 1:31
    
    for chose_ang = 1:2
        fprintf(datestr(now,'yyyy-mm-dd HH:MM:SS')+"Processing subject: %i, angle: %i--------start\n", which_subject, chose_ang);
        %% chose subject data
        data_all = smoothdata(smoothdata(hfdata{which_subject}, 3), 3);
        % data_all = rawdata_d{which_subject};
        label_all = [ts1(which_subject,:)',ts2(which_subject,:)'];
        
        
        %% other params
        labels = label_all(:,chose_ang);
        conds = sort(unique(labels));
        n_cond = length(conds);
        
        pos_colors = hsv(length(conds));
        n_elec = size(data_all,2);
        n_tpt = size(data_all,3);
        n_trial = size(data_all, 1);
        
        %% Step 1: Build spatial IEM (same as in fundamentals tutorial!)
        
        n_chan = 6; % # of channels, evenly spaced around the screen
        % chan_centers = linspace(180/n_chan,180,n_chan);
        chan_centers = conds;
        
        chan_width = 90;
        
        % evaluate basis set at these
        angs = linspace(1,180,180)';
        
        tuning_funcs = build_basis_polar_mat(angs,chan_centers,chan_width);
        
        %% Step 2a: Use basis set to compute channel responses
        
        stim_mask = zeros(n_trial,length(angs));
        for tt = 1:n_trial
            stim_mask(tt,angs==labels(tt)) = 1;
        end
        
        X_all = stim_mask * tuning_funcs;
        
        %% Step 2b & 3: Train/test IEM (full delay period) - leave-one-run-out
        fprintf(datestr(now,'yyyy-mm-dd HH:MM:SS')+"Processing subject: %i, angle: %i--------train IEM\n", which_subject, chose_ang);
        
        rng(3728291);% fill this in with estimated channel responses
        
        chan_resp = nan(n_trial, n_chan, n_tpt);
        
        for tt = 1:n_tpt
            % cat more timepoint
            tpt_idx = abs(tpts-tt)<=time_window;
            data = reshape(data_all(:,:,tpt_idx),n_trial,(sum(tpt_idx))*n_elec);
            chan_resp(:,:,tt) = train_IEM(data, X_all, train_epoch, train_group);
        end
        
        fprintf(datestr(now,'yyyy-mm-dd HH:MM:SS')+"Processing subject: %i, angle: %i--------finish train IEM\n", which_subject, chose_ang);
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
        
        %% 'fidelity'
        compute_fidelity = @(rec) mean( rec .* cosd(angs') ,2);
        % compute_fidelity = @(rec) mean( bsxfun(@times,rec,sind(angs')) ,2);
        
        % compute fidelity for each trial
        all_fidelity = nan(n_trial,n_tpt);
        for tt = 1:n_tpt
            all_fidelity(:,tt) = compute_fidelity(squeeze(recons_aligned(:,:,tt)));
            
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
        
        %% Now, let's decode ALL trials!
        decode_errs = nan(n_trial, n_tpt);
        
        for tt = 1:n_tpt
            vec_sum_all = chan_resp(:,:,200) * [cosd(chan_centers) sind(chan_centers)];
            
            decoded_angs = atan2d(vec_sum_all(:,2),vec_sum_all(:,1));
            
            % and let's look at a histogram of decoding errors
            tmperr = abs(decoded_angs - labels);
            tmperr = min(tmperr, abs(360-tmperr));
            decode_errs(:, tt) = min(tmperr, abs(180-tmperr));
            clear tmperr;
        end
        
        %% retrive data
        fprintf(datestr(now,'yyyy-mm-dd HH:MM:SS')+"Processing subject: %i, angle: %i--------finish\n", which_subject, chose_ang);

        chan_resp_aligned_all(:, :, which_subject, chose_ang) = ...
            squeeze(mean(chan_resp_aligned));
        recons_aligned_all(:, :, which_subject, chose_ang) = ...
            squeeze(mean(recons_aligned));
        all_fidelity_all(:, which_subject, chose_ang) = squeeze(mean(all_fidelity));
        slope_all_all (:, which_subject, chose_ang)= slope_all;
        decode_errs_all(:, which_subject, chose_ang) = ...
            squeeze(mean(decode_errs));
        
    end % chose_ang
    
end % which_subject

%% save
save IEMdata.mat chan_resp_aligned_all recons_aligned_all all_fidelity_all slope_all_all decode_errs_all