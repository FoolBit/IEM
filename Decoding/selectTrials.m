function [conditions, trial_idx] = selectTrials(conditions, infos, max_trials, state)

acc_info = infos.acc_info;
probe_info = infos.probe_info;
cue_info = infos.cue_info;

% -------------------------------- set conditions
% conditions
corrs = cell({1});
probes = cell({0, 40, 42});
cues = cell({0, 50, 52, 54, 56,[50, 54], [52, 56]});

% num of conditions for each phase
n_corr = length(corrs);
n_encode = 1;
n_probe = length(probes);
n_cues = length(cues);

% 
if(isempty(conditions))
   conditions = struct;
   conditions.status = true;
   conditions.cnt = 5;
end

conditions.cnt = conditions.cnt+1;

trial_idx = true(max_trials, 1);

if(state == "encode")
    
    % go through all conditions?
    if(conditions.cnt > n_encode*n_corr)
        conditions.status = false;
        return;
    end
    
    fprintf("conditions: %i, with total conditions: %i\n", conditions.cnt, n_encode*n_corr);
    
    corr_cc = floor((conditions.cnt-1)/n_encode);
    
elseif(state == "probe")
    % go through all conditions?
    if(conditions.cnt > n_probe*n_corr)
        conditions.status = false;
        return;
    end
    
    fprintf("conditions: %i, with total conditions: %i\n", conditions.cnt, n_probe*n_corr);
    
    corr_cc = floor((conditions.cnt-1)/n_probe);
    probe_cc = probes{myMod(conditions.cnt, n_probe)};
    
    if(any(probe_cc>0))
        
        trial_idx = probe_info==probe_cc(1);
        for i = 2:length(probe_cc)
            trial_idx = trial_idx | (probe_info==probe_cc(i));
        end
        
    end
   
elseif(state == "cue")
     % go through all conditions?
    if(conditions.cnt > n_cues*n_corr)
        conditions.status = false;
        return;
    end
    
    fprintf("conditions: %i, with total conditions: %i\n", conditions.cnt, n_cues*n_corr);
    
    corr_cc = floor((conditions.cnt-1)/n_cues);
    cue_cc = cues{myMod(conditions.cnt, n_cues)};
    
    if(any(cue_cc>0))
        
        trial_idx = cue_info==cue_cc(1);
        for i = 2:length(cue_cc)
            trial_idx = trial_idx | (cue_info==cue_cc(i));
        end
    end
    
end

% correct
if corr_cc > 0
    trial_idx = (acc_info==corr_cc) & trial_idx;
end

end

