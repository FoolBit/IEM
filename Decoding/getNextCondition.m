function conditions = getNextCondition(conditions, state)

% n_cond = 3;
corrs = cell({0, 1});
probes = cell({0, 40, 42});
cues = cell({0, 50, 52, 54, 56,[50, 54], [52, 56], [50,52], [54,56]});

n_corr = length(corrs);
n_probe = length(probes)*n_corr;
n_cues = length(cues)*n_corr;

% corr_idx = 1;
% probe_idx = 2;
% cue_idx = 3;

if isempty(conditions)
    conditions = struct;
    conditions.status = true;
    
    conditions.cnt = 1;
    conditions.cond_num = ones(1, 2);
    % encode
    if(state == "encode")
        max_cnt = 2;
    elseif(state == "probe")
        max_cnt = n_probe;
    elseif(state == "cue")
        max_cnt = n_cues;
    end
    
    conditions.cond_num(1) = floor((conditions.cnt-1)/max_cnt);
    conditions.cond_num(2) = myMod(conditions.cnt, max_cnt);
    
    conditions.corr_cc = 0;
    conditions.probe_cc = probes(1);
    conditions.cue_cc = cues(1);
    return
end

%------------------------
conditions.cnt = conditions.cnt


%------------------------


% conditions.cond_num(cue_idx) = conditions.cond_num(cue_idx) + 1;
% 
% conditions.cond_num(probe_idx) = conditions.cond_num(probe_idx) + ...
%     floor( (conditions.cond_num(cue_idx)-1)/n_cues);
% conditions.cond_num(cue_idx) = myMod(conditions.cond_num(cue_idx), n_cues);
% 
% conditions.cond_num(corr_idx) = conditions.cond_num(corr_idx) + ...
%     floor((conditions.cond_num(probe_idx)-1)/n_probe);
% conditions.cond_num(probe_idx) = myMod(conditions.cond_num(probe_idx), n_probe);
% 
% if conditions.cond_num(corr_idx)>2
%     conditions.status = false;
%     return;
% end
% 
% conditions.corr_cc = corrs(conditions.cond_num(corr_idx));
% conditions.probe_cc = probes(conditions.cond_num(probe_idx));
% conditions.cue_cc = cues(conditions.cond_num(cue_idx));

end

