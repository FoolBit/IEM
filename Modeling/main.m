clc;clear;close all;

%% load data
load('WMnew.mat');

%% params
n_tpt = size(h, 1);
n_trial = size(h, 2);

%% init var
acc_neuron_cue = nan(n_tpt, 2, 2);
acc_syn_cue = nan(n_tpt, 2, 2);

%% trial select
idx = true(1,n_trial);
cue1_idx = (cue == 0) & idx;
cue2_idx = (cue == 1) & idx;
cue_idx = [cue1_idx;cue2_idx];

clear idx cue1_idx cue2_idx
%% train
for t = 1:n_tpt
    for c = 1:2
        idx = cue_idx(c,:);
        fprintf('Processing timepoint: %i\n',t);
        trainingData = [squeeze(h(t,idx,:)), double(stim1(idx)')];
        [~, acc_neuron_cue(t, 1, c)] = trainLinearSVM(trainingData);
        
        trainingData = [squeeze(h(t,idx,:)), double(stim2(idx)')];
        [~, acc_neuron_cue(t, 2, c)] = trainLinearSVM(trainingData);
        
        trainingData = [squeeze(syn_efficacy(t,idx,:)), double(stim1(idx)')];
        [~, acc_syn_cue(t, 1, c)] = trainLinearSVM(trainingData);
        
        trainingData = [squeeze(syn_efficacy(t,idx,:)), double(stim2(idx)')];
        [~, acc_syn_cue(t, 2, c)] = trainLinearSVM(trainingData);
    end
end

%% save data
save('accData_wmnew_cue.mat','acc_neuron_cue','acc_syn_cue');