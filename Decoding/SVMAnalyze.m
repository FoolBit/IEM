function SVMAnalyze(state, data, infos_all, angles)


%% choose subject
wrong_sub = [16, 19, 26];
n_wrong = size(wrong_sub, 2);

idx = ones(31, 1);
for i = 1:n_wrong
    idx(wrong_sub(i)) = 0;
end
idx = logical(idx);

%%
data = data(idx);
infos_all = infos_all(idx);
angles{1} = angles{1}(idx, :);
angles{2} = angles{2}(idx, :);

%% params
n_subject = sum(idx);
n_tpt = size(data{1},3);

plot_gate = 0;
acc_CV_SVM_all = nan(n_tpt, 2, n_subject);
% acc_CV_KNN_all = nan(400, 2, 31);
acc_pred_SVM_all = nan(n_tpt, 2, n_subject);
% acc_pred_KNN_all = nan(400, 2, 31);

%% train!
conditions = [];

while(true)
    last_conditions = conditions;
    %% loop for subjects
    for which_subject = 1:n_subject
        fprintf(datestr(now)+"        start process subject: %i\n",which_subject);
        data_all = smoothdata(smoothdata(data{which_subject}, 3), 3);
        label_all = [angles{1}(which_subject,:)',angles{2}(which_subject,:)'];
        
        %% choose trials
        max_trials = size(data_all, 1);
        
        [conditions, trial_idx] = selectTrials(last_conditions, infos_all{which_subject}, max_trials, state);
        if(conditions.status == false)
            return;
        end
        
        data_all = data_all(trial_idx,:,:);
        label_all = label_all(trial_idx, :);
        %% params
        % choose_ang = 1;
        n_trial = size(data_all, 1);
        %% preprocess data
        rng(1);
        acc_CV_SVM = nan(n_tpt, 2);
        % acc_CV_KNN = nan(n_tpt, 2);
        % acc_pred_SVM = nan(size(data_all,3));
        % acc_pred_KNN = nan(size(data_all,3));
        % acc_pred_SVM = nan(n_tpt, 2);
        % acc_pred_KNN = nan(n_tpt, 2);
        % X_test = reshape(permute(data_all,[1 3 2]),n_trial*n_tpt,size(data_all,2));
        % Y = label_all(:,choose_ang);
        
        %% train model
        for choose_ang = 1:2
            Y = label_all(:,choose_ang);
            for tpt = 1:n_tpt
                fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------loop begin\n",which_subject, tpt, choose_ang);
                
                % retrive train data
                X_train = squeeze(data_all(:,:,tpt));
                data_train = [X_train, Y];
                
                % train
                fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------training SVM\n",which_subject, tpt, choose_ang);
                [SVM, acc_CV_SVM(tpt, choose_ang)] = trainSVMClassifier(data_train);
                % fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------training KNN\n",ss, tpt, choose_ang);
                % [KNN, acc_CV_KNN(tpt, choose_ang)] = trainSVMClassifier(data_train);
                
                % predict
                % fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------predicting using SVM\n",which_subject, tpt, choose_ang);
                %         y_pred_SVM = SVM.predictFcn(X_test);
                %         y_pred_SVM = reshape(y_pred_SVM, n_trial, n_tpt);
                %         acc_pred_SVM(tpt, :) = sum(y_pred_SVM==Y, 1)/n_trial;
                % y_pred_SVM = SVM.predictFcn(X_train);
                % acc_pred_SVM(tpt, choose_ang) = sum(y_pred_SVM==Y, 1)/n_trial;
                
                % fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------predicting using KNN\n",ss, tpt, choose_ang);
                %         y_pred_KNN = KNN.predictFcn(X_test);
                %         y_pred_KNN = reshape(y_pred_KNN, n_trial, n_tpt);
                %         acc_pred_KNN(tpt, :) = sum(y_pred_KNN==Y, 1)/n_trial;
                % y_pred_KNN = KNN.predictFcn(X_train);
                % acc_pred_KNN(tpt, choose_ang) = sum(y_pred_KNN==Y, 1)/n_trial;
                
                fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------Finished\n",which_subject, tpt, choose_ang);
            end
        end
        
        acc_CV_SVM_all(:, :, which_subject) = acc_CV_SVM;
        % acc_CV_KNN_all(:, :, ss) = acc_CV_KNN;
        % acc_pred_SVM_all(:, :, which_subject) = acc_pred_SVM;
        % acc_pred_KNN_all(:, :, ss) = acc_pred_KNN;
        fprintf(datestr(now)+"        finish process subject: %i !\n",ss);
    end
    %%
    filename = sprintf("../data/processed_data/SVMdatanew_%s_%i", state, conditions.cnt);
    save(filename, 'acc_CV_SVM_all');
end

end