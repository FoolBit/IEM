function acc = ModelEval(X, Y, Models, classes)
% DISCARD
% compute accurace

%% params
n_tpt = size(X, 3);
n_trial = size(X, 1);

%% discard
% %%
% for tpt = 1:n_tpt
%     X_test = X(:,:,tpt);
%     Scores = zeros(size(X_test,1),numel(classes));
%
%     % predict
%     for j = 1:numel(classes)
%         [~,score] = predict(Models{j},X_test);
%         Scores(:,j) = score(:,2); % Second column contains positive-class scores
%     end
%
%     %%
%     [~,maxScore] = max(Scores,[],2);
%     acc(tpt) = sum(classes(maxScore)==Y)/n_trial;
%
% end

%% predict
% reshape to speed up
X = reshape(permute(X,[1 3 2]),n_trial*n_tpt,size(X,2));

% predict
Scores = zeros(size(X,1),numel(classes));
for j = 1:numel(classes)
    [~,score] = predict(Models{j},X);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);
maxScore = reshape(maxScore, n_trial, n_tpt);

%% get result
acc = sum(classes(maxScore)==Y, 1)/n_trial;
end

