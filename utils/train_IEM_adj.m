function chan_resp = train_IEM_adj(data, X, n_epoch, n_group)

% set params
if nargin < 3 
    n_epoch = 50; % default value
end

if nargin < 4
    n_group = 2;
end
data = data-min(min(min(data)));
% train & test
n_trial = size(data, 1);
chan_resp = zeros(size(X));
for ee = 1: n_epoch
    % set to n_group
    num_per_group = floor(n_trial/n_group);
    idx = randperm(size(data,1));
    group_label = nan(size(idx));
    for ii = 1:n_group
        lb = (ii-1)*num_per_group + 1;
        ub = ii*num_per_group;
        if(ub > n_trial)
            ub = n_trial;
        end
        group_label(lb:ub) = ii;
    end
        
    for rr = 1:n_group
        trnidx = idx(group_label ~= rr); % train using all 'included' trials except testing run
        tstidx = idx(group_label == rr); % for now, reconstruct with all trials (can drop excluded trials later)
        
        trndata = data(trnidx,:);
        tstdata = data(tstidx,:);
        trnX    = X(trnidx,:);
        
        % C(n x k) x W(k x v) = R(n x v)
        % Where C is the n trials x k channels matrix of channel responses (trnx)
        % W is the k channels x v voxels weight matrix that we are trying to find
        % R is the n trials x v voxel response matrix (trndata)
        % To solve, use your favorite method for solving
        W_hat = pinv(trnX)*trndata;
        
        % C(n x k) = R(n x v) * pinv(W(k x v)
        chan_resp(tstidx,:) = chan_resp(tstidx,:)+tstdata*pinv(W_hat);
        % chan_resp(trnidx,:) = trndata*pinv(W_hat);
        clear w_hat trndata trnX tstdata trnidx tstidx;
    end
end

chan_resp = chan_resp / n_epoch;
end

