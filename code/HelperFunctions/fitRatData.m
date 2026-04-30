function rFitResult = fitRatData(rMABpre, rFitParamSet)

if ~isfield(rFitParamSet,'linear_env')
    rFitParamSet.linear_env = 1; % Set default value for linear_env if not provided
end

if ~isfield(rFitParamSet,'rnd_flag')
    rFitParamSet.rnd_flag = 0; % Set default value for rnd_flag if not provided
end

if ~isfield(rFitParamSet,'crossval')
    rFitParamSet.crossval = 'kfold'; % Set default value for crossval if not provided
end

rng(1, 'twister');
% n_subjects = numel(rFitParamSet.subjIndices);
weeks = 4:8;

% to fit first 100 trials of first block
atleast_trials = 100;
trials = 1:100;
sess_idx = find(ismember(rMABpre(subj).weekOfTraining, weeks) & rMABpre(subj).trialNums>=atleast_trials);
% sess_idx = find(rMABpre(subj).trialNums>=atleast_trials);
if size(sess_idx, 1)>1
    selected_blocks = sess_idx(any(rMABpre(subj).blockIDs(sess_idx, trials)'==1));
    rounds = {selected_blocks'};
else
    rounds = {};
end

% to fit all blocks, all trials
% trials = 1:size(rMABpre(subj).choices, 2);
% rounds = {1:size(rMABpre(subj).choices, 1)};

disp(['Rat ' num2str(rMABpre(subj).ratName) ', Task ' rFitParams.task]);

tic;
switch rFitParams.task
    case {'4ABunstr', '4ABstr'}
        nll = NaN(1,length(rounds));
        pR2 = NaN(1,length(rounds));
        params = NaN(1,length(rounds),size(rFitParams.param_bnds,2));
        
        for rd = 1 : length(rounds)
            % check number of sessions, should be more than number of folds
            if size(rounds{1}, 2) < rFitParams.kfolds
            % if size(rounds, 2)< rFitParams.kfolds
                disp("Number of sessions too low.");
                break;
            end
            
            [nll(1,rd), pR2(1,rd), params(1,rd,:)] = ...
                fitModel(rFitParamSet, ...
                rMABpre(subj).choices(rounds{rd}, trials), ...
                rMABpre(subj).rewards(rounds{rd}, trials), ...
                rMABpre(subj).narms);
        end

    % case '1D_Inter'
    %     nll = NaN(1,length(rounds),2); % dim 3: str/unstr
    %     pR2 = NaN(1,length(rounds),2); % dim 3: str/unstr
    %     params = NaN(1,length(rounds),size(hFitParams.param_bnds,2),2); % dim 4: str/unstr

    %     for rd = 1 : length(rounds)
    %         str_idx = hMABall(subj).structured(rounds{rd}) == 1; % identify structure rounds
    %         % str rounds
    %         [nll(1,rd,1), pR2(1,rd,1), params(1,rd,:,1)] = ...
    %             crossValFit(hFitParams.algo, hFitParams.kfolds, hFitParams.iters, hFitParams.param_bnds, ...
    %             hMABall(subj).choices(rounds{rd}(str_idx), hFitParams.trials), ...
    %             hMABall(subj).rewards(rounds{rd}(str_idx), hFitParams.trials), ...
    %             hMABall(subj).narms, hFitParams.optim, hFitParams.linear_env, hFitParams.rnd_flag);
    %         % ustr rounds
    %         [nll(1,rd,2), pR2(1,rd,2), params(1,rd,:,2)] = ...
    %             crossValFit(hFitParams.algo, hFitParams.kfolds, hFitParams.iters, hFitParams.param_bnds, ...
    %             hMABall(subj).choices(rounds{rd}(~str_idx), hFitParams.trials), ...
    %             hMABall(subj).rewards(rounds{rd}(~str_idx), hFitParams.trials), ...
    %             hMABall(subj).narms, hFitParams.optim, hFitParams.linear_env, hFitParams.rnd_flag);
    %     end
end
toc;

rFitResult = struct;

rFitResult.id = id;
rFitResult.subjIndex = subj;
rFitResult.nll = nll;
rFitResult.pR2 = pR2;
rFitResult.params = params;
rFitResult.ratName = rMABpre(subj).ratName;
% rFitResult.sess = rMABall(subj).session; % field does not exist
