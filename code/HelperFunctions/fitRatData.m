function rFitResult = fitRatData(rMABpre, rFitParams, id)

if ~isfield(rFitParams,'linear_env')
    rFitParams.linear_env = 1; % Set default value for linear_env if not provided
end

if ~isfield(rFitParams,'rnd_flag')
    rFitParams.rnd_flag = 0; % Set default value for rnd_flag if not provided
end

if ~isfield(rFitParams,'crossval')
    rFitParams.crossval = 'kfold'; % Set default value for crossval if not provided
end

if ~isfield(rFitParams,'fitType')
    rFitParams.fitType = 'expert'; % 'expert' (days > 30) or 'daywise' (groups of days)
end

switch rFitParams.fitType
    case 'expert'
        def_atleast = 300; % min session length
        def_max = 300;     % fit sessions up to this trial
    case 'daywise'
        def_atleast = 100;
        def_max = 300;
        if ~isfield(rFitParams,'dayGroups')
            rFitParams.dayGroups = {1:7, 8:14, 15:21, 22:28}; % days of training fit together
        end
end

if ~isfield(rFitParams,'atleast_trials')
    rFitParams.atleast_trials = def_atleast;
end

if ~isfield(rFitParams,'max_trials')
    rFitParams.max_trials = def_max;
end

subj = rFitParams.subjIndices(id);
rng(1, 'twister');
% n_subjects = numel(rFitParams.subjIndices);
% weeks = 4:8;

% to fit first 100 trials of first block
% atleast_trials = 100;
% trials = 1:100;
% % sess_idx = find(ismember(rMABpre(subj).weekOfTraining, weeks) & rMABpre(subj).trialNums>=atleast_trials);
% sess_idx = find(rMABpre(subj).weekOfTraining>=4 & rMABpre(subj).trialNums>=atleast_trials);

% % sess_idx = find(rMABpre(subj).trialNums>=atleast_trials);
% if size(sess_idx, 1)>1
%     selected_blocks = sess_idx(any(rMABpre(subj).blockIDs(sess_idx, trials)'==1)); % select specific blocks
%     rounds = {selected_blocks'};
% else
%     rounds = {};
% end

% expert: all days > 30 fit together; daywise: each group of training days fit together
% sessions with atleast_trials, fit up to max_trials
atleast_trials = rFitParams.atleast_trials;
trials = 1:min(rFitParams.max_trials, size(rMABpre(subj).choices, 2));
switch rFitParams.fitType
    case 'expert'
        day_groups = {unique(rMABpre(subj).dayOfTraining(rMABpre(subj).dayOfTraining > 30))'};
    case 'daywise'
        day_groups = rFitParams.dayGroups;
end
n_groups = numel(day_groups);

rounds = cell(1, n_groups);
sess_per_day = cell(1, n_groups); % number of sessions fit on each day of the group
for g = 1 : n_groups
    sess_idx = find(ismember(rMABpre(subj).dayOfTraining, day_groups{g}) & rMABpre(subj).trialNums>=atleast_trials);

    % exclude sessions with a NaN reward or choice within their own trials (shorter sessions are NaN-padded)
    in_sess = trials <= min(rMABpre(subj).trialNums(sess_idx), trials(end));
    nan_sess = any(in_sess & (isnan(rMABpre(subj).rewards(sess_idx, trials)) | isnan(rMABpre(subj).choices(sess_idx, trials))), 2);
    if any(nan_sess)
        disp(['Group ' num2str(g) ': excluding ' num2str(sum(nan_sess)) ' of ' num2str(numel(sess_idx)) ' sessions with NaN in first ' num2str(trials(end)) ' trials']);
    end
    sess_idx = sess_idx(~nan_sess);

    rounds{g} = sess_idx;
    sess_per_day{g} = sum(rMABpre(subj).dayOfTraining(sess_idx) == day_groups{g}(:)', 1);
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
        
        switch rFitParams.crossval
            case 'none'
                aic = NaN(1, length(rounds));
                bic = NaN(1, length(rounds));

                for rd = 1 : length(rounds)
                    if isempty(rounds{rd})
                        continue; % no sessions in this group, leave as NaN
                    end

                    [nll(1,rd), pR2(1,rd), params(1,rd,:), bic(1,rd), aic(1,rd)] = ...
                        fitModel(rFitParams, ...
                        rMABpre(subj).choices(rounds{rd}, trials), ...
                        rMABpre(subj).rewards(rounds{rd}, trials), ...
                        rMABpre(subj).narms);
                end

            otherwise
                for rd = 1 : length(rounds)
                    % check number of sessions, should be more than number of folds
                    if numel(rounds{rd}) < rFitParams.kfolds
                    % if size(rounds, 2)< rFitParams.kfolds
                        disp("Number of sessions too low.");
                        continue;
                    end
                    
                    [nll(1,rd), pR2(1,rd), params(1,rd,:)] = ...
                        fitModel(rFitParams, ...
                        rMABpre(subj).choices(rounds{rd}, trials), ...
                        rMABpre(subj).rewards(rounds{rd}, trials), ...
                        rMABpre(subj).narms);
                end
        end
end
toc;

rFitResult = struct;

rFitResult.id = id;
rFitResult.subjIndex = subj;
rFitResult.nll = nll;
rFitResult.pR2 = pR2;
rFitResult.params = params;
% doesnt work if kfold skips sessions:
rFitResult.nTrials = sum(~isnan(rMABpre(subj).choices(rounds{1}, trials)), 'all'); % exact number of trials fit
rFitResult.nTrialsGroup = cellfun(@(r) sum(~isnan(rMABpre(subj).choices(r, trials)), 'all'), rounds); % trials fit per group
rFitResult.fitType = rFitParams.fitType;
rFitResult.dayGroups = day_groups;
rFitResult.sessPerDay = sess_per_day; % sessions fit per day, aligned with dayGroups
rFitResult.nSess = cellfun(@numel, rounds); % sessions fit per group
rFitResult.sessIdx = rounds;
rFitResult.ratName = rMABpre(subj).ratName;
% rFitResult.sess = rMABall(subj).session; % field does not exist
switch rFitParams.crossval  
    case 'none'
        rFitResult.aic = aic;
        rFitResult.bic = bic;
end