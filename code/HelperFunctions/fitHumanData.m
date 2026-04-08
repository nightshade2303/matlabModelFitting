function hFitResult = fitHumanData(hMABall, hFitParams, id)

subj = hFitParams.subjIndices(id);
rounds = hFitParams.rounds;

if ~isfield(hFitParams,'linear_env')
    hFitParams.linear_env = 0; % Set default value for linear_env if not provided
end

rng(1, 'twister');

disp(['Subject ' num2str(hMABall(subj).subjectID) ', Task ' hFitParams.task]);

tic;
switch hFitParams.task
    case {'1D_Str','1D_Unstr','1D_Block','1D_ContBlock'}
        nll = NaN(1,length(rounds));
        pR2 = NaN(1,length(rounds));
        params = NaN(1,length(rounds),size(hFitParams.param_bnds,2));
        
        for rd = 1 : length(rounds)
            [nll(1,rd), pR2(1,rd), params(1,rd,:)] = ...
                crossValFit(hFitParams.algo, hFitParams.kfolds, hFitParams.iters, hFitParams.param_bnds, ...
                hMABall(subj).choices(rounds{rd}, hFitParams.trials), ...
                hMABall(subj).rewards(rounds{rd}, hFitParams.trials), ...
                hMABall(subj).narms, hFitParams.optim, hFitParams.linear_env, hFitParams.rnd_flag);
        end

    case '1D_Inter'
        nll = NaN(1,length(rounds),2); % dim 3: str/unstr
        pR2 = NaN(1,length(rounds),2); % dim 3: str/unstr
        params = NaN(1,length(rounds),size(hFitParams.param_bnds,2),2); % dim 4: str/unstr

        for rd = 1 : length(rounds)
            str_idx = hMABall(subj).structured(rounds{rd}) == 1; % identify structure rounds
            % str rounds
            [nll(1,rd,1), pR2(1,rd,1), params(1,rd,:,1)] = ...
                crossValFit(hFitParams.algo, hFitParams.kfolds, hFitParams.iters, hFitParams.param_bnds, ...
                hMABall(subj).choices(rounds{rd}(str_idx), hFitParams.trials), ...
                hMABall(subj).rewards(rounds{rd}(str_idx), hFitParams.trials), ...
                hMABall(subj).narms, hFitParams.optim, hFitParams.linear_env, hFitParams.rnd_flag);
            % ustr rounds
            [nll(1,rd,2), pR2(1,rd,2), params(1,rd,:,2)] = ...
                crossValFit(hFitParams.algo, hFitParams.kfolds, hFitParams.iters, hFitParams.param_bnds, ...
                hMABall(subj).choices(rounds{rd}(~str_idx), hFitParams.trials), ...
                hMABall(subj).rewards(rounds{rd}(~str_idx), hFitParams.trials), ...
                hMABall(subj).narms, hFitParams.optim, hFitParams.linear_env, hFitParams.rnd_flag);
        end
end
toc;

hFitResult = struct;

hFitResult.id = id;
hFitResult.subjIndex = subj;
hFitResult.nll = nll;
hFitResult.pR2 = pR2;
hFitResult.params = params;
hFitResult.subjectID = hMABall(subj).subjectID;
hFitResult.sess = hMABall(subj).session;
