function hFitResult = fitHumanData(hMABall, hFitParamSet)

if ~isfield(hFitParamSet,'linear_env')
    hFitParamSet.linear_env = 0; % Set default value for linear_env if not provided
end

if ~isfield(hFitParamSet,'rnd_flag')
    hFitParamSet.rnd_flag = 0; % Set default value for rnd_flag if not provided
end

if ~isfield(hFitParamSet,'crossval')
    hFitParamSet.crossval = 'kfold'; % Set default value for crossval if not provided
end

rng(1, 'twister');
rounds = hFitParamSet.rounds;
n_subjects = numel(hFitParamSet.subjIndices);

switch hFitParamSet.task
    case {'1D_Str','1D_Unstr','1D_Block','1D_ContBlock'}

        nll = NaN(n_subjects,length(rounds));
        pR2 = NaN(n_subjects,length(rounds));
        params = NaN(n_subjects,length(rounds),size(hFitParamSet.param_bnds,2));  

    switch hFitParamSet.crossval
        case 'none'
            aic = NaN(n_subjects,length(rounds));
            bic = NaN(n_subjects,length(rounds));
            for id = 1 : n_subjects % loop over subjects
                disp(['Subject ' num2str(id) ' of ' num2str(n_subjects)]);
                subj = hFitParamSet.subjIndices(id);
                for rd = 1 : length(rounds)
                    [nll(id,rd), pR2(id,rd), params(id,rd,:), bic(id,rd), aic(id,rd)] = ...
                        fitModel(hFitParamSet, hMABall(subj).choices(rounds{rd}, hFitParamSet.trials),...
                        hMABall(subj).rewards(rounds{rd}, hFitParamSet.trials), hMABall(subj).narms);               
                end
            end
        otherwise
            for id = 1 : n_subjects % loop over subjects
                disp(['Subject ' num2str(id) ' of ' num2str(n_subjects)]);
                subj = hFitParamSet.subjIndices(id);
                for rd = 1 : length(rounds)
                    [nll(id,rd), pR2(id,rd), params(id,rd,:)] = ...
                        fitModel(hFitParamSet, hMABall(subj).choices(rounds{rd}, hFitParamSet.trials),...
                        hMABall(subj).rewards(rounds{rd}, hFitParamSet.trials), hMABall(subj).narms);               
                end
            end
    end
        

    case '1D_Inter'

        nll = NaN(n_subjects,length(rounds),3); % dim 3: str/unstr/all
        pR2 = NaN(n_subjects,length(rounds),3); % dim 3: str/unstr/all
        params = NaN(n_subjects,length(rounds),size(hFitParamSet.param_bnds,2),3); % dim 4: str/unstr/all

        switch hFitParamSet.crossval 
            case 'none'
                aic = NaN(n_subjects,length(rounds),3); % dim 3: str/unstr/all
                bic = NaN(n_subjects,length(rounds),3); % dim 3: str/unstr/all

                for id = 1 : n_subjects % loop over subjects
                    disp(['Subject ' num2str(id) ' of ' num2str(n_subjects)]);
                    subj = hFitParamSet.subjIndices(id);
                    for rd = 1 : length(rounds)
                        str_idx = hMABall(subj).structured(rounds{rd}) == 1; % identify structure rounds
                        % str rounds
                        [nll(id,rd,1),pR2(id,rd,1),params(id,rd,:,1),bic(id,rd,1),aic(id,rd,1)] = ...
                            fitModel(hFitParamSet, hMABall(subj).choices(rounds{rd}(str_idx), hFitParamSet.trials),...
                            hMABall(subj).rewards(rounds{rd}(str_idx), hFitParamSet.trials), hMABall(subj).narms);   
                        % ustr rounds
                        [nll(id,rd,2),pR2(id,rd,2),params(id,rd,:,2),bic(id,rd,2),aic(id,rd,2)] = ...
                            fitModel(hFitParamSet, hMABall(subj).choices(rounds{rd}(~str_idx), hFitParamSet.trials),...
                            hMABall(subj).rewards(rounds{rd}(~str_idx), hFitParamSet.trials), hMABall(subj).narms);   
                        % all rounds
                        [nll(id,rd,3),pR2(id,rd,3),params(id,rd,:,3),bic(id,rd,3),aic(id,rd,3)] = ...
                            fitModel(hFitParamSet, hMABall(subj).choices(rounds{rd}, hFitParamSet.trials),...
                            hMABall(subj).rewards(rounds{rd}, hFitParamSet.trials), hMABall(subj).narms);   
                    end
                end        
            otherwise
                for id = 1 : n_subjects % loop over subjects
                    disp(['Subject ' num2str(id) ' of ' num2str(n_subjects)]);
                    subj = hFitParamSet.subjIndices(id);
                    for rd = 1 : length(rounds)
                        str_idx = hMABall(subj).structured(rounds{rd}) == 1; % identify structure rounds
                        % str rounds
                        [nll(id,rd,1), pR2(id,rd,1), params(id,rd,:,1)] = ...
                            fitModel(hFitParamSet, hMABall(subj).choices(rounds{rd}(str_idx), hFitParamSet.trials),...
                            hMABall(subj).rewards(rounds{rd}(str_idx), hFitParamSet.trials), hMABall(subj).narms);   
                        % ustr rounds
                        [nll(id,rd,2), pR2(id,rd,2), params(id,rd,:,2)] = ...
                            fitModel(hFitParamSet, hMABall(subj).choices(rounds{rd}(~str_idx), hFitParamSet.trials),...
                            hMABall(subj).rewards(rounds{rd}(~str_idx), hFitParamSet.trials), hMABall(subj).narms);   
                        % all rounds
                        [nll(id,rd,3), pR2(id,rd,3), params(id,rd,:,3)] = ...
                            fitModel(hFitParamSet, hMABall(subj).choices(rounds{rd}, hFitParamSet.trials),...
                            hMABall(subj).rewards(rounds{rd}, hFitParamSet.trials), hMABall(subj).narms);   
                    end
                end
        end
end

hFitResult = struct;

hFitResult.subjIndices = hFitParamSet.subjIndices;
hFitResult.nll = nll;
hFitResult.pR2 = pR2;
hFitResult.params = params;
hFitResult.sess = vertcat(hMABall(hFitParamSet.subjIndices).session);
switch hFitParamSet.crossval  
    case 'none'
        hFitResult.aic = aic;
        hFitResult.bic = bic;
end