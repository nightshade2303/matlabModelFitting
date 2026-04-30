function [fit_nll, fit_pR2, fit_params, fit_bic, fit_aic] = fitModel(fitParamSet, choices, rewards, num_choices)

if ~isfield(fitParamSet,'optim')
    fitParamSet.optim = 'bads';
end

if ~isfield(fitParamSet,'linear_env')
    fitParamSet.linear_env = 1;
end

if ~isfield(fitParamSet,'rnd_flag')
    fitParamSet.rnd_flag = 0;
end

fit_aic = [];
fit_bic = [];

% Unpack fit params
iters = fitParamSet.iters;
algo = fitParamSet.algo;
n_rounds = size(choices,1);
if strcmp(fitParamSet.crossval, 'looc')
    kfolds = n_rounds;
else
    kfolds = min([fitParamSet.kfolds, n_rounds]); % kfolds should not exceed num of rounds
end
linear_env = fitParamSet.linear_env;

param_bnds = fitParamSet.param_bnds;
n_params = size(param_bnds,2);
lb = param_bnds(1,:);
ub = param_bnds(2,:);
plb = param_bnds(3,:);
pub = param_bnds(4,:);

funNLL = @(params,algo,choices,rewards,num_choices,linear_env) calcNLL(params,algo,choices,rewards,num_choices,linear_env);

switch fitParamSet.optim
    case 'bads'
        options = bads('defaults');  
        options.Display = 'off';
    case 'fmin'
        options = optimset('fminsearch');
        % options.MaxFunEvals = 500*size(param_bnds,2);
        options.Display = 'off';
    case 'lbfgs'
        options = optimoptions('fmincon', ...
                    'Algorithm', 'interior-point', ...
                    'HessianApproximation', 'lbfgs', ...
                    'MaxIterations', 500, ...
                    'Display', 'off');        
end

rnd_nll = calcChanceNLL(choices, num_choices, fitParamSet.rnd_flag);

switch fitParamSet.crossval
    case 'none'
        % no cross-validation
        st = zeros(iters, n_params); % to store random starts
        iter_nlls = zeros(iters,1);
        iter_params = zeros(iters,n_params);
        for i = 1 : iters
            if iters==1 && size(param_bnds,1)>4 
                st(i,:) = param_bnds(5,:); % st params are provided by user
            else    
                st(i,:) = plb + (pub-plb).*rand(1,numel(plb)); % random start within plausible range
            end
        
            % disp(['Iter ' num2str(i) ' of ' num2str(iters)]);
            switch fitParamSet.optim
                case 'bads'
                    iter_params(i,:) = bads(funNLL, st(i,:), lb, ub, plb, pub, [], options, ...
                                            algo, choices, rewards, num_choices, linear_env);
                case 'fmin'
                    iter_params(i,:) = fminsearchbnd(funNLL, st(i,:), lb, ub, options, ...
                                            algo, choices, rewards, num_choices, linear_env);
                case 'lbfgs'
                    funNLL = @(params) calcNLL(params,algo,choices,rewards,num_choices,linear_env);
                    iter_params(i,:) = fmincon(funNLL, st(i,:), [], [], [], [], lb, ub, [], options);
            end
        
            iter_nlls(i) = calcNLL(iter_params(i,:), algo, choices, rewards, num_choices, linear_env);    
        end
        [fit_nll, best_iter] = min(iter_nlls);
        fit_pR2 = 1 - fit_nll/rnd_nll;
        fit_params = iter_params(best_iter,:);
        fit_st = st(best_iter,:);
        
        n_free_params = sum(param_bnds(2,:)~=param_bnds(1,:)); % number of free params for the fit
        fit_aic = 2*fit_nll + 2*n_free_params;
        fit_bic = 2*fit_nll + log(numel(choices))*n_free_params;

    otherwise
        % cross validation
        cv_k = cvpartition(n_rounds, "KFold", kfolds);  
        k_params = zeros(kfolds,n_params);
        test_nll = zeros(kfolds,1); 

        for k = 1 : kfolds
            st = zeros(iters, n_params); % to store random starts
            iter_train_nlls = zeros(iters,1);
            iter_params = zeros(iters, n_params);

            for i = 1 : iters
                if iters==1 && size(param_bnds,1)>4 
                    st(i,:) = param_bnds(5,:); % st params are provided by user
                else    
                    st(i,:) = plb + (pub-plb).*rand(1,numel(plb)); % random start within plausible range
                end

                switch fitParamSet.optim
                    case 'bads'
                        [iter_params(i,:), iter_train_nlls(i)] = bads(funNLL, st(i,:), lb, ub, plb, pub, [], options, ...
                                  algo, choices(cv_k.training(k),:), rewards(cv_k.training(k),:), num_choices, linear_env);
                    case 'fmin'
                        [iter_params(i,:), iter_train_nlls(i)] = fminsearchbnd(funNLL, st(i,:), lb, ub, options, ...
                                  algo, choices(cv_k.training(k),:), rewards(cv_k.training(k),:), num_choices, linear_env);
                    case 'lbfgs'
                        funNLL = @(params) calcNLL(params,algo,choices(cv_k.training(k),:),rewards(cv_k.training(k),:),num_choices,linear_env);
                        [iter_params(i,:), iter_train_nlls(i)] = fmincon(funNLL, st(i,:), [], [], [], [], lb, ub, [], options);
                end
            end
            [~, best_iter] = min(iter_train_nlls);
            k_params(k,:) = iter_params(best_iter,:);
            test_nll(k) = calcNLL(iter_params(best_iter,:), algo, choices(cv_k.test(k),:), rewards(cv_k.test(k),:), num_choices, linear_env);          
        end
        fit_nll = sum(test_nll);        
        fit_pR2 = 1 - fit_nll/rnd_nll;
        fit_params = mean(k_params,1); 
        % fit_params = median(k_params,1); % alternative

        if nargout > 3 % calculate aic / bic based on averaged best fit params
            nll_for_abic = calcNLL(fit_params, algo, choices, rewards, num_choices, linear_env); 
            n_free_params = sum(param_bnds(2,:)~=param_bnds(1,:)); % number of free params for the fit
            fit_aic = 2*nll_for_abic + 2*n_free_params;
            fit_bic = 2*nll_for_abic + log(numel(choices))*n_free_params;
        end
end