function [fit_nll, fit_pR2, fit_params] = crossValFit(algo, kfolds, iters, param_bnds, choices, rewards, num_choices, optim, linear_env, rnd_flag)

if nargin < 8
    optim = 'bads';
end

if nargin < 9
    linear_env = 0;
end

if nargin < 10
    rnd_flag = 0;
end

if kfolds < 2
    disp('Error: Cross val folds under 2');
    return;
end


n_rounds = size(choices,1);

n_params = size(param_bnds,2);
lb = param_bnds(1,:);
ub = param_bnds(2,:);
plb = param_bnds(3,:);
pub = param_bnds(4,:);

funNLL = @(params,algo,choices,rewards,num_choices,linear_env) calcNLL(params,algo,choices,rewards,num_choices,linear_env);

switch optim
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


cv_k = cvpartition(n_rounds, "KFold", kfolds);

st = zeros(iters, n_params); % to store random starts
iter_nlls = zeros(iters,1);
for i = 1 : iters
    if iters==1 && size(param_bnds,1)>4 
        st(i,:) = param_bnds(5,:); % st params are provided by user
    else    
        st(i,:) = plb + (pub-plb).*rand(1,numel(plb)); % random start within plausible range
    end
    % disp(['Iter ' num2str(i) ' of ' num2str(iters)]);
    
    test_nll = zeros(kfolds,1); 
    for k = 1 : kfolds
        % disp(['Fold ' num2str(k) ' of ' num2str(kfolds)]);     
        switch optim
            case 'bads'
                i_params = bads(funNLL, st(i,:), lb, ub, plb, pub, [], options, ...
                                        algo, choices(cv_k.training(k),:), rewards(cv_k.training(k),:), num_choices, linear_env);
            case 'fmin'
                i_params = fminsearchbnd(funNLL, st(i,:), lb, ub, options, ...
                                        algo, choices(cv_k.training(k),:), rewards(cv_k.training(k),:), num_choices, linear_env);
            case 'lbfgs'
                funNLL = @(params) calcNLL(params,algo,choices(cv_k.training(k),:),rewards(cv_k.training(k),:),num_choices,linear_env);
                i_params = fmincon(funNLL, st(i,:), [], [], [], [], lb, ub, [], options);
        end

        test_nll(k) = calcNLL(i_params, algo, choices(cv_k.test(k),:), rewards(cv_k.test(k),:), num_choices, linear_env);
    end
    iter_nlls(i) = sum(test_nll);
end
[fit_nll, min_iter] = min(iter_nlls);

rnd_nll = calcChanceNLL(choices, num_choices, rnd_flag);
fit_pR2 = 1 - fit_nll/rnd_nll;

if nargout > 2
    switch optim
        case 'bads'
            fit_params = bads(funNLL, st(min_iter,:), lb, ub, plb, pub, [], options, ...
                                            algo, choices, rewards, num_choices, linear_env);
        case 'fmin'
            fit_params = fminsearchbnd(funNLL, st(min_iter,:), lb, ub, options, ...
                                            algo, choices, rewards, num_choices, linear_env);
        case 'lbfgs'
            funNLL = @(params) calcNLL(params,algo,choices,rewards,num_choices,linear_env);
            fit_params = fmincon(funNLL, st(i,:), [], [], [], [], lb, ub, [], options);
    end
end