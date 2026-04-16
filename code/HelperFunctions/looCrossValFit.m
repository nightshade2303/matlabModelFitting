function [fit_nll, fit_pR2, fit_params] = looCrossValFit(algo, iters, param_bnds, choices, rewards, num_choices, optim, linear_env, rnd_flag)

% leave one round out cross val fit

if nargin < 7
    optim = 'bads';
end

if nargin < 8
    linear_env = 0;
end

if nargin < 9
    rnd_flag = 0;
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

cv_rds = 1:n_rounds;    
rd_params = zeros(n_rounds,n_params);
test_nll = zeros(n_rounds,1); 

for rd = 1 : n_rounds % over cross-val folds
    % disp(['Fold ' num2str(rd) ' of ' num2str(n_rounds)]);     
    
    st = zeros(iters, n_params); % to store random starts
    iter_train_nlls = zeros(iters,1);
    iter_params = zeros(iters, n_params);
    for i = 1 : iters
        if iters==1 && size(param_bnds,1)>4 
            st(i,:) = param_bnds(5,:); % st params are provided by user
        else    
            st(i,:) = plb + (pub-plb).*rand(1,numel(plb)); % random start within plausible range
        end
        % disp(['Iter ' num2str(i) ' of ' num2str(iters)]);

        switch optim
            case 'bads'
                [iter_params(i,:), iter_train_nlls(i)] = bads(funNLL, st(i,:), lb, ub, plb, pub, [], options, ...
                                                        algo, choices(cv_rds~=rd,:), rewards(cv_rds~=rd,:), num_choices, linear_env);
            case 'fmin'
                [iter_params(i,:), iter_train_nlls(i)] = fminsearchbnd(funNLL, st(i,:), lb, ub, options, ...
                                                        algo, choices(cv_rds~=rd,:), rewards(cv_rds~=rd,:), num_choices, linear_env);
            case 'lbfgs'
                funNLL = @(params) calcNLL(params,algo,choices(cv_rds~=rd,:),rewards(cv_rds~=rd,:),num_choices,linear_env);
                [iter_params(i,:), iter_train_nlls(i)] = fmincon(funNLL, st(i,:), [], [], [], [], lb, ub, [], options);
        end
    end
    [~, best_iter] = min(iter_train_nlls);
    rd_params(rd,:) = iter_params(best_iter,:);
    test_nll(rd) = calcNLL(iter_params(best_iter,:), algo, choices(cv_rds==rd,:), rewards(cv_rds==rd,:), num_choices, linear_env);
end
fit_nll = sum(test_nll);
rnd_nll = calcChanceNLL(choices, num_choices, rnd_flag);
fit_pR2 = 1 - fit_nll/rnd_nll;
fit_params = mean(rd_params,1);    
