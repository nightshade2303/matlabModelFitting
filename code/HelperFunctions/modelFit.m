function [fit_nll, fit_pR2, fit_params, fit_bic, fit_aic, fit_st] = modelFit(algo, iters, param_bnds, choices, rewards, num_choices, optim, linear_env, rnd_flag)

if nargin < 7
    optim = 'bads';
end

if nargin < 8
    linear_env = 0;
end

if nargin < 9
    rnd_flag = 0;
end

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
    switch optim
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
[fit_nll, min_iter] = min(iter_nlls);
rnd_nll = calcChanceNLL(choices, num_choices, rnd_flag);
fit_pR2 = 1 - fit_nll/rnd_nll;
fit_params = iter_params(min_iter,:);
fit_st = st(min_iter,:);

n_free_params = sum(abs(param_bnds(2,:)-param_bnds(1,:)) > 0); % number of free params for the fit
fit_aic = 2*fit_nll + 2*n_free_params;
fit_bic = 2*fit_nll + log(numel(choices))*n_free_params;