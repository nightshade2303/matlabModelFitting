function [fit_nll, fit_pR2, fit_params] = nestedCrossValFit(algo, k_outfolds, k_infolds, iters, param_bnds, choices, rewards, num_choices)

n_rounds = size(choices,1);

n_params = size(param_bnds,2);
lb = param_bnds(1,:);
ub = param_bnds(2,:);
plb = param_bnds(3,:);
pub = param_bnds(4,:);

funNLL = @(params,algo,choices,rewards,num_choices) calcNLL(params,algo,choices,rewards,num_choices);

options = bads('defaults');  
options.Display = 'off';

cv_k_out = cvpartition(n_rounds, "KFold", k_outfolds);

for k_out = 1 : k_outfolds


    st = zeros(iters, n_params); % to store random starts
    iter_nlls = zeros(iters,1);
    for i = 1 : iters
        % disp(['Iter ' num2str(i) ' of ' num2str(iters)]);
        st(i,:) = plb + (pub-plb).*rand(1,numel(plb)); % random start within plausible range
        test_nll = zeros(k_infolds,1); 
        for k_in = 1 : k_infolds
            % disp(['Fold ' num2str(k) ' of ' num2str(kfolds)]);            
            [i_params, ~] = bads(funNLL, st(i,:), lb, ub, plb, pub, [], options, ...
                                    algo, choices(cv_k.training(k_in),:), rewards(cv_k.training(k_in),:), num_choices);
            test_nll(k_in) = calcNLL(i_params, algo, choices(cv_k.test(k_in),:), rewards(cv_k.test(k_in),:), num_choices);
        end
        iter_nlls(i) = sum(test_nll);
    end
    [fit_nll, min_iter] = min(iter_nlls);
end

rnd_nll = calcChanceNLL(choices, num_choices);
fit_pR2 = 1 - fit_nll/rnd_nll;

if nargout > 2
    
    fit_params = bads(funNLL, st(min_iter,:), lb, ub, plb, pub, [], options, ...
                                    algo, choices, rewards, num_choices);
end