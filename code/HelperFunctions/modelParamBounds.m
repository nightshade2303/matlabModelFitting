function [algo, param_bnds, param_names, free_params] = modelParamBounds(modelName)
%MODELPARAMBOUNDS return algo, upper and lower param bounds for model sim / fit

%% param bounds (lb;ub)

% General
rnd_b = [1e-2;2e1]; % rnd_w
s_b = [0;8];  % sty_w
sl_b = [1e-2;3]; % sty_l
g_b = [1e-2;3]; % lambda
u_b = [0;2]; % ucb_w

% Q-learning specific
qlr_b = [1e-2;1]; % lr_max
q0_b = [0.5;0.5]; % q0

% Beta-Bayes specific
blr_b = [1;1]; % lr for BetaBayes
a0_b = [1;1]; % a0
b0_b = [1;1]; % b0

% GP specific
t_b = [5;200]; % time_l

% Q-learning + Beta-Bayes
d_b = [0;1]; % decay

if contains(modelName,"Q")    

        % index         1       2       3      4    5     6      7       8
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];
        algo = 'Qlearning';

        param_bnds = zeros(2,length(param_names));
        % default params
        param_bnds(:,1) = qlr_b;
        param_bnds(:,4) = q0_b;
        param_bnds(:,6) = rnd_b;

        % optional params
        if contains(modelName,"g")
            param_bnds(:,2) = g_b;
        end
        if contains(modelName,"d")
            param_bnds(:,3) = d_b;
        end
        if contains(modelName,"u")
            param_bnds(:,5) = u_b;
        end
        if contains(modelName,"s")
            param_bnds(:,7) = s_b;
        end
        if contains(modelName,"l")
            param_bnds(:,8) = sl_b;
        end        

elseif contains(modelName,"B")

        % index         1       2       3     4    5     6      7     8        9
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];        
        algo = 'BayesUCB';

        param_bnds = zeros(2,length(param_names));
        % default params
        param_bnds(:,1) = blr_b;
        param_bnds(:,4) = a0_b;
        param_bnds(:,5) = b0_b;
        param_bnds(:,7) = rnd_b;

        % optional params
        if contains(modelName,"g")
            param_bnds(:,2) = g_b;
        end
        if contains(modelName,"d")
            param_bnds(:,3) = d_b;
        end
        if contains(modelName,"u")
            param_bnds(:,6) = u_b;
        end
        if contains(modelName,"s")
            param_bnds(:,8) = s_b;
        end
        if contains(modelName,"l")
            param_bnds(:,9) = sl_b;
        end        

elseif contains(modelName,"G")
    
        % index           1         2      3     4      5       6
        param_names = ["lambda","trials","ucb","beta","sty","sty-lambda"];
        algo = 'GPspace';

        param_bnds = zeros(2,length(param_names));
        % default params
        param_bnds(:,1) = g_b;
        param_bnds(:,4) = rnd_b;
        % optional params
        if contains(modelName,"d")
            param_bnds(:,2) = t_b;
            algo = 'GPspacetime';
        end
        if contains(modelName,"u")
            param_bnds(:,3) = u_b;
        end
        if contains(modelName,"s")
            param_bnds(:,5) = s_b;
        end
        if contains(modelName,"l")
            param_bnds(:,6) = sl_b;
        end
end

n_params = size(param_bnds,2);
free_params = param_bnds(1,:) ~= param_bnds(2,:);