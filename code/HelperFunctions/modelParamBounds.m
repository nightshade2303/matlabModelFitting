function [algo, param_bnds, param_names, free_params] = modelParamBounds(modelName)
%MODELPARAMBOUNDS return algo, upper and lower param bounds for model sim / fit

%% param bounds (lb;ub)

% General
rnd_b = [1e-2;2e1]; % rnd_w
s_b = [0;8];  % sty_w
sl_b = [1e-2;3]; % sty_l
slr_b = [0;1]; % sty_lr change
slr_c = [1;1]; % sty_lr constnat
g_b = [1e-2;3]; % lambda
u_b = [0;2]; % ucb_w
% prob_comb = [perms([80 46 15 10]); perms([46 80 46 15])];
m0_c = [0.4225;0.4225]; % m0 or q0 constant % change 
m0_b = [0.01;0.95]; % m0 or q0 bounds

% Q-learning specific
qlr_b = [1e-2;1]; % lr_max
% q0_b = [0.5;0.5]; % q0

% Beta-Bayes specific
blr_b = [1;1]; % lr for BetaBayes
% a0_b = [1;1]; % a0
% b0_b = [1;1]; % b0

% GP specific
t_b = [1e-2;1]; % time_l expressed as decay rate 
mem_b = [5000;5000]; % max memory for GP space

% SI specific
c = [-1;1];
gam = [0;1];

% Q-learning + Beta-Bayes
d_b = [0;1]; % decay

if contains(modelName,"Q")    

        % index         1       2       3      4    5     6      7       8          9
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lr","sty-lambda"];
        algo = 'Qlearning';

        param_bnds = zeros(2,length(param_names));
        % default params
        param_bnds(:,1) = qlr_b;
        param_bnds(:,4) = m0_c;
        param_bnds(:,6) = rnd_b;

        % optional params
        if contains(modelName,"g")
            param_bnds(:,2) = g_b;
        end
        if contains(modelName,"d")
            param_bnds(:,3) = d_b;
        end
        if contains(modelName,"m")
            param_bnds(:,4) = m0_b;
        end
        if contains(modelName,"u")
            param_bnds(:,5) = u_b;
        end
        if contains(modelName,"s")
            param_bnds(:,7) = s_b;
            param_bnds(:,8) = slr_c;
        end
        if contains(modelName, "h")
            param_bnds(:,7) = s_b;
            param_bnds(:,8) = slr_b;
        end
        if contains(modelName,"l")
            param_bnds(:,9) = sl_b;
        end 
        
elseif contains(modelName,"4")    

        % index         1       2       3      4    5     6      7     8    9       10   11
        param_names = ["lr","lambda","decay","q1", "q2", "q3", "q4","ucb","beta","sty","sty-lambda"];
        algo = 'Qlearningbiasq0';

        param_bnds = zeros(2,length(param_names));
        % default params
        param_bnds(:,1) = qlr_b;
        param_bnds(:,4) = m0_b;
        param_bnds(:,5) = m0_b;
        param_bnds(:,6) = m0_b;
        param_bnds(:,7) = m0_b;
        param_bnds(:,9) = rnd_b;

        % optional params
        if contains(modelName,"g")
            param_bnds(:,2) = g_b;
        end
        if contains(modelName,"d")
            param_bnds(:,3) = d_b;
        end
        if contains(modelName,"u")
            param_bnds(:,8) = u_b;
        end
        if contains(modelName,"s")
            param_bnds(:,10) = s_b;
        end
        if contains(modelName,"l")
            param_bnds(:,11) = sl_b;
        end 


elseif contains(modelName,"B")

        % index         1       2       3     4      5      6     7        8     9
        param_names = ["lr","lambda","decay","mu0","ucb","beta","sty", "sty_lr","sty-lambda"];        
        algo = 'BayesUCB';

        param_bnds = zeros(2,length(param_names));
        % default params
        param_bnds(:,1) = blr_b;
        param_bnds(:,4) = m0_c;
        param_bnds(:,6) = rnd_b;

        % optional params
        if contains(modelName,"g")
            param_bnds(:,2) = g_b;
        end
        if contains(modelName,"d")
            param_bnds(:,3) = d_b;
        end
        if contains(modelName,"m")
            param_bnds(:,4) = m0_b;
        end
        if contains(modelName,"u")
            param_bnds(:,5) = u_b;
        end
        if contains(modelName,"s")
            param_bnds(:,7) = s_b;
            param_bnds(:,8) = slr_c;
        end
        if contains(modelName, "h")
            param_bnds(:,7) = s_b;
            param_bnds(:,8) = slr_b;
        end
        if contains(modelName,"l")
            param_bnds(:,9) = sl_b;
        end        

elseif contains(modelName,"G")
    
        % index           1         2      3     4     5      6       7     8
        param_names = ["lambda","trials","mu0","ucb","beta","sty","sty_lr","sty-lambda"];
        algo = 'GPspace';

        param_bnds = zeros(2,length(param_names));
        % default params
        param_bnds(:,1) = g_b;
        param_bnds(:,5) = rnd_b;
        param_bnds(:,3) = m0_c;

        % optional params
        if contains(modelName,"d")
            param_bnds(:,2) = t_b;
            algo = 'GPspacetime';
        else
            param_bnds(:,2) = mem_b;
        end
        if contains(modelName,"m")
            param_bnds(:,3) = m0_b;
        end
        if contains(modelName,"u")
            param_bnds(:,4) = u_b;
        end
        if contains(modelName,"s")
            param_bnds(:,6) = s_b;
            param_bnds(:,7) = slr_c;
        end
        if contains(modelName, "h")
            param_bnds(:,6) = s_b;
            param_bnds(:,7) = slr_b;
        end
        if contains(modelName,"l")
            param_bnds(:,8) = sl_b;
        end

elseif contains(modelName,"I")
        % index          1      2          3     4   5      6         7 
        param_names = ["gamma","c","lambda","beta", "sty","sty_lr", "sty-lambda"];
        algo = 'StateInference';
        param_bnds = zeros(2,length(param_names));
        param_bnds(:, 1) = gam;
        param_bnds(:, 2) = c;
        param_bnds(:, 4) = rnd_b;
        
        if contains(modelName, "g")
            param_bnds(:,3) = g_b;
        end  
        if contains(modelName,"s")
            param_bnds(:,5) = s_b;
            param_bnds(:,6) = slr_c;
        end
        if contains(modelName, "h")
            param_bnds(:,5) = s_b;
            param_bnds(:,6) = slr_b;
        end
        if contains(modelName,"l")
            param_bnds(:,7) = sl_b;
        end
end

n_params = size(param_bnds,2);
free_params = param_bnds(1,:) ~= param_bnds(2,:);