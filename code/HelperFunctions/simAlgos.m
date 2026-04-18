function [choices, rewards, rew_probs, latent_vars] = simAlgos(algo, params, n_rounds, n_trials, rprobset, linear_env)

if nargin < 6
    linear_env = 0;
end

num_choices = size(rprobset,2);

choices = zeros(n_rounds, n_trials);
rewards = zeros(n_rounds, n_trials);

% assign rew probs to rounds
rew_probs = rprobset(randi(size(rprobset,1),n_rounds,1),:);

% pre-define random number array
rand_nums = rand(n_rounds, n_trials, 2); % pre-define random numbers for choice and reward determination

switch algo
    case 'GPspace'
        %% Gaussian Process Classification over space with optional UCB-2, stickyness, forgetting
        
        % unpack params
        gen_l = params(1); % length scale of GP
        time_l = floor(params(2)); % length of memory window (is an integer)
        ucb_w = params(3); % weight of ucb param
        rnd_w = params(4); % weight of random exploration (inverse softmax temp)
        sty_w = params(5); % weight of sticky param
        sty_l = params(6); % length scale of sticky process 

        % GP settings
        meanfunc = @meanConst;        
        likfunc = @likLogistic;
        infmethod = @infLaplace;

        % gp constants
        sig_sd = 1;
        mean_const = 0;
        
        choice_vec = 1:num_choices;
        if linear_env == 0
            covfunc = {@covDiscrete, num_choices};
            % Set up discrete cov kernel function: SE for circular space
            [~, L] = covfunc_1Denv(num_choices, gen_l, sig_sd);
            hyp = struct('mean', mean_const, 'cov', L);
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            covfunc = @covSEiso;
            hyp = struct('mean', mean_const, 'cov', log([gen_l, sig_sd]));
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        latent_vars = zeros(n_rounds, n_trials, num_choices, 2); % mu and sd of gaussian process
                
        for rd = 1 : n_rounds
            choices(rd, 1) = ceil(rand_nums(rd,1,1)*num_choices); % make a random choice on the first trial
            rewards(rd, 1) = 2*(rand_nums(rd,1,2) < rew_probs(rd, choices(rd,1)))-1; % determine reward based on probability

            [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, [], [], (1:num_choices)');
            mu = 1./(1+exp(-lmu));
            ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
            sd = ucb-mu;
            latent_vars(rd,1,:,1) = mu; latent_vars(rd,1,:,2) = sd;

            % loop over trials
            for tr = 2 : n_trials
                [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, ...
                        choices(rd,max([1,tr-time_l]):tr-1)', rewards(rd,max([1,tr-time_l]):tr-1)', ...
                        (1:num_choices)');

                mu = 1./(1+exp(-lmu));
                ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
                sd = ucb-mu;

                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = sd;
                
                % sticky
                if sty_l > 0
                    sticky_vec = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*sty_l^2)) * sty_w;
                else
                    sticky_vec = zeros(1,num_choices);
                    sticky_vec(choices(rd,tr-1)) = sty_w;
                end
                
                % softmax
                choice_pref = (mu + ucb_w*sd)*rnd_w + sticky_vec;
                choice_pref = choice_pref - max(choice_pref); % to avoid overflow errors in softmax
                choice_probs = exp(choice_pref) / sum(exp(choice_pref)); % softmax

                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0,cumsum(choice_probs)]),1); % make a random choice for the current trial
                rewards(rd, tr) = 2*(rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)))-1; % determine reward based on probability                
            end
        end

        rewards(rewards<0) = 0;

    case 'GPspacetime'
        %% Gaussian Process Classification over space (RBF) and trials (RBF - forgetting process) with optional UCB-2, stickyness

        % unpack params
        gen_l = params(1); % length scale of GP
        time_l = params(2); % time scale of GP
        ucb_w = params(3); % weight of ucb2 param
        rnd_w = params(4); % weight of random exploration (inverse softmax temp)
        sty_w = params(5); % weight of sticky param
        sty_l = params(6); % length scale of sticky process    

        % GP settings
        meanfunc = @meanConst;
        likfunc = @likLogistic;
        infmethod = @infLaplace;
        
        % gp constants
        sig_sd = 1;
        mean_const = 0;

        choice_vec = 1:num_choices;
        if linear_env == 0
            covfunc = @cov_SEcircSpaceTime;
            hyp = struct('mean', mean_const, 'cov', [gen_l, num_choices, time_l, sig_sd]);
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            covfunc = @covSEard;
            hyp = struct('mean', mean_const, 'cov', log([gen_l, time_l, sig_sd]));
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        latent_vars = zeros(n_rounds, n_trials, num_choices, 2); % mu and sd of gaussian process
                
        for rd = 1 : n_rounds
            % make a random choice on the first trial
            choices(rd, 1) = ceil(rand_nums(rd,1,1)*num_choices); 
            rewards(rd, 1) = 2*(rand_nums(rd,1,2) < rew_probs(rd,choices(rd,1)))-1; % determine reward based on probability
            [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, [], [], ...
                [(1:num_choices)', ones(num_choices,1)]);
            mu = 1./(1+exp(-lmu));
            ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
            sd = ucb-mu;
            latent_vars(rd,1,:,1) = mu; latent_vars(rd,1,:,2) = sd; 

            % loop over trials
            for tr = 2 : n_trials
                [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, ...
                    [choices(rd,1:tr-1)', (1:tr-1)'], rewards(rd,1:tr-1)', ...
                    [(1:num_choices)', ones(num_choices,1)*tr]);

                mu = 1./(1+exp(-lmu));
                ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
                sd = ucb-mu;

                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = sd; 
                
                % sticky
                if sty_l > 0
                    sticky_vec = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*sty_l^2)) * sty_w;
                else
                    sticky_vec = zeros(1,num_choices);
                    sticky_vec(choices(rd,tr-1)) = sty_w;
                end
                
                % softmax
                choice_pref = (mu + ucb_w*sd)*rnd_w + sticky_vec;
                choice_pref = choice_pref - max(choice_pref); % to avoid overflow errors in softmax
                choice_probs = exp(choice_pref) / sum(exp(choice_pref)); % softmax

                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0,cumsum(choice_probs)]),1); % make a choice for the current trial
                rewards(rd, tr) = 2*(rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)))-1; % determine reward based on probability                
            end
        end
        
        rewards(rewards<0) = 0;


    case 'Qlearning'
        %% Q-learning with optional generalisation, UCB-1, stickyness, Q-decay

        choice_vec = 1:num_choices;
        if linear_env == 0
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        % unpack params
        lr_max = params(1); % learning rate
        gen_l = params(2);  % length_scale parameter
        q_decay = params(3);% decay of q values towards 0
        q0 = params(4);     % initial Q-value
        ucb_w = params(5);  % weight of ucb1 param
        rnd_w = params(6);  % weight of random exploration (inverse softmax temp)
        sty_w = params(7);  % weight of sticky param
        sty_l = params(8);  % length scale of sticky process    

        latent_vars = zeros(n_rounds, n_trials, num_choices, 2); % q values and ucb-1 term

        for rd = 1 : n_rounds
            q_a = zeros(1,num_choices)+q0; % initialise q-values to q0
            n_a = zeros(1,num_choices)+1; % choice sample counter
            latent_vars(rd,1,:,1) = q_a; latent_vars(rd,1,:,2) = 0;

            % make a random choice on the first trial
            choices(rd, 1) = ceil(rand_nums(rd,1,1)*num_choices); 
            rewards(rd, 1) = rand_nums(rd,1,2) < rew_probs(rd,choices(rd,1)); % determine reward based on probability

            for tr = 2 : n_trials
                prev_choice = zeros(1,num_choices);
                prev_choice(choices(rd,tr-1)) = 1;

                rpe = (rewards(rd,tr-1) - q_a(choices(rd,tr-1)));

                % q values decay towards 0 (except previously chosen action)
                q_a(prev_choice==0) = q_a(prev_choice==0) - q_decay*(q_a(prev_choice==0)-0); 

                % % q_a = q_a - q_decay*(q_a-0);       % all q-values decay                              
                                
                % Q learning    
                if gen_l > 0
                    lr_q = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*gen_l^2)) * lr_max;                    
                    q_a = q_a + lr_q * rpe;
                else
                    q_a(choices(rd,tr-1)) = q_a(choices(rd,tr-1)) + lr_max * rpe;
                end        
                
                clip(q_a, 0, 1);     

                n_a(choices(rd,tr-1)) = n_a(choices(rd,tr-1))+1;
                u_a = sqrt(2*log(tr-1) ./ n_a);

                latent_vars(rd,tr,:,1) = q_a; latent_vars(rd,tr,:,2) = u_a;        
                
                if sty_l > 0
                    sticky_vec = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*sty_l^2)) * sty_w;
                else
                    sticky_vec = zeros(1,num_choices);
                    sticky_vec(choices(rd,tr-1)) = sty_w;
                end

                choice_pref = (q_a + ucb_w*u_a)*rnd_w + sticky_vec;
                choice_pref = choice_pref - max(choice_pref);  % to avoid overflow errors in softmax
                choice_probs = exp(choice_pref) / sum(exp(choice_pref));  % softmax
                choice_probs(choice_probs==0) = eps;

                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0,cumsum(choice_probs)]),1); % make a choice for the current trial
                rewards(rd, tr) = rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)); % determine reward based on probability 
            end
        end

    case 'BayesUCB'
        %% Bayesian option learning plus softmax with optional generalisation, UCB-2, stickyness, Q-decay

        % unpack params
        lr = params(1); % learning rate
        gen_l = params(2); % generalisation parameter
        b_decay = params(3); % decay rate
        a0 = params(4); % initial a-value for beta distr
        b0 = params(5); % initial b-value for beta distr
        ucb_w = params(6); % weight of ucb param
        rnd_w = params(7); % weight of random exploration (inverse softmax temp)
        sty_w = params(8); % weight of sticky param        
        sty_l = params(9);  % length scale of sticky process    

        choice_vec = 1:num_choices;
        if linear_env == 0
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        latent_vars = zeros(n_rounds, n_trials, num_choices, 2); % beta distr mean and sd

        for rd = 1 : n_rounds
            S = zeros(1,num_choices);
            F = zeros(1,num_choices);
            
            beta_m0 = a0./(a0+b0); beta_sd0 = sqrt(beta_m0.*(1-beta_m0)./(a0+b0+1));
            latent_vars(rd,1,:,1) = beta_m0; latent_vars(rd,1,:,2) = beta_sd0;

            % make a random choice on the first trial
            choices(rd, 1) = ceil(rand_nums(rd,1,1)*num_choices); 
            rewards(rd, 1) = rand_nums(rd,1,2) < rew_probs(rd,choices(rd,1)); % determine reward based on probability
        
            for tr = 2 : n_trials
                prev_choice = zeros(1,num_choices);
                prev_choice(choices(rd,tr-1)) = 1;

                % S/F decay (except previously chosen action)
                S(prev_choice==0) = S(prev_choice==0)*(1-b_decay);
                F(prev_choice==0) = F(prev_choice==0)*(1-b_decay);
                
                if gen_l > 0
                    lr_b = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*gen_l^2)) * lr;  
                    S = S + lr_b*rewards(rd,tr-1);
                    F = F + lr_b*(1-rewards(rd,tr-1));
                else
                    S(choices(rd,tr-1)) = S(choices(rd,tr-1)) + lr*rewards(rd,tr-1);
                    F(choices(rd,tr-1)) = F(choices(rd,tr-1)) + lr*(1-rewards(rd,tr-1));        
                end
                clip(S, 0, Inf);
                clip(F, 0, Inf);

                as = a0 + S;
                bs = b0 + F;
        
                beta_m = as./(as+bs);
                beta_sd = sqrt(beta_m.*(1-beta_m)./(as+bs+1));

                latent_vars(rd,tr,:,1) = beta_m; latent_vars(rd,tr,:,2) = beta_sd;

                if sty_l > 0
                    sticky_vec = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*sty_l^2)) * sty_w;
                else
                    sticky_vec = zeros(1,num_choices);
                    sticky_vec(choices(rd,tr-1)) = sty_w;
                end
        
                choice_pref = (beta_m + ucb_w*beta_sd)*rnd_w + sticky_vec;
                choice_pref = choice_pref - max(choice_pref);
                choice_probs = exp(choice_pref) / sum(exp(choice_pref));
                choice_probs(choice_probs==0) = eps;

                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0,cumsum(choice_probs)]),1); % make a choice for the current trial
                rewards(rd, tr) = rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)); % determine reward based on probability 
            end
        end

    case 'WSLS'
        %% win stay lose shift

        choice_vec = 1:num_choices;
        if linear_env == 0
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        % unpack params
        p_switch_r = params(1); % switch prob after reward
        p_switch_n = params(2); % switch prob after no-reward
        sd_r = params(3); % std dev of distance moved after reward
        sd_n = params(4); % std dev of distance moved after no-reward

        latent_vars = [];

        for rd = 1 : n_rounds
            % make a random choice on the first trial
            choices(rd, 1) = ceil(rand_nums(rd,1,1)*num_choices); 
            rewards(rd, 1) = rand_nums(rd,1,2) < rew_probs(rd,choices(rd,1)); % determine reward based on probability

            for tr = 2 : n_trials
               if rewards(rd,tr-1) > 0  % rew
                    a_stay = zeros(1,num_choices);
                    a_stay(choices(rd,tr-1)) = 1;
        
                    a_switch = normpdf(dist_mat(choices(rd,tr-1),:),0,sd_r);
                    a_switch(choices(rd,tr-1)) = 0; 
                    a_switch(a_switch==0) = eps;
                    a_switch = a_switch/sum(a_switch);
        
                    choice_probs = a_stay*(1-p_switch_r) + a_switch*p_switch_r;
        
                else            % no-rew
                    a_stay = zeros(1,num_choices);
                    a_stay(choices(rd,tr-1)) = 1;
        
                    a_switch = normpdf(dist_mat(choices(rd,tr-1),:),0,sd_n);
                    a_switch(choices(rd,tr-1)) = 0; 
                    a_switch(a_switch==0) = eps;
                    a_switch = a_switch/sum(a_switch);
        
                    choice_probs = a_stay*(1-p_switch_n) + a_switch*p_switch_n;
                end
                choice_probs(choice_probs == 0) = eps;
                
                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0,cumsum(choice_probs)]),1); % make a choice for the current trial
                rewards(rd, tr) = rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)); % determine reward based on probability 

            end
        end

end