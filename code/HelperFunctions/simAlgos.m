
function [choices, rewards, rew_probs, latent_vars] = simAlgos(algo, params, n_rounds, n_trials, rprobset)

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
        use_latent = 0;   
        
        % unpack params
        length_scale = params(1); % length scale of GP
        ucb_w = params(2); % weight of ucb param
        rnd_w = params(3); % weight of random exploration (inverse softmax temp)
        sty_w = params(4); % weight of sticky param
        mem_l = floor(params(5)); % length of memory window (is an integer)

        % GP settings
        meanfunc = @meanConst;
        covfunc = {@covDiscrete, num_choices};
        likfunc = @likLogistic;
        infmethod = @infLaplace;

        % gp constants
        sig_sd = 1;
        mean_const = 0;
        
        % Set up discrete cov kernel function: SE for circular space
        [~, L] = covfunc_1Denv(num_choices, length_scale, sig_sd);
        hyp = struct('mean', mean_const, 'cov', L);

        latent_vars = zeros(n_rounds, n_trials, num_choices, 2); % mu and sd of gaussian process
                
        for rd = 1 : n_rounds
            choices(rd, 1) = ceil(rand_nums(rd,1,1)*num_choices); % make a random choice on the first trial
            rewards(rd, 1) = rand_nums(rd,1,2) < rew_probs(rd, choices(rd,1)); % determine reward based on probability
            if use_latent == 1
                [~, ~, mu, s2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, [], [], (1:num_choices)');
            else
                [mu, s2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, [], [], (1:num_choices)');
            end
            latent_vars(rd,1,:,1) = mu; latent_vars(rd,1,:,2) = s2; 

            % loop over trials
            for tr = 2 : n_trials
                if use_latent == 1
                    [~, ~, mu, s2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, ...
                        choices(rd,max([1,tr-mem_l]):tr-1)', rewards(rd,max([1,tr-mem_l]):tr-1)', ...
                        (1:num_choices)');
                else
                    [mu, s2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, ...
                        choices(rd,max([1,tr-mem_l]):tr-1)', rewards(rd,max([1,tr-mem_l]):tr-1)', ...
                        (1:num_choices)');
                end
                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = s2; 
                
                % sticky
                prev_choice = zeros(num_choices,1);
                prev_choice(choices(rd,tr-1)) = 1;
                
                % softmax
                choice_pref = (mu + ucb_w*sqrt(s2))*rnd_w + prev_choice*sty_w;
                choice_pref = choice_pref - max(choice_pref); % to avoid overflow errors in softmax
                choice_probs = exp(choice_pref) / sum(exp(choice_pref)); % softmax

                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0;cumsum(choice_probs)]),1); % make a random choice for the current trial
                rewards(rd, tr) = rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)); % determine reward based on probability                
            end
        end



    case 'GPspacetime'
        %% Gaussian Process Classification over space (RBF) and trials (RBF - forgetting process) with optional UCB-2, stickyness

        use_latent = 0; % if 1, use mean and sd of latent GP instead of logistic transform
        
        % unpack params
        length_scale = params(1); % length scale of GP
        ucb_w = params(2); % weight of ucb2 param
        rnd_w = params(3); % weight of random exploration (inverse softmax temp)
        sty_w = params(4); % weight of sticky param
        time_scale = params(5); % time scale of GP

        % GP settings
        meanfunc = @meanConst;
        covfunc = @cov_SEcircSpaceTime;
        likfunc = @likLogistic;
        infmethod = @infLaplace;
        
        % gp constants
        sig_sd = 1;
        mean_const = 0;

        hyp = struct('mean', mean_const, 'cov', [length_scale, num_choices, time_scale, sig_sd]);

        latent_vars = zeros(n_rounds, n_trials, num_choices, 2); % mu and sd of gaussian process
                
        for rd = 1 : n_rounds
            % make a random choice on the first trial
            choices(rd, 1) = ceil(rand_nums(rd,1,1)*num_choices); 
            rewards(rd, 1) = rand_nums(rd,1,2) < rew_probs(rd,choices(rd,1)); % determine reward based on probability
            if use_latent == 1
                [~, ~, mu, s2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, [], [], [(1:num_choices)', ones(num_choices,1)]);
            else
                [mu, s2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, [], [], [(1:num_choices)', ones(num_choices,1)]);
            end
            latent_vars(rd,1,:,1) = mu; latent_vars(rd,1,:,2) = s2; 

            % loop over trials
            for tr = 2 : n_trials
                if use_latent == 1
                    [~, ~, mu, s2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, ...
                        [choices(rd,1:tr-1)', (1:tr-1)'], rewards(rd,1:tr-1)', ...
                        [(1:num_choices)', ones(num_choices,1)*tr]);
                else
                    [mu, s2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, ...
                        [choices(rd,1:tr-1)', (1:tr-1)'], rewards(rd,1:tr-1)', ...
                        [(1:num_choices)', ones(num_choices,1)*tr]);
                end
                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = s2; 
                
                % sticky
                prev_choice = zeros(num_choices,1);
                prev_choice(choices(rd,tr-1)) = 1;
                
                % softmax
                choice_pref = (mu + ucb_w*sqrt(s2))*rnd_w + prev_choice*sty_w;
                choice_pref = choice_pref - max(choice_pref); % to avoid overflow errors in softmax
                choice_probs = exp(choice_pref) / sum(exp(choice_pref)); % softmax

                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0;cumsum(choice_probs)]),1); % make a choice for the current trial
                rewards(rd, tr) = rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)); % determine reward based on probability                
            end
        end
        


    case 'Qlearning'
        %% Q-learning with optional generalisation, UCB-1, stickyness, Q-decay

        % unpack params
        lr_max = params(1); % learning rate
        length_scale = params(2); % length_scale parameter
        q_decay = params(3); % decay of q values towards 0
        ucb_w = params(4); % weight of ucb1 param
        rnd_w = params(5); % weight of random exploration (inverse softmax temp)
        sty_w = params(6); % weight of sticky param
        q0 = params(7); % initial Q-value

        arm_vec = 1:num_choices;
        dist_mat = min(abs(arm_vec(:)-arm_vec), num_choices-abs(arm_vec(:)-arm_vec));

        latent_vars = zeros(n_rounds, n_trials, num_choices, 2); % q values and ucb-1 term

        for rd = 1 : n_rounds
            q_a = zeros(1,num_choices)+q0; % initialise q-values to q0
            n_a = zeros(1,num_choices)+1; % choice sample counter

            latent_vars(rd,1,:,1) = q_a; latent_vars(rd,1,:,2) = 0;

            % make a random choice on the first trial
            choices(rd, 1) = ceil(rand_nums(rd,1,1)*num_choices); 
            rewards(rd, 1) = rand_nums(rd,1,2) < rew_probs(rd,choices(rd,1)); % determine reward based on probability

            for tr = 2 : n_trials
                q_a = q_a - q_decay*(q_a-0); % q values decay towards 0

                rpe = (rewards(rd,tr-1) - q_a(choices(rd,tr-1)));
                                
                % Q learning    
                if length_scale > 0
                    lr_q = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*length_scale^2)) * lr_max;                    
                    q_a = q_a + lr_q * rpe;
                else
                    q_a(choices(rd,tr-1)) = q_a(choices(rd,tr-1)) + lr_max * rpe;
                end         
                clip(q_a, 0, 1);        

                n_a(choices(rd,tr-1)) = n_a(choices(rd,tr-1))+1;
                u_a = sqrt(2*log(tr-1) ./ n_a);

                latent_vars(rd,tr,:,1) = q_a; latent_vars(rd,tr,:,2) = u_a;
        
                prev_choice = zeros(1,num_choices);
                prev_choice(choices(rd,tr-1)) = 1;
        
                choice_pref = (q_a + ucb_w*u_a)*rnd_w + prev_choice*sty_w;
                choice_pref = choice_pref - max(choice_pref);  % to avoid overflow errors in softmax
                choice_probs = exp(choice_pref) / sum(exp(choice_pref));  % softmax
                choice_probs(choice_probs==0) = eps;

                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0;cumsum(choice_probs).']),1); % make a choice for the current trial
                rewards(rd, tr) = rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)); % determine reward based on probability 
            end
        end

    case 'BetaBayes'
        %% Bayesian option learning plus softmax with optional generalisation, UCB-2, stickyness, Q-decay

        % unpack params
        lr = params(1); % learning rate
        length_scale = params(2); % generalisation parameter
        dr = params(3); % decay rate
        ucb_w = params(4); % weight of ucb param
        rnd_w = params(5); % weight of random exploration (inverse softmax temp)
        sty_w = params(6); % weight of sticky param        
        a0 = params(7); % initial a-value for beta distr
        b0 = params(8); % initial b-value for beta distr

        arm_vec = 1:num_choices;
        dist_mat = min(abs(arm_vec(:)-arm_vec), num_choices-abs(arm_vec(:)-arm_vec));

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
                S = S*(1-dr);
                F = F*(1-dr);
                
                if length_scale > 0
                    lr_b = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*length_scale^2)) * lr;  
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
        
                prev_choice = zeros(1,num_choices);
                prev_choice(choices(rd,tr-1)) = 1;
        
                choice_pref = (beta_m + ucb_w*beta_sd)*rnd_w + prev_choice*sty_w;
                choice_pref = choice_pref - max(choice_pref);
                choice_probs = exp(choice_pref) / sum(exp(choice_pref));
                choice_probs(choice_probs==0) = eps;

                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0;cumsum(choice_probs)]),1); % make a choice for the current trial
                rewards(rd, tr) = rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)); % determine reward based on probability 
            end
        end

    case 'WSLS'
        %% win stay lose shift

        % unpack params
        p_switch_r = params(1); % switch prob after reward
        p_switch_n = params(2); % switch prob after no-reward
        sd_r = params(3); % std dev of distance moved after reward
        sd_n = params(4); % std dev of distance moved after no-reward

        arm_vec = 1:num_choices;
        dist_mat = min(abs(arm_vec(:)-arm_vec), num_choices-abs(arm_vec(:)-arm_vec));

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
                
                choices(rd, tr) = find(histcounts(rand_nums(rd,tr,1), [0;cumsum(choice_probs)]),1); % make a choice for the current trial
                rewards(rd, tr) = rand_nums(rd,tr,2) < rew_probs(rd, choices(rd, tr)); % determine reward based on probability 

            end
        end

end
