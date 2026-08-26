function [nll_tot, nll_mat, latent_vars] = calcNLL(params, algo, choices, rewards, num_choices, linear_env)
% Calculate NLL for param and algo choice

if nargin < 6
    linear_env = 1;
end

n_rounds = size(choices,1);
n_trials = size(choices,2);

nll_mat = NaN(n_rounds, n_trials-1);

switch algo
    case 'GPspace'
        %% Gaussian Process Classification over space with optional UCB-2, stickyness, forgetting

        % unpack params
        gen_l = params(1); % length scale of GP
        time_l = floor(params(2)); % length of memory window (is an integer)
        mu0 = params(3); % mean function
        ucb_w = params(4); % weight of ucb param
        rnd_w = params(5); % weight of random exploration (inverse softmax temp)
        sty_w = params(6); % weight of sticky param
        sty_l = params(7); % length scale of sticky process    
       
        % GP settings
        meanfunc = @meanConst;        
        likfunc = @likLogistic;
        infmethod = @infLaplace;

        % gp constants
        sig_sd = 1;
        mean_const = log(mu0/(1-mu0));
        
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

        rewards(rewards==0) = -1; % for GP classification 

        latent_vars = NaN(n_rounds, n_trials, num_choices, 2); % mu and sd of gaussian process

        for rd = 1 : n_rounds
            
            % [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, [], [], (1:num_choices)');
            % mu = 1./(1+exp(-lmu));
            % ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
            % sd = ucb-mu;
            % latent_vars(rd,1,:,1) = mu; latent_vars(rd,1,:,2) = sd;

            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end
                [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, ...
                        choices(rd,max([1,tr-time_l]):tr-1)', rewards(rd,max([1,tr-time_l]):tr-1)', ...
                        (1:num_choices)');

                mu = 1./(1+exp(-lmu));
                ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
                sd = ucb-mu;

                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = sd; 

                if sty_l > 0
                    sticky_vec = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*sty_l^2))' * sty_w;
                else
                    sticky_vec = zeros(num_choices,1);
                    sticky_vec(choices(rd,tr-1)) = sty_w;
                end
        
                choice_pref = (mu + ucb_w*sd)*rnd_w + sticky_vec;
                choice_pref = choice_pref - max(choice_pref); % to avoid overflow errors in softmax
                choice_prob = exp(choice_pref) / sum(exp(choice_pref)); % softmax
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
            end
        end

    case 'GPspacetime'
        %% Gaussian Process Classification over space (RBF) and trials (RBF - forgetting process) with optional UCB-2, stickyness

        % unpack params
        gen_l = params(1); % length scale of GP
        time_l = params(2); % time scale of GP
        mu0 = params(3); % mean function
        ucb_w = params(4); % weight of ucb2 param
        rnd_w = params(5); % weight of random exploration (inverse softmax temp)
        sty_w = params(6); % weight of sticky param
        sty_l = params(7); % length scale of sticky process        

        % GP settings
        meanfunc = @meanConst;
        likfunc = @likLogistic;
        infmethod = @infLaplace;
        
        % gp constants
        sig_sd = 1;
        mean_const = log(mu0/(1-mu0));

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

        rewards(rewards==0) = -1; % for GP classification 
        
        latent_vars = NaN(n_rounds, n_trials, num_choices, 2); % mu and sd of gaussian process

        for rd = 1 : n_rounds
            % [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, [], [], ...
            %     [(1:num_choices)', ones(num_choices,1)]);
            % mu = 1./(1+exp(-lmu));
            % ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
            % sd = ucb-mu;
            % latent_vars(rd,1,:,1) = mu; latent_vars(rd,1,:,2) = sd;

            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end
                [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, ...
                    [choices(rd,1:tr-1)', (1:tr-1)'], rewards(rd,1:tr-1)', ...
                    [(1:num_choices)', ones(num_choices,1)*tr]);

                mu = 1./(1+exp(-lmu));
                ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
                sd = ucb-mu;

                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = sd; 

                if sty_l > 0
                    sticky_vec = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*sty_l^2))' * sty_w;
                else
                    sticky_vec = zeros(num_choices,1);
                    sticky_vec(choices(rd,tr-1)) = sty_w;
                end
        
                choice_pref = (mu + ucb_w*sd)*rnd_w + sticky_vec; 
                choice_pref = choice_pref - max(choice_pref); % to avoid overflow errors in softmax
                choice_prob = exp(choice_pref) / sum(exp(choice_pref)); % softmax
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
            end
        end

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

        latent_vars = NaN(n_rounds, n_trials, num_choices, 2); % q values and ucb-1 term

        for rd = 1 : n_rounds
            q_a = zeros(1,num_choices)+q0; % initialise q-values to q0
            n_a = zeros(1,num_choices)+1;
            latent_vars(rd,1,:,1) = q_a; latent_vars(rd,1,:,2) = 0;

            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end

                prev_choice = zeros(1,num_choices);
                prev_choice(choices(rd,tr-1)) = 1;

                rpe = (rewards(rd,tr-1) - q_a(choices(rd,tr-1)));

                % q values decay towards 0 (except previously chosen action)
                q_a(prev_choice==0) = q_a(prev_choice==0) - q_decay*(q_a(prev_choice==0)-q0); 
                
                % q_a = q_a - q_decay*(q_a-0);    % all q-values decay             
                
                % Q learning    
                if gen_l > 0
                    lr_q = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*gen_l^2)) * lr_max;                    
                    q_a = q_a + lr_q * rpe;
                else % local learning
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
                choice_prob = exp(choice_pref) / sum(exp(choice_pref));  % softmax
                choice_prob(choice_prob==0) = eps;
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));                
            end
        end    

    case 'Qlearningbiasq0'
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
        bias1 = params(4);  % bias1
        bias2 = params(5); % bias2
        bias3 = params(6); % bias3
        bias4 = params(7); % bias4     % initial Q-value
        ucb_w = params(8);  % weight of ucb1 param
        rnd_w = params(9);  % weight of random exploration (inverse softmax temp)
        sty_w = params(10);  % weight of sticky param
        sty_l = params(11);  % length scale of sticky process 
         

        q0 = [bias1, bias2, bias3, bias4];
        latent_vars = NaN(n_rounds, n_trials, num_choices, 2); % q values and ucb-1 term

        for rd = 1 : n_rounds
            q_a = zeros(1,num_choices)+q0; % initialise q-values to q0
            n_a = zeros(1,num_choices)+1;
            latent_vars(rd,1,:,1) = q_a; latent_vars(rd,1,:,2) = 0;

            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end

                prev_choice = zeros(1,num_choices);
                prev_choice(choices(rd,tr-1)) = 1;

                rpe = (rewards(rd,tr-1) - q_a(choices(rd,tr-1)));

                % q values decay towards 0 (except previously chosen action)
                q_a(prev_choice==0) = q_a(prev_choice==0) - q_decay*(q_a(prev_choice==0)-q0); 
                
                % q_a = q_a - q_decay*(q_a-0);    % all q-values decay             
                
                % Q learning    
                if gen_l > 0
                    lr_q = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*gen_l^2)) * lr_max;                    
                    q_a = q_a + lr_q * rpe;
                else % local learning
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
                choice_prob = exp(choice_pref) / sum(exp(choice_pref));  % softmax
                choice_prob(choice_prob==0) = eps;
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));                
            end
        end

    case 'Qlearningbiassoftmax'
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
        q0 = params(4); % initial Q-value
        bias1 = params(5);  % bias1
        bias2 = params(6); % bias2
        bias3 = params(7); % bias3
        bias4 = params(8); % bias4     
        ucb_w = params(9);  % weight of ucb1 param
        rnd_w = params(10);  % weight of random exploration (inverse softmax temp)
        sty_w = params(11);  % weight of sticky param
        sty_l = params(12);  % length scale of sticky process 
         

        bias_vec = [bias1, bias2, bias3, bias4];
        latent_vars = NaN(n_rounds, n_trials, num_choices, 2); % q values and ucb-1 term

        for rd = 1 : n_rounds
            q_a = zeros(1,num_choices)+q0; % initialise q-values to q0
            n_a = zeros(1,num_choices)+1;
            latent_vars(rd,1,:,1) = q_a; latent_vars(rd,1,:,2) = 0;

            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end

                prev_choice = zeros(1,num_choices);
                prev_choice(choices(rd,tr-1)) = 1;

                rpe = (rewards(rd,tr-1) - q_a(choices(rd,tr-1)));

                % q values decay towards 0 (except previously chosen action)
                q_a(prev_choice==0) = q_a(prev_choice==0) - q_decay*(q_a(prev_choice==0)-q0); 
                
                % q_a = q_a - q_decay*(q_a-0);    % all q-values decay             
                
                % Q learning    
                if gen_l > 0
                    lr_q = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*gen_l^2)) * lr_max;                    
                    q_a = q_a + lr_q * rpe;
                else % local learning
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
        
                choice_pref = (q_a + ucb_w*u_a)*rnd_w + sticky_vec + bias_vec;
                choice_pref = choice_pref - max(choice_pref);  % to avoid overflow errors in softmax
                choice_prob = exp(choice_pref) / sum(exp(choice_pref));  % softmax
                choice_prob(choice_prob==0) = eps;
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));                
            end
        end

    case 'BayesUCB'
        %% Bayesian option learning plus softmax with optional generalisation, UCB-2, stickyness, Q-decay

        % unpack params
        lr = params(1); % learning rate
        gen_l = params(2); % generalisation parameter
        b_decay = params(3); % decay rate
        mu0 = params(4); % initial mean for beta distr        
        ucb_w = params(5); % weight of ucb param
        rnd_w = params(6); % weight of random exploration (inverse softmax temp)
        sty_w = params(7); % weight of sticky param        
        sty_l = params(8);  % length scale of sticky process    

        choice_vec = 1:num_choices;
        if linear_env == 0
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        a0 = 2*mu0; % initial a value for beta distribution
        b0 = 2*(1-mu0); % initial b value for beta distributio

        latent_vars = zeros(n_rounds, n_trials, num_choices, 2); % beta distr mean and sd

        for rd = 1 : n_rounds
            S = zeros(1,num_choices);
            F = zeros(1,num_choices);

            beta_m0 = a0./(a0+b0); beta_sd0 = sqrt(beta_m0.*(1-beta_m0)./(a0+b0+1));
            latent_vars(rd,1,:,1) = beta_m0; latent_vars(rd,1,:,2) = beta_sd0;
        
            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end

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
                choice_prob = exp(choice_pref) / sum(exp(choice_pref));
                choice_prob(choice_prob==0) = eps;
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
            end
        end  

    case 'StateInference'
        %% Hidden-state inference model matching code/state_inference.py

        % unpack params
        gam = params(1);    % transition decay over distance between states (generalization?)
        c = params(2);        % observation confidence / evidence strength
        gen_l = params(3);  % generalzation length scale
        rnd_w = params(4);    % softmax temperature 
        sty_w = params(5);   % weight of sticky param        
        sty_l = params(6);   % length scale of sticky process 
    
        if num_choices ~= 4
            error('StateInference expects 4 choices, but received %d. Go write your own model.', num_choices);
        end

        choice_vec = 1:num_choices;
        if linear_env == 0
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        b0 = ones(1, num_choices) / num_choices;
        T = [0.25+0.75*gam, 0.25-0.25*gam, 0.25-0.25*gam, 0.25-0.25*gam;
            0.25-0.25*gam, 0.25+0.75*gam, 0.25-0.25*gam, 0.25-0.25*gam;
            0.25-0.25*gam, 0.25-0.25*gam, 0.25+0.75*gam, 0.25-0.25*gam;
            0.25-0.25*gam, 0.25-0.25*gam, 0.25-0.25*gam, 0.25+0.75*gam];

        latent_vars = NaN(n_rounds, n_trials, num_choices, 2);

        for rd = 1 : n_rounds
            b = b0;  % set initial state of the beliefs

            for tr = 2 : n_trials
                if isnan(choices(rd, tr))
                    break;
                end

                % prev_choice = zeros(1,num_choices);
                % prev_choice(choices(rd,tr-1)) = 1;

                latent_vars(rd, tr, :, 1) = b;

                if sty_l > 0
                    sticky_vec = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*sty_l^2)) * sty_w;
                else
                    sticky_vec = zeros(1,num_choices);
                    sticky_vec(choices(rd,tr-1)) = sty_w;
                end
                
                % calc gen 
                if gen_l > 0
                    c_val = (2*c*exp(-dist_mat(choices(rd, tr-1),:).^2 / (2*gen_l^2))) - c;
                else
                    c_val = zeros(1, num_choices);
                    c_val(choices(rd, tr-1)) = c;
                end

                choice_pref = b.*rnd_w + sticky_vec;                % inv temp x preferences + sticky bonus
                choice_pref = choice_pref - max(choice_pref);
                choice_prob = exp(choice_pref) / sum(exp(choice_pref));
                latent_vars(rd, tr, :, 2) = choice_prob;

                nll_mat(rd, tr-1) = -log(choice_prob(choices(rd, tr)));

                l = stateInferenceLikelihood(choices(rd, tr), rewards(rd, tr), c_val, num_choices);
                b_post = l .* b;

                if sum(b_post) > 0
                    b_post = b_post / sum(b_post);
                else
                    b_post = b;
                end

                b = b_post * T;
            end
        end
        
    case 'WSLS'
        %%
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
            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end

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
                
                nll_mat(rd,tr-1) = -log(choice_probs(choices(rd,tr)));
            end
        end
        
    end

% report total NLL
nll_tot = sum(nll_mat(:),1,"omitmissing");

end