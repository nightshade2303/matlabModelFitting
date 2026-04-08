function [nll_tot, nll_mat, latent_vars] = calcNLL(params, algo, choices, rewards, num_choices, linear_env)
% Calculate NLL for param and algo choice

if nargin < 6
    linear_env = 0;
end

n_rounds = size(choices,1);
n_trials = size(choices,2);

nll_mat = NaN(n_rounds, n_trials-1);

switch algo
    case 'GPspace'
        %% Gaussian Process Classification over space with optional UCB-2, stickyness, forgetting

        % unpack params
        length_scale = params(1); % length scale of GP
        ucb_w = params(2); % weight of ucb param
        rnd_w = params(3); % weight of random exploration (inverse softmax temp)
        sty_w = params(4); % weight of sticky param
        mem_l = floor(params(5)); % length of memory window (is an integer)
       
        % GP settings
        meanfunc = @meanConst;
        
        likfunc = @likLogistic;
        infmethod = @infLaplace;

        % gp constants
        sig_sd = 1;
        mean_const = 0;
        
        if linear_env == 0
            covfunc = {@covDiscrete, num_choices};
            % Set up discrete cov kernel function: SE for circular space
            [~, L] = covfunc_1Denv(num_choices, length_scale, sig_sd);
            hyp = struct('mean', mean_const, 'cov', L);
        else
            covfunc = @covSEiso;
            hyp = struct('mean', mean_const, 'cov', log([length_scale, sig_sd]));
        end

        rewards(rewards==0) = -1; % for GP classification 

        latent_vars = NaN(n_rounds, n_trials, num_choices, 2); % mu and sd of gaussian process

        for rd = 1 : n_rounds
            
            [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, [], [], (1:num_choices)');
            mu = 1./(1+exp(-lmu));
            ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
            sd = ucb-mu;
            latent_vars(rd,1,:,1) = mu; latent_vars(rd,1,:,2) = sd;

            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end
                [~, ~, lmu, ls2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, ...
                        choices(rd,max([1,tr-mem_l]):tr-1)', rewards(rd,max([1,tr-mem_l]):tr-1)', ...
                        (1:num_choices)');

                mu = 1./(1+exp(-lmu));
                ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
                sd = ucb-mu;

                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = sd; 

                prev_choice = zeros(num_choices,1);
                prev_choice(choices(rd,tr-1)) = 1;
        
                choice_pref = (mu + ucb_w*sd)*rnd_w + prev_choice*sty_w;
                choice_pref = choice_pref - max(choice_pref); % to avoid overflow errors in softmax
                choice_prob = exp(choice_pref) / sum(exp(choice_pref)); % softmax
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
            end
        end

    case 'GPspacetime'
        %% Gaussian Process Classification over space (RBF) and trials (RBF - forgetting process) with optional UCB-2, stickyness

        % unpack params
        length_scale = params(1); % length scale of GP
        ucb_w = params(2); % weight of ucb2 param
        rnd_w = params(3); % weight of random exploration (inverse softmax temp)
        sty_w = params(4); % weight of sticky param
        time_scale = params(5); % time scale of GP

        % GP settings
        meanfunc = @meanConst;
        likfunc = @likLogistic;
        infmethod = @infLaplace;
        
        % gp constants
        sig_sd = 1;
        mean_const = 0;

        if linear_env == 0
            covfunc = @cov_SEcircSpaceTime;
            hyp = struct('mean', mean_const, 'cov', [length_scale, num_choices, time_scale, sig_sd]);
        else
            covfunc = @covSEard;
            hyp = struct('mean', mean_const, 'cov', log([length_scale, time_scale, sig_sd]));
        end

        rewards(rewards==0) = -1; % for GP classification 
        
        latent_vars = NaN(n_rounds, n_trials, num_choices, 2); % mu and sd of gaussian process

        for rd = 1 : n_rounds
            % 
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
                    [choices(rd,1:tr-1)', (1:tr-1)'], rewards(rd,1:tr-1)', ...
                    [(1:num_choices)', ones(num_choices,1)*tr]);

                mu = 1./(1+exp(-lmu));
                ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
                sd = ucb-mu;

                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = sd; 

                prev_choice = zeros(num_choices,1);
                prev_choice(choices(rd,tr-1)) = 1;
        
                choice_pref = (mu + ucb_w*sd)*rnd_w + prev_choice*sty_w; 
                choice_pref = choice_pref - max(choice_pref); % to avoid overflow errors in softmax
                choice_prob = exp(choice_pref) / sum(exp(choice_pref)); % softmax
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
            end
        end

    case 'GPspace_linear'
        %% Gaussian Process Classification over space with optional UCB-2, stickyness, forgetting
 

        % unpack params
        length_scale = params(1); % length scale of GP
        ucb_w = params(2); % weight of ucb param
        rnd_w = params(3); % weight of random exploration (inverse softmax temp)
        sty_w = params(4); % weight of sticky param
        mem_l = floor(params(5)); % length of memory window (is an integer)
       
        % GP settings
        meanfunc = @meanConst;
        covfunc = @covSEiso;
        likfunc = @likLogistic;
        infmethod = @infLaplace;

        % gp constants
        sig_sd = 1;
        mean_const = 0;
        
        hyp = struct('mean', mean_const, 'cov', log([length_scale, sig_sd]));

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
                        choices(rd,max([1,tr-mem_l]):tr-1)', rewards(rd,max([1,tr-mem_l]):tr-1)', ...
                        (1:num_choices)');

                mu = 1./(1+exp(-lmu));
                ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
                sd = ucb-mu;

                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = sd; 

                prev_choice = zeros(num_choices,1);
                prev_choice(choices(rd,tr-1)) = 1;
        
                choice_pref = (mu + ucb_w*sd)*rnd_w + prev_choice*sty_w;
                choice_pref = choice_pref - max(choice_pref); % to avoid overflow errors in softmax
                choice_prob = exp(choice_pref) / sum(exp(choice_pref)); % softmax
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
            end
        end

    case 'GPspacetime_linear'
        %% Gaussian Process Classification over space (RBF) and trials (RBF - forgetting process) with optional UCB-2, stickyness

        % use_latent = 0; % if 1, use mean and sd of latent GP instead of logistic transform
        
        % unpack params
        length_scale = params(1); % length scale of GP
        ucb_w = params(2); % weight of ucb2 param
        rnd_w = params(3); % weight of random exploration (inverse softmax temp)
        sty_w = params(4); % weight of sticky param
        time_scale = params(5); % time scale of GP

        % GP settings
        meanfunc = @meanConst;
        covfunc = @covSEard;
        likfunc = @likLogistic;
        infmethod = @infLaplace;
        
        % gp constants
        sig_sd = 1;
        mean_const = 0;

        rewards(rewards==0) = -1; % for GP classification 
        
        hyp = struct('mean', mean_const, 'cov', log([length_scale, time_scale, sig_sd]));

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
                    [choices(rd,1:tr-1)', (1:tr-1)'], rewards(rd,1:tr-1)', ...
                    [(1:num_choices)', ones(num_choices,1)*tr]);

                mu = 1./(1+exp(-lmu));
                ucb = 1./(1+exp(-(lmu+2*sqrt(ls2))));
                sd = ucb-mu;
                
                latent_vars(rd,tr,:,1) = mu; latent_vars(rd,tr,:,2) = sd; 

                prev_choice = zeros(num_choices,1);
                prev_choice(choices(rd,tr-1)) = 1;
        
                choice_pref = (mu + ucb_w*sd)*rnd_w + prev_choice*sty_w; 
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
        length_scale = params(2); % length_scale parameter
        q_decay = params(3); % decay of q values towards 0
        ucb_w = params(4); % weight of ucb1 param
        rnd_w = params(5); % weight of random exploration (inverse softmax temp)
        sty_w = params(6); % weight of sticky param
        q0 = params(7); % initial Q-value

        latent_vars = NaN(n_rounds, n_trials, num_choices, 2); % q values and ucb-1 term

        for rd = 1 : n_rounds
            q_a = zeros(1,num_choices)+q0; % initialise q-values to q0
            n_a = zeros(1,num_choices)+1;
            latent_vars(rd,1,:,1) = q_a; latent_vars(rd,1,:,2) = 0;

            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end

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

                u_a = sqrt(2*log(tr-1) ./ n_a);
                
                latent_vars(rd,tr,:,1) = q_a; latent_vars(rd,tr,:,2) = u_a;
        
                prev_choice = zeros(1,num_choices);
                prev_choice(choices(rd,tr-1)) = 1;
        
                choice_pref = (q_a + ucb_w*u_a)*rnd_w + prev_choice*sty_w;
                choice_pref = choice_pref - max(choice_pref);  % to avoid overflow errors in softmax
                choice_prob = exp(choice_pref) / sum(exp(choice_pref));  % softmax
                choice_prob(choice_prob==0) = eps;
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
                n_a(choices(rd,tr-1)) = n_a(choices(rd,tr-1))+1;
            end
        end

    case 'QLocal'
        %% Independent Q learning with local sampling policy

        choice_vec = 1:num_choices;
        if linear_env == 0
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        % unpack params
        lr = params(1); % learning rate
        length_scale = params(2); % length_scale parameter
        q_decay = params(3); % decay of q values towards 0
        ucb_w = params(4); % weight of ucb1 param
        rnd_w = params(5); % weight of random exploration (inverse softmax temp)
        sty_w = params(6); % weight of sticky param
        q0 = params(7); % initial Q-value

        latent_vars = NaN(n_rounds, n_trials, num_choices, 2); % q values and ucb-1 term

        for rd = 1 : n_rounds
            q_a = zeros(1,num_choices)+q0; % initialise q-values to q0
            n_a = zeros(1,num_choices)+1;
            latent_vars(rd,1,:,1) = q_a; latent_vars(rd,1,:,2) = 0;

            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end

                q_a = q_a - q_decay*(q_a-0); % q values decay towards 0

                rpe = (rewards(rd,tr-1) - q_a(choices(rd,tr-1)));
                                
                % Q learning    

                q_a(choices(rd,tr-1)) = q_a(choices(rd,tr-1)) + lr * rpe;                  
                clip(q_a, 0, 1);        

                u_a = sqrt(2*log(tr-1) ./ n_a);
                
                latent_vars(rd,tr,:,1) = q_a; latent_vars(rd,tr,:,2) = u_a;
        
                sticky_vec = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*length_scale^2)) * sty_w;
        
                choice_pref = (q_a + ucb_w*u_a)*rnd_w + sticky_vec;
                choice_pref = choice_pref - max(choice_pref);  % to avoid overflow errors in softmax
                choice_prob = exp(choice_pref) / sum(exp(choice_pref));  % softmax
                choice_prob(choice_prob==0) = eps;
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
                n_a(choices(rd,tr-1)) = n_a(choices(rd,tr-1))+1;
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

        choice_vec = 1:num_choices;
        if linear_env == 0
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        for rd = 1 : n_rounds
            S = zeros(1,num_choices);
            F = zeros(1,num_choices);
        
            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end

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
        
                prev_choice = zeros(1,num_choices);
                prev_choice(choices(rd,tr-1)) = 1;
        
                choice_pref = (beta_m + ucb_w*beta_sd)*rnd_w + prev_choice*sty_w;
                choice_pref = choice_pref - max(choice_pref);
                choice_prob = exp(choice_pref) / sum(exp(choice_pref));
                choice_prob(choice_prob==0) = eps;
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
            end
        end

    case 'BayesLocal'
        %% Bayesian option learning plus softmax with optional stickyness generalisation, UCB-2, value-decay

        % unpack params
        lr = params(1); % learning rate
        length_scale = params(2); % generalisation parameter for sticky param
        dr = params(3); % decay rate
        ucb_w = params(4); % weight of ucb param
        rnd_w = params(5); % weight of random exploration (inverse softmax temp)
        sty_w = params(6); % weight of sticky param        
        a0 = params(7); % initial a-value for beta distr
        b0 = params(8); % initial b-value for beta distr

        choice_vec = 1:num_choices;
        if linear_env == 0
            dist_mat = min(abs(choice_vec(:)-choice_vec), num_choices-abs(choice_vec(:)-choice_vec));
        else
            dist_mat = abs(choice_vec(:)-choice_vec);
        end

        for rd = 1 : n_rounds
            S = zeros(1,num_choices);
            F = zeros(1,num_choices);
        
            for tr = 2 : n_trials
                if isnan(choices(rd,tr))
                    break;
                end

                S = S*(1-dr);
                F = F*(1-dr);
                
                S(choices(rd,tr-1)) = S(choices(rd,tr-1)) + lr*rewards(rd,tr-1);
                F(choices(rd,tr-1)) = F(choices(rd,tr-1)) + lr*(1-rewards(rd,tr-1));
                clip(S, 0, Inf);
                clip(F, 0, Inf);

                as = a0 + S;
                bs = b0 + F;
        
                beta_m = as./(as+bs);
                beta_sd = sqrt(beta_m.*(1-beta_m)./(as+bs+1));

                sticky_vec = exp(-dist_mat(choices(rd,tr-1),:).^2 / (2*length_scale^2)) * sty_w; 

                choice_pref = (beta_m + ucb_w*beta_sd)*rnd_w + sticky_vec;
                choice_pref = choice_pref - max(choice_pref);
                choice_prob = exp(choice_pref) / sum(exp(choice_pref));
                choice_prob(choice_prob==0) = eps;
                nll_mat(rd,tr-1) = -log(choice_prob(choices(rd,tr)));
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
        
                    p_a = a_stay*(1-p_switch_r) + a_switch*p_switch_r;
        
                else            % no-rew
                    a_stay = zeros(1,num_choices);
                    a_stay(choices(rd,tr-1)) = 1;
        
                    a_switch = normpdf(dist_mat(choices(rd,tr-1),:),0,sd_n);
                    a_switch(choices(rd,tr-1)) = 0; 
                    a_switch(a_switch==0) = eps;
                    a_switch = a_switch/sum(a_switch);
        
                    p_a = a_stay*(1-p_switch_n) + a_switch*p_switch_n;
                end
                p_a(p_a == 0) = eps;
                
                nll_mat(rd,tr-1) = -log(p_a(choices(rd,tr)));
            end
        end
        
end

% report total NLL
nll_tot = sum(nll_mat(:),1,"omitmissing");


