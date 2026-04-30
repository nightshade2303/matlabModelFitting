function [algo, param_bnds, param_names, free_params] = modelParamBounds(modelName)
%MODELPARAMBOUNDS return algo, upper and lower param bounds for model sim / fit

switch char(modelName)
    case 'Q' % Q learning (basic)
        algo = 'Qlearning';
        % params     lr_max lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w, sty_l
        param_bnds = [1e-2,  0,     0,       .5,   0,    1e-2,   0,     0;    % lb
                      1,     0,     0,       .5,   0,     2e1,   0,     0];   % ub 
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];

    case 'Qs' % Q learning + sticky
        algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w,  sty_l
        param_bnds = [1e-2,  0,      0,      .5,   0,      1e-2,   0.2,   0;  % lb
                      1,     0,      0,      .5,   0,      2e1,    2,     0]; % ub
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];

    case 'Qgs' % Q learning + lr-gen + sticky
        algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w,  sty_l
        param_bnds = [1e-2,  0.4,    0,       .5,     0,   1e-2,   0.2,    0; 
                      1,     4,      0,       .5,     0,   2e1,    2,      0];
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];

    case 'Qsl' % Q learning + sticky-local
        algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w,  sty_l
        param_bnds = [1e-2,  0,      0,       .5,     0,   1e-2,   0.2,    0.4; 
                      1,     0,      0,       .5,     0,   2e1,    2,      4];
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];

    case 'B' % Beta-Bayes (Option learning)
        algo = 'BayesUCB';
        % params     lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w,  sty_w, sty_l 
        param_bnds = [1,      0,      0,     1,  1,     0,    1e-2,     0,     0;    % lb
                      1,      0,      0,     1,  1,     0,     2e1,     0,     0];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'Bs' % Bayes + sticky
        algo = 'BayesUCB';
        % params     lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w,  sty_w, sty_l 
        param_bnds = [1,      0,      0,     1,  1,     0,    1e-2,     0.2,   0;    % lb
                      1,      0,      0,     1,  1,     0,     2e1,     2,     0];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'Bus' % Bayes + UCB-2 + sticky
        algo = 'BayesUCB';
        % params     lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w,  sty_w, sty_l 
        param_bnds = [1,      0,      0,     1,  1,    0.1,   1e-2,     0.2,   0;    % lb
                      1,      0,      0,     1,  1,     1,     2e1,     2,     0];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'Bgus' % Bayes + UCB-2 + sticky
        algo = 'BayesUCB';
        % params     lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w,  sty_w, sty_l 
        param_bnds = [1,      1e-1,      0,     1,  1,    0.1,   1e-2,    0.2,   0;    % lb
                      1,      3,      0,     1,  1,     1,     2e1,     2,     0];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'Busl' % Bayes + UCB + sticky-local
        algo = 'BayesUCB';
        % params     lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w,  sty_w, sty_l 
        param_bnds = [1,      0,      0,     1,  1,    0.1,   1e-2,     0.2,   0.4;    % lb
                      1,      0,      0,     1,  1,     1,     2e1,     2,     4];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'G' % GP-UCB
        algo = 'GPspace';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [0.4,    60,     0,    1e-2,   0,     0;     % lb
                      4,      60,     0,    2e1,    0,     0];    % ub
        param_names = ["lambda","trials","ucb","beta","sty","sty-lambda"];

    case 'Gs' % GP-UCB + sticky
        algo = 'GPspace';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [0.4,    60,     0,    1e-2,   0.2,   0;     % lb
                      4,      60,     0,    2e1,    2,     0];    % ub
        param_names = ["lambda","trials","ucb","beta","sty","sty-lambda"];

    case 'Gus' % GP-UCB + ucb-2 + sticky
        algo = 'GPspace';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [0.4,    60,    0.1,    1e-2,   0.2,    0;     % lb
                      4,      60,     1,    2e1,    2,      0];    % ub
        param_names = ["lambda","trials","ucb","beta","sty","sty-lambda"];
end

n_params = size(param_bnds,2);
free_params = param_bnds(1,:) ~= param_bnds(2,:);