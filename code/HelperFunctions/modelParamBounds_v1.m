function [algo, param_bnds, param_names, free_params] = modelParamBounds(modelName)
%MODELPARAMBOUNDS return algo, upper and lower param bounds for model sim / fit

switch char(modelName)
    case 'Q' % Q learning (basic)
        algo = 'Qlearning';
        % params     lr_max lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w, sty_l
        param_bnds = [0,     0,     0,       .5,   0,    1e-2,   0,     0;    % lb
                      1,     0,     0,       .5,   0,     2e1,   0,     0];   % ub 
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];

    case 'Qs' % Q learning + sticky
        algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w,  sty_l
        param_bnds = [0,     0,      0,      .5,   0,      1e-2,   0,     0;  % lb
                      1,     0,      0,      .5,   0,      2e1,    4,     0]; % ub
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];

    case 'Qgs' % Q learning + lr-gen + sticky
        algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w,  sty_l
        param_bnds = [0,     0,      0,       .5,     0,   1e-2,   0,      0; 
                      1,     3,       0,       .5,     0,   2e1,    4,      0];
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];

    case 'Qsl' % Q learning + sticky-local
        algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w,  sty_l
        param_bnds = [0,     0,      0,       .5,     0,   1e-2,   0,      0; 
                      1,     0,      0,       .5,     0,   2e1,    4,      3];
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];

    case 'Qgsl' % Q learning + lr-gen + sticky-local
        algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w,  sty_l
        param_bnds = [1e-2,  1e-2,    0,       .5,     0,   1e-2,   0,      0; 
                      1,     3,       0,       .5,     0,   2e1,    4,      3];
        param_names = ["lr","lambda","decay","q0","ucb","beta","sty","sty-lambda"];

    case 'Qdgsl' % Q learning + decay + lr-gen + sticky-local
        algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0,  ucb1_w, rnd_w, sty_w,  sty_l
        param_bnds = [0,     1e-2,    0,       .5,     0,   1e-2,   0,      0; 
                      1,     3,       1,       .5,     0,   2e1,    4,      3];
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
        param_bnds = [1,      0,      0,     1,  1,     0,    1e-2,     0,     0;    % lb
                      1,      0,      0,     1,  1,     0,     2e1,     4,     0];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'Bsl' % Bayes + sticky-local
        algo = 'BayesUCB';
        % params     lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w,  sty_w, sty_l 
        param_bnds = [1,      0,      0,     1,  1,     0,    1e-2,     0,     0;    % lb
                      1,      0,      0,     1,  1,     0,     2e1,     4,     3];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'Bus' % Bayes + UCB-2 + sticky
        algo = 'BayesUCB';
        % params     lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w,  sty_w, sty_l 
        param_bnds = [1,      0,      0,     1,  1,     0,    1e-2,     0,     0;    % lb
                      1,      0,      0,     1,  1,     2,     2e1,     4,     0];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'Busl' % Bayes + UCB + sticky-local
        algo = 'BayesUCB';
        % params     lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w,  sty_w, sty_l 
        param_bnds = [1,      0,      0,     1,  1,     0,    1e-2,     0,     0;    % lb
                      1,      0,      0,     1,  1,     2,     2e1,     4,     3];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'Bdusl' % Bayes + decay + UCB + sticky-local
        algo = 'BayesUCB';
        % params     lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w,  sty_w, sty_l 
        param_bnds = [1,      0,      0,     1,  1,     0,    1e-2,     0,     0;    % lb
                      1,      0,      1,     1,  1,     2,     2e1,     4,     3];   % ub
        param_names = ["lr","lambda","decay","a0","b0","ucb","beta","sty","sty-lambda"];

    case 'G' % GP-UCB
        algo = 'GPspace';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [1e-2,   100,     0,    1e-2,   0,     0;     % lb
                      3,      100,     0,    2e1,    0,     0];    % ub
        param_names = ["lambda","trials","ucb","beta","sty","sty-lambda"];

    case 'Gs' % GP-UCB + sticky
        algo = 'GPspace';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [1e-2,   100,     0,    1e-2,   0,     0;     % lb
                      3,      100,     0,    2e1,    4,     0];    % ub
        param_names = ["lambda","trials","ucb","beta","sty","sty-lambda"];

    case 'Gsl' % GP-UCB + sticky-local
        algo = 'GPspace';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [1e-2,   100,     0,    1e-2,   0,     1e-2;     % lb
                      3,      100,     0,    2e1,    4,     3];    % ub
        param_names = ["lambda","trials","ucb","beta","sty","sty-lambda"];

    case 'Gus' % GP-UCB + ucb-2 + sticky
        algo = 'GPspace';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [1e-2    100,     0,   1e-2,    0,     0;     % lb
                      3,      100,     2,    2e1,    4,     0];    % ub
        param_names = ["lambda","trials","ucb","beta","sty","sty-lambda"];

    case 'Gusl' % GP-UCB + ucb-2 + sticky-local
        algo = 'GPspace';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [1e-2    100,     0,   1e-2,    0,     0;     % lb
                      3,      100,     2,    2e1,    4,     3];    % ub
        param_names = ["lambda","trials","ucb","beta","sty","sty-lambda"];

    case 'Gdus' % GP-UCB + decay + ucb-2 + sticky
        algo = 'GPspacetime';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [1e-2    5,     0,   1e-2,    0,     0;     % lb
                      3,      100,   2,    2e1,    4,     0];    % ub
        param_names = ["lambda","tscale","ucb","beta","sty","sty-lambda"];

    case 'Gdusl' % GP-UCB + decay + ucb-2 + sticky-local
        algo = 'GPspacetime';
        % params     gen_l  time_l  ucb_w  rnd_w  sty_w  sty_l
        param_bnds = [1e-2    5,     0,   1e-2,    0,     0;  % lb
                      3,      100,   2,    2e1,    4,     3];    % ub
        param_names = ["lambda","tscale","ucb","beta","sty","sty-lambda"];

end

n_params = size(param_bnds,2);
free_params = param_bnds(1,:) ~= param_bnds(2,:);