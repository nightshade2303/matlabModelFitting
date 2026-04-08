function [nll_rnd, nll_rnd_mat] = calcChanceNLL(choices, num_choices, rnd_flag)

if nargin < 3
    rnd_flag = 0;
end
%  depending on the num choices array get max n_rounds, n_trials for this iteration
n_rounds = size(choices,1);
n_trials = size(choices,2);

% change type of chance NLL based on different flags
if rnd_flag == 1
    null_probs = ones(1,num_choices)/num_choices; % equal chance of selecting choices (0.25 for 4 choices)
else
    null_probs = arrayfun(@(x) sum(choices(:)==x)/numel(choices), 1:num_choices); % prob of choosing ports is dictated by overall choice frequency
end

nll_rnd_mat = zeros(n_rounds, n_trials-1);
for rd = 1 : n_rounds

    for tr = 2 : n_trials  
        if isnan(choices(rd,tr))
            break;
        end      
        nll_rnd_mat(rd,tr-1) = -log(null_probs(choices(rd,tr))); % null model (calculate it per trial)

    end
end

nll_rnd = sum(nll_rnd_mat(:));




