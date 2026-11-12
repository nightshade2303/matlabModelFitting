function [rprobset] = generateRProbSet(num_sess, n_trials, arms, shuffle, n_blocks_per_sess)
    if nargin < 5
        n_blocks_per_sess = 1;
    end
    rprobset = zeros(num_sess, n_trials*n_blocks_per_sess, arms);
    for j = 1:num_sess
        for k = 1:n_blocks_per_sess
            x = linspace(1,arms,arms);
            sig = 1.75/2;
            amp = 0.7;
            vo = 0.1;
            mean_arm = randi(arms);
            gx = (amp*exp(-0.5*((x-mean_arm).^2)/(sig.^2)))+vo;
    
            if shuffle
                all_permutations = perms(gx);
                sel_permutation = all_permutations(randi(size(all_permutations, 1)), :);
                % start_ind = (n_trials*k) - n_trials+1;
                rprobset(j, (n_trials*k)-n_trials+1:n_trials*k, :) = repmat(round(sel_permutation, 2), n_trials, 1);
            else
                rprobset(j, (n_trials*k)-n_trials+1:n_trials*k, :) = repmat(round(gx, 2), n_trials, 1);
            end
        end
    end
end