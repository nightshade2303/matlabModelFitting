function [rprobset] = generateRProbSet(num_sess, arms, shuffle)
    rprobset = zeros(num_sess, arms);
    for j = 1:num_sess
        x = linspace(1,arms,arms);
        sig = 1.75/2;
        amp = 0.7;
        vo = 0.1;
        mean_arm = randi(arms);
        gx = (amp*exp(-0.5*((x-mean_arm).^2)/(sig.^2)))+vo;

        if shuffle
            all_permutations = perms(gx);
            sel_permutation = all_permutations(randi(size(all_permutations, 1)), :);
            rprobset(j, :) = round(sel_permutation, 2);
        else
            rprobset(j, :) = round(gx, 2);
        end
    end
end