function [rprobset, is_unstr] = generateRProbSet(num_sess, n_trials, arms, shuffle, n_blocks_per_sess, pure)
% GENERATERPROBSET  Simulated reward-probability schedules for the bandit task.
%
%   [rprobset, is_unstr] = generateRProbSet(num_sess, n_trials, arms, shuffle, ...
%                                           n_blocks_per_sess, pure)
%
% Inputs
%   num_sess          number of sessions to generate
%   n_trials          trials per block
%   arms              number of arms (ports)
%   shuffle           false = structured (correlated) environment,
%                     true  = unstructured environment
%   n_blocks_per_sess blocks per session (default 1); each block draws its
%                     own reward profile
%   pure              false (default) = environments as run in the task:
%                       - structured env: each session is, at random, an
%                         unstructured session with probability 1/9 (about
%                         1 in every 9 sessions, but not at predictable positions)
%                       - unstructured env: any unique permutation of the profile
%                     true = pure environments, no mixing:
%                       - structured env: structured sessions only
%                       - unstructured env: only purely unstructured
%                         permutations (see below)
%
% Outputs
%   rprobset          num_sess x (n_trials*n_blocks_per_sess) x arms reward
%                     probabilities (0-1, rounded to 2 decimals). The
%                     probabilities are constant within a block.
%   is_unstr          num_sess x 1 logical, true for unstructured sessions
%
% Reward profiles
%   A structured profile is a Gaussian over the arms (peak 0.8, floor 0.1)
%   centred on a randomly chosen arm, so neighbouring arms have similar
%   reward probabilities.
%   A purely unstructured profile is an order of one of the structured
%   profiles' values that is not itself a structured profile (i.e. it is
%   not the Gaussian centred on any arm).
%   - Unstructured sessions in the structured env: all distinct such
%     orders, over all profiles, form one pool and each is equally likely.
%     For 4 arms that is 22 orders of {80,46,15,10} and 10 of {80,46,46,15}
%     (22/32 and 10/32).
%   - Pure unstructured env: a random peak arm picks the value set, then a
%     random non-structured order of it (value sets equally likely).

    % ---- defaults ----
    if nargin < 5 || isempty(n_blocks_per_sess)
        n_blocks_per_sess = 1;
    end
    if nargin < 6 || isempty(pure)
        pure = false;
    end
    p_unstr_in_str = 1/9;   % chance that a structured-env session is unstructured

    % ---- structured profiles, one per possible peak arm ----
    x = 1:arms;
    sig = 1.75/2;   % width of the Gaussian, in arms
    amp = 0.7;      % peak height above the floor
    vo = 0.1;       % floor (lowest reward probability)
    profiles = zeros(arms, arms);   % row m = profile peaking at arm m
    for m = 1:arms
        profiles(m, :) = round(amp*exp(-0.5*((x - m).^2)/(sig.^2)) + vo, 2);
    end

    % ---- possible orders of each profile's probabilities ----
    % all_perms{m}: every distinct order of profile m's values
    all_perms = cell(1, arms);
    for m = 1:arms
        all_perms{m} = unique(perms(profiles(m, :)), 'rows');   % unique: drop repeats from equal values
    end
    % unstr_pool: one pool of all distinct purely unstructured orders, over
    % every profile (as in the task). Each order is equally likely, so a
    % profile's value set is drawn in proportion to how many orders it has.
    % For 4 arms: 22 orders of {80,46,15,10} and 10 of {80,46,46,15},
    % i.e. 22/32 and 10/32.
    % Used only for the unstructured sessions inside the structured env.
    unstr_pool = unique(cell2mat(all_perms'), 'rows');
    unstr_pool = unstr_pool(~ismember(unstr_pool, profiles, 'rows'), :);
    % unstr_perms{m}: purely unstructured orders of profile m's values only,
    % used for the pure unstructured env
    unstr_perms = cell(1, arms);
    for m = 1:arms
        unstr_perms{m} = all_perms{m}(~ismember(all_perms{m}, profiles, 'rows'), :);
    end

    % ---- decide which sessions are unstructured ----
    if shuffle
        % unstructured environment: every session is unstructured
        is_unstr = true(num_sess, 1);
    elseif pure
        % pure structured environment: none
        is_unstr = false(num_sess, 1);
    else
        % structured environment as in the task: random ~1/9 of sessions
        is_unstr = rand(num_sess, 1) < p_unstr_in_str;
    end

    % ---- fill in each block ----
    rprobset = zeros(num_sess, n_trials*n_blocks_per_sess, arms);
    for j = 1:num_sess
        for k = 1:n_blocks_per_sess
            mean_arm = randi(arms);   % which profile's values this block uses
            if shuffle && ~pure
                % unstructured env (task): any distinct order of this
                % block's profile values
                P = all_perms{mean_arm};
                gx = P(randi(size(P, 1)), :);
            elseif shuffle && pure
                % pure unstructured env: a non-structured order of this
                % block's profile values (value sets drawn equally often)
                P = unstr_perms{mean_arm};
                gx = P(randi(size(P, 1)), :);
            elseif is_unstr(j)
                % unstructured session inside the structured env: one order
                % from the pooled set of purely unstructured orders, i.e.
                % 22/32 vs 10/32 for 4 arms (mean_arm is not used)
                gx = unstr_pool(randi(size(unstr_pool, 1)), :);
            else
                % structured: the Gaussian itself
                gx = profiles(mean_arm, :);
            end
            trial_idx = (k-1)*n_trials + (1:n_trials);
            rprobset(j, trial_idx, :) = repmat(gx, n_trials, 1);
        end
    end
end
