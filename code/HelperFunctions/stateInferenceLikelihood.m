function likelihoods = stateInferenceLikelihood(action, reward, c, arms)
% STATEINFERENCELIKELIHOOD observation model used by the state inference branch

likelihoods = zeros(1, arms);

for state = 1:arms
    if (action == state) && (reward == 1)
        likelihood = 0.5 * (1 + c(state));
    elseif (action == state) && (reward == 0)
        likelihood = 0.5 * (1 - c(state));
    elseif (action ~= state) && (reward == 1)
        likelihood = 0.5 * (1 - c(state));
    elseif (action ~= state) && (reward == 0)
        likelihood = 0.5 * (1 + c(state));
    else
        error('Unhandled state inference likelihood case.');
    end

    likelihoods(state) = likelihood;
end