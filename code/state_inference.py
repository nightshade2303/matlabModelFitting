#%%
# state inference model:
import os
import numpy as np
import csv
# from optim import resolve_optimizer
import pandas as pd
os.chdir("/mnt/pve/Homes/homna/code/code-server/model_fitting")
import sys
from pathlib import Path
_CODE_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_SERVER_DIR))
import random
import numpy as np
import csv
from supplementaryFunctions import tic, toc
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import datetime
from scipy.optimize import minimize
from scipy.optimize import shgo
from optim import resolve_optimizer
from joblib import Parallel, delayed

#%%

def softmax(b, tau): # pass beliefs and inverse temp
    p = np.exp(b/tau)
    p = p/np.sum(p)
    return p


def softmax_sticky(b, tau, sticky, k):
    p = np.exp(b/ tau + k*sticky) # add a bonus to the chosen arm using k
    p = p/np.sum(p)
    return p

def likelihood(a, r, c, arms):
    """
    Calculates P(o_t | s_t) for all possible states s_t.
    o_t is the pair {action, reward}
    """
    likelihoods = np.zeros(arms)
    for state in range(arms):
        if (a == state) & (r == 1):
            l = 0.5 *(1 + c)
        elif (a == state) & (r == 0):
            l = 0.5 *(1 - c)
        elif (a != state) & (r == 1):
            l = 0.5 *(1 - c)
        elif (a != state) & (r == 0):
            l = 0.5 *(1 + c)
        likelihoods[state] = l
    return likelihoods


def fit_si(x0, data, arms, return_values = False):
    tau, c, gamma = x0
    tau = np.exp(tau)
    chosen_actions = data[0]
    rewards_received = data[1]
    sessions = chosen_actions.shape[0]
    trials = chosen_actions.shape[1]

    b0 = np.ones(arms) / arms # intialize value of b0 to be 0.25 for each state (port)

    # transition matrix (symmetric, i.e, probability of switching from 1 --> 2 === 1---> 3, etc)
    stay = 0.25 + 0.75*gamma
    switch = 0.25 - 0.25*gamma
    T = np.full((4,4), switch)
    np.fill_diagonal(T, stay)

    belief_states = np.zeros((sessions, trials, arms)) # will store the belief state of the model at each trial
    probs = np.full((sessions, trials), np.nan) # stores the probability of choosing each action on each trial using soft max
    for session in range(sessions):
        b = b0.copy()
        for trial in range(trials):
            action = chosen_actions[session, trial]
            if np.isnan(action):
                continue
            belief_states[session,trial,:] = b
            # pick an action usig the curent belief and the 
            prob_a = softmax(b, tau) # return the probability of each action given current belief state

            a = int(action)
            r = rewards_received[session, trial]
            if np.isnan(r):
                continue

            # probability of action chosen by animal being chosen
            probs[session, trial] = prob_a[a]
            # update belief state based on the action and reward you got
            # get likelihood
            l = likelihood(a,r, c, arms)

            b_post = l * b
            if np.sum(b_post) > 0:
                b_post = b_post / np.sum(b_post)
            else:
                # Fallback if numerical underflow occurs (rare)
                b_post = b

            # Prediction
            b = b_post @ T

    # compute log likelihood
    ll = np.nansum(np.log(probs))
    n_valid_trials = np.sum(~np.isnan(probs))
    nll = -ll
    if return_values == True:
        return nll, belief_states, probs
    return nll

def fit_si_sticky(x0, data, arms, return_values = False):
    tau, c, gamma, k = x0
    tau = np.exp(tau)
    chosen_actions = data[0]
    rewards_received = data[1]
    # block = data[2]
    sessions = chosen_actions.shape[0]
    trials = chosen_actions.shape[1]

    b0 = np.ones(arms) / arms # intialize value of b0 to be 0.25 for each state (port)
    sticky = np.zeros(arms) # initialize sticky vector to be 0 for all arms
    # transition matrix (symmetric, i.e, probability of switching from 1 --> 2 === 1---> 3, etc)
    stay = 0.25 + 0.75*gamma
    switch = 0.25 - 0.25*gamma
    T = np.full((4,4), switch)
    np.fill_diagonal(T, stay)

    belief_states = np.zeros((sessions, trials, arms)) # will store the belief state of the model at each trial
    probs = np.full((sessions, trials), np.nan) # stores the probability of choosing each action on each trial using soft max
    for session in range(sessions):
        b = b0.copy()

        for trial in range(trials):
            action = chosen_actions[session, trial]
            if np.isnan(action):
                continue
            belief_states[session,trial,:] = b
            # pick an action usig the curent belief and the 
            prob_a = softmax_sticky(b, tau, sticky, k) # return the probability of each action given current belief state

            a = int(action)
            sticky = np.zeros(arms)
            sticky[a] = 1 # update sticky vector based on chosen action
            r = rewards_received[session, trial]
            if np.isnan(r):
                continue

            # probability of action chosen by animal being chosen
            probs[session, trial] = prob_a[a]
            # update belief state based on the action and reward you got
            # get likelihood
            l = likelihood(a,r, c, arms)

            b_post = l * b
            if np.sum(b_post) > 0:
                b_post = b_post / np.sum(b_post)
            else:
                # Fallback if numerical underflow occurs (rare)
                b_post = b

            # Prediction
            b = b_post @ T

    # compute log likelihood
    ll = np.nansum(np.log(probs))
    n_valid_trials = np.sum(~np.isnan(probs))
    nll = -ll
    if return_values == True:
        return nll, belief_states, probs
    return nll

def fit_si_bias(x0, data, arms, return_values = False):
    tau, c, gamma, b1, b2, b3 = x0
    tau = np.exp(tau)
    b4 = 1-(b1+b2+b3)
    if b4 <0:
        return 1e10 # incase the 4th bias term is not a plausible probability
    if b4 > 1:
        return 1e10
    chosen_actions = data[0]
    rewards_received = data[1]
    # block = data[2]
    sessions = chosen_actions.shape[0]
    trials = chosen_actions.shape[1]

    # b0 = np.ones(arms) / arms # intialize value of b0 to be 0.25 for each state (port)
    b0 = np.array([b1, b2, b3, (1-(b1+b2+b3))])

    # transition matrix (symmetric, i.e, probability of switching from 1 --> 2 === 1---> 3, etc)
    stay = 0.25 + 0.75*gamma
    switch = 0.25 - 0.25*gamma
    T = np.full((4,4), switch)
    np.fill_diagonal(T, stay)

    belief_states = np.zeros((sessions, trials, arms)) # will store the belief state of the model at each trial
    probs = np.full((sessions, trials), np.nan) # stores the probability of choosing each action on each trial using soft max
    for session in range(sessions):
        b = b0.copy()
        # belief_states[session, 0, :] = b0
        # probs[session, 0] = softmax(b0, tau)
        for trial in range(trials):
            action = chosen_actions[session, trial]
            if np.isnan(action):
                continue  # or break if trials are padded at the end
            belief_states[session,trial,:] = b
            # pick an action usig the curent belief and the 
            prob_a = softmax(b, tau) # return the probability of each action given current belief state

            a = int(action)
            r = rewards_received[session, trial]

            # probability of action chosen by animal being chosen
            probs[session, trial] = prob_a[a]
            # update belief state based on the action and reward you got
            # get likelihood
            l = likelihood(a,r, c, arms)

            b_post = l * b
            if np.sum(b_post) > 0:
                b_post = b_post / np.sum(b_post)
            else:
                # Fallback if numerical underflow occurs (rare)
                b_post = b

            # Prediction
            b = b_post @ T

    # compute log likelihood
    ll = np.nansum(np.log(probs))
    n_valid_trials = np.sum(~np.isnan(probs))
    nll = -ll
    if return_values == True:
        return nll, belief_states, probs
    return nll

def fit_si_T(x0, data, arms, return_values = False):
    print("Trying to fit SI with genralization transition matrix. 4 parameters")
    tau, c, prob_stay, decay = x0
    tau = np.exp(tau)
    chosen_actions = data[0]
    rewards_received = data[1]
    sessions = chosen_actions.shape[0]
    trials = chosen_actions.shape[1]

    b0 = np.ones(arms) / arms # intialize value of b0 to be 0.25 for each state (port)

    # transition matrix (symmetric, i.e, probability of switching from 1 --> 2 != 1 ----> 3
    T = np.full((4,4), (1-prob_stay))
    # fill diagonal with stay probability
    np.fill_diagonal(T, prob_stay)
    # Tij = pstay   if i = j
    # Tij = (1 − pstay) · e−β|i−j| / Z(i)   if i ≠ j
    for i in range(4):
        norm = np.sum(np.array([np.exp(-decay*np.abs(i-k)) for k in range(4) if k != i]))
        for j in range(4):
            if i != j:
                exponent = -decay * np.abs(i-j)
                
                T[i, j] = (1-prob_stay)*np.exp(exponent)/norm

    belief_states = np.zeros((sessions, trials, arms)) # will store the belief state of the model at each trial
    probs = np.full((sessions, trials), np.nan) # stores the probability of choosing each action on each trial using soft max
    for session in range(sessions):
        b = b0.copy()
        # belief_states[session, 0, :] = b0
        # probs[session, 0] = softmax(b0, tau)
        for trial in range(trials):
            action = chosen_actions[session, trial]
            if np.isnan(action):
                continue  # or break if trials are padded at the end
            belief_states[session,trial,:] = b
            # pick an action usig the curent belief and the 
            prob_a = softmax(b, tau) # return the probability of each action given current belief state

            a = int(action)
            r = rewards_received[session, trial]

            # probability of action chosen by animal being chosen
            probs[session, trial] = prob_a[a]
            # update belief state based on the action and reward you got
            # get likelihood
            l = likelihood(a,r, c, arms)

            b_post = l * b
            if np.sum(b_post) > 0:
                b_post = b_post / np.sum(b_post)
            else:
                # Fallback if numerical underflow occurs (rare)
                b_post = b

            # Prediction
            b = b_post @ T

    # compute log likelihood
    ll = np.nansum(np.log(probs))
    n_valid_trials = np.sum(~np.isnan(probs))
    nll = -ll
    if return_values == True:
        return nll, belief_states, probs
    return nll
#%%
# load data
def load_data_into_dict(block_subset = None, seed_n = 42): # block_subset subsets the dataframe to only those blocks
    rng = np.random.default_rng(seed_n)
    import pickle
    import pandas as pd
    # with open('/mnt/pve/Homes/homna/model_fitting/cleandf.pkl', 'rb') as f:
    #     df = pickle.load(f)
    df = pd.read_csv('/mnt/pve/Homes/homna/code/code-server/model_fitting/cleaned_df.csv')

    # subset for expert performance
    subset_df = df[(df.sess_bin>=4) & (df.task == 'unstr')]
    from dfLoading import subset_trials, add_block_groups
    
    # add block groups and select only first block 
    subset_df = add_block_groups(subset_df)
    if block_subset is not None:
        subset_df = subset_df[subset_df.block_group==block_subset] # only select trials from the first block
    trialsinsess = 50

    # then subset for blocks with atleast 50 trials
    subset_df = subset_trials(subset_df, trialsinsess=trialsinsess) # block 1, blocks which have atleast a 100 trials in the session, subselect 100 trials

    # np.random.seed(seed_n)

    test_sess_dict = {}
    # pick 10% sessions for testing
    # for animal in subset_df.animal.unique():
    #     unique_sess = subset_df[subset_df.animal == animal].session.nunique()
    #     test_sess = rng.choice(subset_df[subset_df.animal == animal].session.unique(), int(unique_sess*0.2), replace = False)
    #     test_sess_dict[animal] = test_sess
    #     subset_df.drop(subset_df[(subset_df.animal == animal) & (subset_df.session.isin(test_sess))].index, inplace = True)

    sess_dict = {key: group for key, group in subset_df.groupby('animal')}

    print('loaded data')
    return sess_dict, test_sess_dict


def process_data_blockwise(sess_dict, animal):
    subset_df = sess_dict[animal]
    num_sess = subset_df[subset_df.task == 'unstr'].sess_block.unique().shape[0]
    num_t = subset_df[subset_df.task == 'unstr'].groupby('sess_block').trial.count().max()
    actions = np.ones(shape = (num_sess, num_t))*np.nan
    rewards = np.ones(shape = (num_sess, num_t))*np.nan
    block_id = np.ones(shape = (num_sess, num_t))*np.nan
    #added
    rew_probs_env = np.ones(shape = (num_sess, num_t, 4))*np.nan
    for ind, (sess_bl, group) in enumerate(subset_df[subset_df.task == 'unstr'].groupby('sess_block')):
        actions[ind] = np.concatenate([group.port-1, np.ones(shape = (num_t - group.port.shape[0]))*np.nan])
        rewards[ind] = np.concatenate([group.reward, np.ones(shape = (num_t - group.reward.shape[0]))*np.nan])
        block_id[ind] = np.concatenate([group.block_group, np.ones(shape = (num_t - group.block_group.shape[0]))*np.nan])
        vals = group.rewprobfull.values  # array of strings        
        rew_probs = np.vstack([
            np.fromstring(v.strip("[]"), sep=" ")
            for v in vals
        ])  # shape (n_trials, 4)

        rew_probs_env[ind, :rew_probs.shape[0], :] = rew_probs

    # print('processed blockwise')
    return np.array([actions, rewards, block_id]), rew_probs_env


def optimize_single(animal, n, train_data, test_data, bounds, extra_params, csv_path, x0 = None, fit_fn= None):
    from scipy.optimize import minimize
    if fit_fn is None:
        fit_fn = fit_si
    data,rew_probs_env = process_data_blockwise(sess_dict = train_data, animal = animal)
    # num_trials for bic
    n_trials = (~np.isnan(data[0])).sum()
    k = bounds.shape[0]
    rng = np.random.default_rng()
    if x0 is None:
        x0 = rng.random(bounds.shape[0])
        x0 = ((bounds[:, 1] - bounds[:, 0]) * x0) + bounds[:, 0]
    arms = 4
    result = minimize(
        fit_fn,
        x0=x0,
        args=(data, arms),
        bounds=bounds,
        method='L-BFGS-B',
        options={'disp': True}
    )
    nll = result.fun
    optimized_x = result.x.tolist()
    success = result.success
    bic = k * np.log(n_trials) + 2 * nll

    row = [
        animal,
        n,
        ','.join(map(str, x0)),
        ','.join(map(str, optimized_x)),
        nll,
        success,
        test_data.get(animal, ''),
        bic
    ]

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    print(f'{animal} optim {n} done')
    print("Start:", x0)
    print("Opt:", result.x)
    print("Success:", result.success)
    print("Message:", result.message)
    return result

def optimize_single_param_recovery(repeat, n, repeat_data, bounds, extra_params, csv_path, x0 = None, fit_fn= None):
    from scipy.optimize import minimize
    if fit_fn is None:
        fit_fn = fit_si
    true_tau, true_c, true_gamma, true_k = repeat_data[repeat]["true_param"]
    data = repeat_data[repeat]["data"]
    # num_trials for bic
    n_trials = (~np.isnan(data[0])).sum()
    k = bounds.shape[0]
    rng = np.random.default_rng()
    if x0 is None:
        x0 = rng.random(bounds.shape[0])
        x0 = ((bounds[:, 1] - bounds[:, 0]) * x0) + bounds[:, 0]
    arms = 4
    result = minimize(
        fit_fn,
        x0=x0,
        args=(data, arms),
        bounds=bounds,
        method='L-BFGS-B',
        options={'disp': True}
    )
    nll = result.fun
    optimized_x = result.x.tolist()
    success = result.success
    bic = k * np.log(n_trials) + 2 * nll

    row = [
        repeat,
        n,
        true_tau, true_c, true_gamma, true_k,
        x0,
        optimized_x[0],
        optimized_x[1],
        optimized_x[2],
        optimized_x[3],
        nll,
        bic,
        success
    ]

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    print(f'Repeat {repeat} optim {n} done')
    print("Start:", x0)
    print("Opt:", result.x)
    print("Success:", result.success)
    print("Message:", result.message)
    return row

#%%
# run model
import numpy as np
import csv
if __name__ == '__main__':
    from supplementaryFunctions import tic, toc
    tic()
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import datetime
    from scipy.optimize import minimize
    from scipy.optimize import shgo
    from optim import resolve_optimizer
    from joblib import Parallel, delayed
    import os
    
    animals = ["Giratina", "Zacian", "Zorua", "Vanillish"]
    train_data, test_data = load_data_into_dict()
    extra_params = 4 # tau, c, gamma, decay
    n_optim = 10
    date = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    filepath_base = f'/mnt/pve/Homes/homna/HomnaData/model_fitting/SI_results/SI_all_blocks_generalization_matrix_{extra_params}params_'+date
    results = {}
    
    print("Running state inference model fitting with generalization matrix, all blocks")
    print(f"Results will be saved to: {filepath_base}.csv")
    print(f"Optimizing {len(animals)} animals with {n_optim} optimizations ")
    print(f"Number of parameters: {extra_params}")

    # create output dir if necessary
    os.makedirs(os.path.dirname(filepath_base), exist_ok=True)

    csv_path = f'{filepath_base}.csv'
    # Initialize CSV with header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['animal', 'optim_idx', 'x0', 'optimized_x', 'nll', 'success', 'test_sessions', 'bic'])

    # prepare args for each animal that exists in train_data
    animals_to_run = [a for a in animals if a in train_data]
    print("starting jobs")
    
    # # Create list of all (animal, n) pairs
    tasks = [(animal, n) for animal in animals_to_run for n in range(n_optim)]
    
    # # Use all available cores, or set n_jobs as needed
    n_jobs = 24
    start_params = [0.025, 0.5, 0.75]
    # start_params = [0.025, 0.5, 0.75, 0.25, 0.25, 0.25, 0.25]
    # # boundaries for parameters
    bounds = np.array([
        [np.log(0.0015), np.log(100)],    # tau
        [-1, 1],           # C
        [0 , 1],     # gamma (for genralization matrix this is p stay)
        [0,3]  # decay
    ])
    if (bounds.shape[0] != extra_params) | (len(start_params) != extra_params):
        print("Number of parameters mismatch error")
        print(f"num prams: {extra_params}")
        print(f"start params: {start_params}")
        print(f"bounds: {bounds}")
    all_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(optimize_single)(
            animal,
            n,
            train_data,
            test_data,
            bounds,
            extra_params,
            csv_path,
            x0= None,
            fit_fn = fit_si_T
        )
        for animal, n in tasks
    )

    #OPTUNA
    # def bias_constraint(trial):
    #     b1 = trial.params["b1"]
    #     b2 = trial.params["b2"]
    #     b3 = trial.params["b3"]

    #     return [b1 + b2 + b3 - 1.0]
    
    # with open(csv_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['animal', 'optimized_x', 'nll', 'success', 'bic'])
    # processed_data = {a: process_data_blockwise(train_data, a) for a in animals_to_run}
    # for a in animals_to_run[:5]:  # just first 5 animals
    #     data_arrays = processed_data[a][0]
    #     print(f"{a}:")
    #     print("  n_trials:", len(data_arrays[0]))
    #     print("  first 5 choices:", data_arrays[0][:5])
    #     print("  first 5 rewards:", data_arrays[1][:5])
    # print("=========================\n")
    # print("\n=== STRUCTURE CHECK ===")
    # example_animal = animals_to_run[0]
    # print("Type:", type(processed_data[example_animal]))
    # print("Length:", len(processed_data[example_animal]))
    # print("Element 0 type:", type(processed_data[example_animal][0]))
    # print("========================\n")
    # optimizer = resolve_optimizer("optuna")
    # optimizer.n_trials = 500
    # # optimizer.constraints_func = bias_constraint
    # optuna_bounds = [
    #     ("tau", (0.0015, 100)),
    #     ("c", (-1, 1)),
    #     ("gamma", (0, 1)),
    #     ("b1", (-10, 10)), 
    #     ("b2", (-10, 10)),
    #     ("b3", (-10, 10)),
    #     ("b4", (-10, 10))
    # ]


    # print(f"Starting parallel Optuna studies on {n_jobs} cores...")

    # all_results = Parallel(n_jobs=24)(
    #     delayed(optimizer.fit)(
    #         # Capture 'animal' correctly using 'a=animal'
    #         objective=lambda theta, a=animal: fit_si_bias_v2(theta, processed_data[a][0], arms=4),
    #         bounds=optuna_bounds, seeds=np.random.randint(0, 2**31 - 1, size=30).tolist()
    #     )
    #     for idx, animal in enumerate(animals_to_run)
    # )

    # # # Save the results after Parallel completes
    # with open(csv_path, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     for i, (best_fun, best_x, _) in enumerate(all_results):
    #         animal = animals_to_run[i]
            
    #         # Calculate BIC
    #         n_trials = (~np.isnan(processed_data[animal][0][0])).sum()
    #         k = len(optuna_bounds)
    #         bic = k * np.log(n_trials) + 2 * best_fun
            
    #         # Format for CSV: optimized_x as a comma-separated string
    #         row = [
    #             animal,
    #             ','.join(map(str, best_x)),
    #             best_fun,
    #             True,
    #             bic
    #         ]
    #         writer.writerow(row)

    # all_results = Parallel(n_jobs=n_jobs, backend="loky")(
    #     delayed(optimize_single)(animal, n, train_data, test_data, bounds, extra_params, csv_path, x0= [0.025, 0.5, 0.75]) # provide the model parameters you found works well when simulating data as a starting point
    #     for animal, n in tasks
    # )

    # write elapsed time log
    with open(f'{filepath_base}.log', 'a') as f:
        f.write(toc())

    print('Done. All results written to:', csv_path)



#%%
##########
# code to simulae si


def likelihood_sim(a, r, c):
    likelihoods = np.zeros(4)
    a = int(a)
    r = int(r)  # ensure integer comparison
    for state in range(4):
        if (a == state) and (r == 1):
            l = 0.5 * (1 + c)
        elif (a == state) and (r == 0):
            l = 0.5 * (1 - c)
        elif (a != state) and (r == 1):
            l = 0.5 * (1 - c)
        elif (a != state) and (r == 0):
            l = 0.5 * (1 + c)
        else:
            raise ValueError(f"Unhandled case: a={a}, r={r}, state={state}")
        likelihoods[state] = l
    return likelihoods

def simulate_si(sessions, trials, rew_probs, tau, c, gamma=None, T_unsymmetric=None, b0_bias=None, k = None):

    arms = 4
    tau = np.exp(tau)
    b0 = b0_bias.copy() if b0_bias is not None else np.ones(arms) * 0.25
    if T_unsymmetric is not None:
        T = T_unsymmetric.copy()
    elif gamma is not None:
        stay = 0.25 + 0.75 * gamma
        switch = (1 - stay) / 3  # Ensure 4-arm rows sum to 1.0
        T = np.full((arms, arms), switch)
        np.fill_diagonal(T, stay)
    sticky = np.zeros(arms)
    actions = np.zeros((sessions, trials))
    rewards = np.zeros((sessions, trials))
    belief_states = np.zeros((sessions, trials, arms))

    for session in range(sessions):
        b = b0.copy() # change reset every 3 sessions
        rp_s = rew_probs[session]
        for trial in range(trials):
            # first store the belief state on this trial before estimating for the next trial
            belief_states[session,trial,:] = b
            # pick an action
            if k is not None: # only if sticky is not None, add sticky bias to softmax otherwise calculate softmax without sticky bias
                prob_a = softmax_sticky(b, tau, sticky, k) # return the probability of each action given current belief state and sticky bias
            else:
                prob_a = softmax(b, tau) # return the probability of each action given current belief state
            
            # flip coin to pick action from these reward probabilities
            a_array = np.random.multinomial(1, prob_a)
            a = np.where(a_array)[0][0]
            sticky = np.zeros(arms)
            sticky[a] = 1 # update sticky bias for next trial

            # based on the action you picked determine if you got reward or not
            r = int(np.random.uniform() <= rp_s[a])

            actions[session, trial] = a
            rewards[session,trial] = r
            # update belief state based on the action and reward you got
            # get likelihood
            l = likelihood_sim(a,r, c)

            b_post = l * b
            if np.sum(b_post) > 0:
                b_post = b_post / np.sum(b_post)
            else:
                # Fallback if numerical underflow occurs (rare)
                b_post = b

            # Prediction
            b = b_post @ T
    return actions, rewards, belief_states

def simulate_si_block(sessions, trials, blocks, rew_probs, tau, c, gamma=None, T_unsymmetric = None, b0_bias = None, k = None):

    arms = 4
    tau = np.exp(tau)
    b0 = b0_bias.copy() if b0_bias is not None else np.ones(arms) * 0.25
    if T_unsymmetric is not None:
        T = T_unsymmetric.copy()
    elif gamma is not None:
        stay = 0.25 + 0.75 * gamma
        switch = (1 - stay) / 3  # Ensure 4-arm rows sum to 1.0
        T = np.full((arms, arms), switch)
        np.fill_diagonal(T, stay)
    sticky = np.zeros(arms)
    actions = np.zeros((sessions, blocks, trials))
    rewards = np.zeros((sessions,blocks, trials))
    belief_states = np.zeros((sessions, blocks, trials, arms))
    i = 0 # counter for the number of blocks (total = sessions * blocks)

    for session in range(sessions):
        b = b0.copy() # change reset every 3 sessions
        for block in range(blocks):
            rp_b = rew_probs[i] # stores the reward probability of the environment of that block
            for trial in range(trials):
                # first store the belief state on this trial before estimating for the next trial
                belief_states[session,block, trial,:] = b
                # pick an action
                if k is not None: # only if sticky is not None, add sticky bias to softmax otherwise calculate softmax without sticky bias
                    prob_a = softmax_sticky(b, tau, sticky, k) # return the probability of each action given current belief state and sticky bias
                else:
                    prob_a = softmax(b, tau) # return the probability of each action given current belief state
                
                # flip coin to pick action from these reward probabilities
                a_array = np.random.multinomial(1, prob_a)
                a = np.where(a_array)[0][0]
                sticky = np.zeros(arms)
                sticky[a] = 1 # update sticky bias for next trial
                
                # based on the action you picked determine if you got reward or not
                r = int(np.random.uniform() <= rp_b[a])

                actions[session,block, trial] = a
                rewards[session,block, trial] = r
                # update belief state based on the action and reward you got
                # get likelihood
                l = likelihood_sim(a,r, c)

                b_post = l * b
                if np.sum(b_post) > 0:
                    b_post = b_post / np.sum(b_post)
                else:
                    # Fallback if numerical underflow occurs (rare)
                    b_post = b

                # Prediction
                b = b_post @ T
            i += 1
    print(i)
    return actions, rewards, belief_states


def simulate_si_block_T(sessions, trials, blocks, rew_probs, tau, c, prob_stay, decay, k= None):

    arms = 4
    tau = np.exp(tau)
    b0 = np.ones(arms) * 0.25
    T = np.full((4,4), (1-prob_stay))
    # fill diagonal with stay probability
    np.fill_diagonal(T, prob_stay)
    # Tij = pstay   if i = j
    # Tij = (1 − pstay) · e−β|i−j| / Z(i)   if i ≠ j
    for i in range(4):
        norm = np.sum(np.array([np.exp(-decay*np.abs(i-k)) for k in range(4) if k != i]))
        for j in range(4):
            if i != j:
                exponent = -decay * np.abs(i-j)
                
                T[i, j] = (1-prob_stay)*np.exp(exponent)/norm
    sticky = np.zeros(arms)
    actions = np.zeros((sessions, blocks, trials))
    rewards = np.zeros((sessions,blocks, trials))
    belief_states = np.zeros((sessions, blocks, trials, arms))
    i = 0 # counter for the number of blocks (total = sessions * blocks)

    for session in range(sessions):
        b = b0.copy() # change reset every 3 sessions
        for block in range(blocks):
            rp_b = rew_probs[i] # stores the reward probability of the environment of that block
            for trial in range(trials):
                # first store the belief state on this trial before estimating for the next trial
                belief_states[session,block, trial,:] = b
                # pick an action
                if k is not None: # only if sticky is not None, add sticky bias to softmax otherwise calculate softmax without sticky bias
                    prob_a = softmax_sticky(b, tau, sticky, k) # return the probability of each action given current belief state and sticky bias
                else:
                    prob_a = softmax(b, tau) # return the probability of each action given current belief state
                
                # flip coin to pick action from these reward probabilities
                a_array = np.random.multinomial(1, prob_a)
                a = np.where(a_array)[0][0]
                sticky = np.zeros(arms)
                sticky[a] = 1 # update sticky bias for next trial
                
                # based on the action you picked determine if you got reward or not
                r = int(np.random.uniform() <= rp_b[a])

                actions[session,block, trial] = a
                rewards[session,block, trial] = r
                # update belief state based on the action and reward you got
                # get likelihood
                l = likelihood_sim(a,r, c)

                b_post = l * b
                if np.sum(b_post) > 0:
                    b_post = b_post / np.sum(b_post)
                else:
                    # Fallback if numerical underflow occurs (rare)
                    b_post = b

                # Prediction
                b = b_post @ T
            i += 1
    print(i)
    return actions, rewards, belief_states

def simulate_si_block_real_data(data, sessions, trials, blocks, tau, c, gamma=None, k=None):
    """
    data: array containing actual actions and rewards experienced by the animal
    sessions: integer, number of sessions to simulate
    blocks: an array of shape (sessions,) that contains the number of blocks in each session. This is used to determine the reward probabilities for each block in each session
    trials: an array of shape (sessions, blocks) that contains the number of trials for each session and block. This is used to determine how many trials to simulate for each block in each
    rew_probs: an array of shape (sessions, blocks, arms) that contains the reward probabilities for each block in each session. This is used to determine the reward probabilities for each block in each session
    tau: float, temperature parameter for softmax action selection
    c: float, confidence parameter for belief updating
    gamma: float, transition probability for staying in the same state (if T_unsymmetric is not provided)
    k: float, strength of sticky bias (if not None)
    """
    
    arms = 4
    tau = np.exp(tau)
    actions = data[0] # array of shape (sessions, blocks, trials) that contains the actual actions taken by the animal in each session, block and trial. This is used to determine the action taken by the animal on each trial in order to update the belief state based on the actual experience of the animal rather than the simulated action. This allows us to simulate the belief states and probabilities of action selection that would have occurred given the actual actions and rewards experienced by the animal.
    rewards = data[1] # array of shape (sessions, blocks, trials) that contains the actual rewards received by the animal in each session, block and trial. This is used to determine the reward received by the animal on each trial in order to update the belief state based on the actual experience of the animal rather than the simulated reward. This allows us to simulate the belief states and probabilities of action selection that would have occurred given the actual actions and rewards experienced by the animal.
    b0 = np.ones(arms) * 0.25


    stay = 0.25 + 0.75 * gamma
    switch = 0.25 - (0.25 * gamma)  # Ensure 4-arm rows sum to 1.0
    T = np.full((arms, arms), switch)
    np.fill_diagonal(T, stay)
    sticky = np.zeros(arms)
    max_blocks = int(np.max(blocks))  # blocks is (sessions,) so just take max directly
    max_trials = int(np.nanmax(trials)) # trials is (sessions, blocks), nanmax flattens and ignores nans # get the maximum number of trials across blocks and sessions to determine the shape of the actions and rewards arrays
    belief_states = np.full((sessions, max_blocks, max_trials, arms), np.nan)
    action_probs = np.full((sessions, max_blocks, max_trials, arms), np.nan)
    i = 0 # counter for the number of blocks (total = sessions * blocks)

    for session in range(sessions):
        b = b0.copy() # change reset every 3 sessions
        for block in range(int(blocks[session])): # simulate for all blocks in that session

            for trial in range(int(trials[session, block])):
                print(f"Simulating session {session}, block {block}, trial {trial}")
                # first store the belief state on this trial before estimating for the next trial
                belief_states[session,block, trial,:] = b
                # pick an action
                if k is not None: # only if sticky is not None, add sticky bias to softmax otherwise calculate softmax without sticky bias
                    prob_a = softmax_sticky(b, tau, sticky, k) # return the probability of each action given current belief state and sticky bias
                    action_probs[session, block, trial, :] = prob_a
                else:
                    prob_a = softmax(b, tau) # return the probability of each action given current belief state
                    action_probs[session, block, trial, :] = prob_a            

                a = (actions[session, block, trial]) # if action is nan (not available in data), use simulated action, otherwise use actual action from data
                if np.isnan(a):
                    print(f"Action is nan for session {session}, block {block}, trial {trial}. Skipping trial.")
                    continue
                a = int(a) # convert action to integer

                sticky = np.zeros(arms)
                sticky[a] = 1 # update sticky bias for next trial
                
                # based on the action you picked determine if you got reward or not
                r = rewards[session, block, trial] # if reward is nan (not available in data), use simulated reward, otherwise use actual reward from data
                if np.isnan(r):
                    print(f"Reward is nan for session {session}, block {block}, trial {trial}. Skipping trial.")
                    continue


                # update belief state based on the action and reward you got
                # get likelihood
                l = likelihood_sim(a,r, c)

                b_post = l * b
                if np.sum(b_post) > 0:
                    b_post = b_post / np.sum(b_post)
                else:
                    # Fallback if numerical underflow occurs (rare)
                    b_post = b

                # Prediction
                b = b_post @ T
            i += 1
    print(i)
    return belief_states, action_probs

#%%
# genereate rewad probabilities for simulation

def fxn(mean, arms, permute = False):
    x = np.linspace(1, arms, arms)
    sig = 1.75/2
#     amp = 1/(sig*np.sqrt(2*np.pi))
    amp = 0.7
    vo = 0.1
    gx = (amp*np.exp(-0.5*((x-mean)**2)/(sig**2)))+vo
    if permute:
        gx = np.random.permutation(gx)
    return gx

#%%
def recover_param(bounds, csv_path, trial = 100, sessions = 100, n_repeats = 100, n_optim = 10):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'repeat', 'optim_idx',
            'true_tau', 'true_c', 'true_gamma', "true_k",
            'x0',
            'recovered_tau', 'recovered_c', 'recovered_gamma', "recovered_k",
            'nll', 'bic', 'success'
        ])
    tasks = [(repeat, n) for repeat in range(n_repeats) for n in range(n_optim)]
    arms = 4
    rew_probs = [fxn(np.random.randint(1, 4+1), 4, True) for i in range(sessions)]
    extra_params = bounds.shape[0]
    repeat_data = {}
    for repeat in range(n_repeats):
        true_tau = random.uniform(bounds[0,0], bounds[0,1])
        true_c = random.uniform(bounds[1,0], bounds[1,1])
        true_gamma = random.uniform(bounds[2,0], bounds[2,1])
        true_k = random.uniform(bounds[3,0], bounds[3,1])

        sim_actions, sim_rewards, sim_belief_states = simulate_si(tau=true_tau, c=true_c, gamma=true_gamma, sessions = sessions, trials = trial, rew_probs = rew_probs, k=true_k)
        data = np.array([sim_actions, sim_rewards])
        repeat_data[repeat] = {"true_param": (true_tau, true_c, true_gamma, true_k), "data": data}
    
    n_jobs = min(24, len(tasks))

    all_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(optimize_single_param_recovery)(
            repeat = repeat,
            n=  n,
            repeat_data = repeat_data,
            bounds = bounds,
            extra_params = extra_params,
            csv_path = csv_path,
            x0 = None,
            fit_fn = fit_si_sticky
        )
        for repeat, n in tasks
    )
    print(f"Parameter recovery done. Results saved to: {csv_path}")
    all_results = pd.DataFrame(all_results, columns=[
            'repeat', 'optim_idx',
            'true_tau', 'true_c', 'true_gamma', "true_k",
            'x0',
            'recovered_tau', 'recovered_c', 'recovered_gamma', "recovered_k",
            'nll', 'bic', 'success'
        ])
    return all_results

# check param recovery
# def run_single_optim(i, data, bounds, arms, x0 = None):
#     # for param recovery check
#     # random start
#     if x0 is None:
#         x0 = np.random.rand(3)
#         x0 = bounds[:,0] + (bounds[:,1] - bounds[:,0]) * x0

#     #with sticky bonus
#         res = minimize(
#         lambda x: fit_si(x, data, arms, False, True),
#         x0=x0,
#         bounds=bounds,
#         method='L-BFGS-B'
#     )

#     print(f"Done with optimization {i}")
#     return res

# print("Check Param recovery")
# # generate 3 sets of parameters tau, alpha, q_init from specified bounds
# param = []
# bounds = np.array([
#     [0.01, 1],   # temp not inverse temp
#     [-1, 1],   # c
#     [0, 1],      # gamma
# ])
# print("Parameter bounds for param recovery:")
# print(bounds)
# # pick a random set of parameters within bounds
# rng = np.random.default_rng()
# x0 = rng.random(bounds.shape[0])
# x0 = ((bounds[:, 1] - bounds[:, 0]) * x0) + bounds[:, 0]
# print(f"Parameter sets used for simulation: {x0}")
# print("Simulating data...")
# # simulate data for each parameter set
# print("Simulating data without blocks")
# sim_actions, sim_rewards, sim_q_values = simulate_si_block(x0, sessions=1000, trials=100, blocks=1, rew_probs=[fxn(np.random.randint(1, 4+1), 4, False) for i in range(1000)], with_sticky=True)
# data = np.array([sim_actions, sim_rewards])
# print("Fitting model to simulated data...")
# # fit model to simulated data
# # run optimization multiple times to find best fit
# n_optims = 10
# results = []
# for i in range(n_optims):
#     res = run_single_optim(i, data, bounds, arms=4, x0=None)
#     results.append(res)
# print("Optimization results:")
# results = sorted(results, key=lambda x: x.fun)  # sort by NLL
# best_fit = results[0]
# print(best_fit.x)

# #%%
# #---- simulate ----
# print("starting simulation")
# true_params = (0.025, 0.5, 0.75) # tau, c , gamma
# trials = 100
# sessions = 1000
# rew_probs = [fxn(np.random.randint(1, 4+1), 4, False) for i in range(sessions)]
# actions, rewards, _ = simulate_si(
#     tau=true_params[0],
#     c=true_params[1],
#     gamma=true_params[2],
#     sessions=sessions,
#     trials=trials,
#     rew_probs=rew_probs
# )

