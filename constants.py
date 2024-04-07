
import numpy as np
import AlgorithmsModule.Algorithms as A
import RegretMinimisersModule.RegretMinimizer as R


# SEEDS - Define the seed for reproducibility - some instantiations may generate infinite values issues due to the bandit computations
seeds = [20, 10, 30, 40, 51, 60, 70, 80, 90, 100]

#%% Is it a test?
test = False
if test:
    num_experiments = 10
    T = 100
    n = 3
    m = 1
    B = 10
    stored = True
    store_new_data = False
    show_plots = False
    verbose_results = False
    verbose_game = False

    best_FDs = []  # List of best Fixed Distributions. If the data is not generated it will be computed automatically.
    best_FDs_rewards = [] # List of corresponding total rewards of the best Fixed Distributions.
    worst_FDs = []  # List of worst Fixed Distributions. If the data is not generated it will be computed automatically.
    scenario = "best"  # Whether to use the best Fixed Distribution as prediction or to use the fed prediction.
    
    mean_rewards = [0.9, 1.2]
    mean_costs = [1.2, 0.5]
    sigma_rewards = [0.5, 0.5]
    sigma_costs = [0.5, 0.5]
    

# EXPERIMENTS - Choose the parameters to run experiments
else:
    num_experiments = 10  # Number of experiments
    best_FDs = []  # List of best Fixed Distributions. If the data is not generated it will be computed automatically.
    best_FDs_rewards = [] # List of corresponding total rewards of the best Fixed Distributions.
    worst_FDs = []  # List of worst Fixed Distributions. If the data is not generated it will be computed automatically.
    stored = True  # Whether the data are already stored or not.
    store_new_data = False  # Whether to store new data or not.
    scenario = "best"  # Whether to use the best Fixed Distribution as prediction or to use the fed prediction.
    show_plots = False  # Regardless, plots are saved in the directory. It indicates whether to show the plots or not.
    verbose_results = False  # prints the values of the results in a table for each single experiment.
    verbose_game = False  # gives a complete understanding of each iteration of the game.

    # GAME - Define structural variables
    T = 50000  # Number of time steps
    n = 5  # Number of actions
    m = 1  # Number of knapsacks
    B = 5000 # Long term budget

    # GENERATED DATA - Deifne the parameters for generating lognormal data
    mean_rewards = [0.9, 1.2, 0.1, 0.4]
    mean_costs = [0.8, 0.6, 1.2, 0.5]
    sigma_rewards = [0.5, 0.5, 0.5, 0.5]
    sigma_costs = [0.5, 0.5, 0.5, 0.5]
    assert len(mean_rewards) == n-1
    assert len(mean_costs) == n-1
    assert len(sigma_rewards) == n-1
    assert len(sigma_costs) == n-1

# ALGORITHMS GENERAL - Define the algorithm to run
rp_starting_point = np.ones(n)/n
rd_starting_point = np.ones(m)/(2*(m*B/T))
learning_rate_primal = 1/np.sqrt(T)
assert np.sum(rp_starting_point) == 1
assert np.sum(rd_starting_point) < 1/(B/T)

# ALGORITHMS PARAMETERS - Define parameters for the algorithms
parameters_pd_ff = {  # For primal_dual full feedback
    "algorithm_name": "primal_dual_ff",
    "algorithm": A.primal_dual,
    "bandit": False,
    "RP": R.Hedge(starting_point=rp_starting_point, learning_rate=learning_rate_primal, nActions=n),
    "RD": R.DualRegretMinimizer(starting_point=rd_starting_point, learning_rate=1/np.sqrt(T), rho=B/T),
    "T": T,
    "n": n,
    "m": m,
    "B": B,
    "rho": B/T,
    "learning_rate": 1/np.sqrt(T)
}

parameters_pd_bandit = {  # For primal_dual bandit feedback
    "algorithm_name": "primal_dual_bandit",
    "algorithm": A.primal_dual,
    "bandit": True,
    "RP": R.EXP3(starting_point=rp_starting_point, learning_rate=learning_rate_primal, nActions=n),
    "RD": R.DualRegretMinimizer(starting_point=rd_starting_point, learning_rate=1/np.sqrt(T), rho=B/T),
    "T": T,
    "n": n,
    "m": m,
    "B": B,
    "rho": B/T,
    "learning_rate": 1/np.sqrt(T)
}


parameters_augmented_ff = {  # For augmented learning algorothm full feedback
    "algorithm_name": "adversarial_with_prediction_ff",
    "algorithm": A.adversarial_with_prediction,
    "bandit": False,
    "learning_rate": 1/np.sqrt(T),
    "RP": R.Hedge(starting_point=rp_starting_point, learning_rate=learning_rate_primal, nActions=n),
    "RD": R.DualRegretMinimizer(starting_point=rd_starting_point, learning_rate=1/np.sqrt(T), rho=B/T),
    "T": T,
    "n": n,
    "m": m,
    "B": B,
    "rho": B/T,
    "p": 0.9,
    "mu": 0.0,
    "mixed_action_predicted": np.array([0.85, 0.05, 0.05, 0.05, 0.0])
}


parameters_augmented_bandit = {  # For augmented learning algorothm bandit feedback
    "algorithm_name": "adversarial_with_prediction_bandit",
    "algorithm": A.adversarial_with_prediction,
    "bandit": True,
    "learning_rate": 1/np.sqrt(T),
    "RP": R.EXP3(starting_point=rp_starting_point, learning_rate=learning_rate_primal, nActions=n),
    "RD": R.DualRegretMinimizer(starting_point=rd_starting_point, learning_rate=1/np.sqrt(T), rho=B/T),
    "T": T,
    "n": n,
    "m": m,
    "B": B,
    "t": 0,
    "rho": B/T,
    "p": 0.9,
    "mu": 0.0,
    "mixed_action_predicted": np.array([0.85, 0.05, 0.05, 0.05, 0.0])
}
