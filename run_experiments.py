"""
This script runs experiments for the Adversarial Bandit with Knapsacks problem.
It defines the necessary variables, algorithms, and data generation functions.
The main function, run_experiments, generates data, runs the experiments, and returns the results.
"""

import RegretMinimerzModule.RegretMinimizer as R
import DataGeneratorModule.DataGenerator as DG
import AlgorithmsModule.Algorithms as A
import gurobipy as gp
import numpy as np
import Game as G
import pandas as pd
from tabulate import tabulate

# Define structural variables
T = 10000  # Number of time steps
n = 4  # Number of actions
m = 1  # Number of knapsacks
B = 5000  # Knapsack capacity # Already at 50000-4000 the algorithm outperforms ours

# Define the algorithm to run
rp_starting_point = np.ones(n)/n
rd_starting_point = np.ones(m)/(m*B/T)
learning_rate_primal = 1/np.sqrt(T)

# Define parameters for the algorithm
parameters_pd_ff = {  # For primal_dual full feedback
    "algorithm_name": "primal_dual",
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
    "algorithm_name": "primal_dual",
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


parameters_augmented_ff = {
    "algorithm_name": "adversarial_with_prediction",
    "algorithm": A.adversarial_with_prediction,
    "bandit": False,
    "learning_rate": 1/np.sqrt(T),
    "RP": R.Hedge(starting_point=rp_starting_point, learning_rate=learning_rate_primal, nActions=n),
    "RD": R.DualRegretMinimizer(starting_point=rd_starting_point, learning_rate=1/np.sqrt(T), rho=T/B),
    "T": T,
    "n": n,
    "m": m,
    "B": B,
    "rho": B/T,
    "p": 0.5,
    "nu": 0.0,
    "mu": 0.0,
    "mixed_action_predicted": np.array([0.85, 0.05, 0.05, 0.05])
}

parameters_augmented_bandit = {
    "algorithm_name": "adversarial_with_prediction",
    "algorithm": A.adversarial_with_prediction,
    "bandit": True,
    "learning_rate": 1/np.sqrt(T),
    "RP": R.EXP3(starting_point=rp_starting_point, learning_rate=learning_rate_primal, nActions=n),
    "RD": R.DualRegretMinimizer(starting_point=rd_starting_point, learning_rate=1/np.sqrt(T), rho=T/B),
    "T": T,
    "n": n,
    "m": m,
    "B": B,
    "rho": B/T,
    "p": 0.5,
    "nu": 0.0,
    "mu": 0.0,
    "mixed_action_predicted": np.array([0.85, 0.05, 0.05, 0.05])
}

# Define action means
mean_rewards = [0.8, 0.9, 0.3, 0.5]
mean_costs = [0.6, 0.8, 0.9, 0.5]
sigma_rewards = [0.5, 0.5, 0.5, 0.5]
sigma_costs = [0.5, 0.5, 0.5, 0.5]

def compute_best_FD(rewards, costs, B):
    """
    Computes the best feasible solution for the knapsack problem.

    Args:
        rewards (numpy.ndarray): Array of shape (T, n) representing the rewards for each action at each time step.
        costs (numpy.ndarray): Array of shape (T, n, m) representing the costs for each action and knapsack at each time step.
        B (float): Knapsack capacity.

    Returns:
        numpy.ndarray: Array of shape (n,) representing the best feasible solution for the knapsack problem.
        float: The objective value of the best feasible solution.
    """
    n = rewards.shape[1]+1
    T = rewards.shape[0]
    m = costs.shape[2]
    rewards_aux = rewards.copy()
    costs_aux = costs.copy()
    rewards_aux = np.hstack((rewards_aux, np.zeros((T, 1))))
    costs_aux = np.hstack((costs_aux, np.zeros((T, 1, m))))

    # Define the model and its variables
    model = gp.Model("lp")
    xi = model.addVars(n, lb=0, ub=1, name="xi")
    model.update()

    model.Params.LogToConsole = 0

    obj = gp.quicksum(gp.quicksum(rewards_aux[:, i] * xi[i]) for i in range(n))
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    model.addConstr(gp.quicksum(xi[i] for i in range(n)) == 1)
    for j in range(m):
        model.addConstr(gp.quicksum(gp.quicksum(costs_aux[:, i, j] * xi[i]) for i in range(n)) <= B)
    model.optimize()
    return np.array([model.getVars()[i].X for i in range(n)]), model.ObjVal

def generate_report(game):
    # Create a dictionary with the desired fields
    report_data = {
        "Total Pseudo Regret": [game.cumulative_pseudo_regret.copy()[-1]],
        "Total Reward": [game.cumulative_reward.copy()[-1]],
        "Cumulative Cost": [game.cumulative_cost.copy()[-1][0]]
    }

    last_iteration = None
    for i, action in enumerate(game.vector_of_actions):
        if action >= 0.0 and action < n:
            last_iteration = i

    report_data["Stopping Iteration"] = [last_iteration]

    # Create a DataFrame from the dictionary
    report_df = pd.DataFrame(report_data)

    # Return the DataFrame
    return report_df

def get_average_result(results, T, B, algorithm_name, bandit):
    average_result = results.mean() 
    average_result["Algorithm"] = algorithm_name
    average_result["Horizon"] = T
    average_result["Budget"] = B
    average_result["Bandit"] = bandit
    return average_result
    

def show_average_result(average_result):
    headers = average_result.index.tolist()
    values = average_result.values.tolist()
    table = zip(headers, values)
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

def run_experiments(num_experiments, parameters, generate_data=True, best_FDs=[], best_scenario=False, verbose=False):
    """
    Runs experiments for the Adversarial Bandit with Knapsacks problem.

    Args:
        num_experiments (int): Number of experiments to run.
        generate_data (bool): Whether to generate new data for each experiment.
        best_FDs (list): List to store the best feasible solutions for each experiment.
        return_best_FD (bool): Whether to return the best feasible solutions.

    Returns:
        list: List of dictionaries containing the results of each experiment.
        list: List of best feasible solutions for each experiment (if return_best_FD is True).
    """
    results = pd.DataFrame()

    if generate_data:
        data_generator = DG.DataGenerator(n, m, T, B)

    for i in range(num_experiments):

        # Generate data
        if generate_data:
            # data_generator.generate_data_lognormal(mean_rewards, sigma_rewards, mean_costs, sigma_costs)
            # data_generator.generate_data_lognormal_adversarial(mean_rewards, sigma_rewards, mean_costs, sigma_costs)
            data_generator.generate_data_lognormal_adversarial_v2(mean_rewards, sigma_rewards, mean_costs, sigma_costs, rate=1000)
            rewards = data_generator.data[0].copy()
            costs = data_generator.data[1].copy()
            # np.savetxt(f"experiment_data//rewards_{i}_{T}_{B}.csv", rewards, delimiter=",")
            # TODO: Fix cause being 3-D "savetxt" does not work. np.savetxt(f"experiment_data//costs_{i}_{T}_{B}.csv", costs, delimiter=",")

            best_FD = compute_best_FD(rewards, costs, B)[0]
            print(best_FD)
            best_FDs.append(best_FD)
            
        if best_scenario:
            best_FD_sum = np.sum(best_FD[:-1])
            parameters["mixed_action_predicted"] = best_FD[:-1] / best_FD_sum
        print(parameters["mixed_action_predicted"])
        # Run experiment
        game = G.Game(T, B, n, m, bandit=parameters["bandit"])
        game.run(rewards, costs, parameters, best_FD)

        res_row = generate_report(game)
        results = pd.concat([results, pd.DataFrame(res_row)], ignore_index=True)
        # print("Results: ", results)

        # Reset algorithm
        parameters["RP"].reset()
        parameters["RD"].reset()
        parameters.pop("B_current_A", None)
        parameters.pop("B_current_WC", None)
        data_generator.reset()
        game.reset()
    
    average_results = get_average_result(results, parameters["T"], parameters["B"], parameters["algorithm_name"], parameters["bandit"]).copy()
    if verbose:
        show_average_result(average_results)
    return average_results


num_experiments = 4
generate_data = True
best_FDs = []
best_scenario = True
verbose = True
# parameters = parameters_pd_ff
# parameters = parameters_pd_bandit
# parameters = parameters_augmented_ff
# parameters = parameters_augmented_bandit
res = run_experiments(num_experiments, parameters_pd_ff, generate_data, best_FDs, best_scenario, verbose)
res = run_experiments(num_experiments, parameters_pd_bandit, generate_data, best_FDs, best_scenario, verbose)
res = run_experiments(num_experiments, parameters_augmented_ff, generate_data, best_FDs, best_scenario, verbose)
res = run_experiments(num_experiments, parameters_augmented_bandit, generate_data, best_FDs, best_scenario, verbose)

# Should I allow the algorithm to play the outside action?