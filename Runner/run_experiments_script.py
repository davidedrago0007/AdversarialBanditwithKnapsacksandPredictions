"""
This script runs experiments for the Adversarial Bandit with Knapsacks problem.
It defines the necessary variables, algorithms, and data generation functions.
The main function, run_experiments, generates data, runs the experiments, and returns the results.
"""

import GameModule.Game as G

import gurobipy as gp
import pandas as pd
import numpy as np
import pickle

from tabulate import tabulate

def compute_best_FD(rewards, costs, B):
    """
    Computes the best offline fixed feasible solution for the knapsack problem.

    Args:
        rewards (numpy.ndarray): Array of shape (T, n) representing the rewards for each action at each time step.
        costs (numpy.ndarray): Array of shape (T, n, m) representing the costs for each action and knapsack at each time step.
        B (float): Long term budget.

    Returns:
        numpy.ndarray: Array of shape (n,) representing the best feasible solution for the knapsack problem.
        float: The objective value of the best feasible solution.
    """
    n = rewards.shape[1]
    T = rewards.shape[0]
    m = costs.shape[2]

    # Define the model and its variables
    model = gp.Model("lp")
    xi = model.addVars(n, lb=0, ub=1, name="xi")
    model.update()

    # Do not log to console
    model.Params.LogToConsole = 0

    # Define the objective function
    obj = gp.quicksum(gp.quicksum(rewards[:, i] * xi[i]) for i in range(n))
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(xi[i] for i in range(n)) == 1)

    # Add constraints
    for j in range(m):
        model.addConstr(gp.quicksum(gp.quicksum(costs[:, i, j] * xi[i]) for i in range(n)) <= B)
    
    # Optimize the model
    model.optimize()
    return np.array([model.getVars()[i].X for i in range(n)]), model.ObjVal

def compute_worst_FD(rewards, costs, B):
    """
    Computes the worst-case fractional decision (FD) for a given set of rewards and costs.

    Parameters:
    - rewards (numpy.ndarray): Array of shape (T, n) representing the rewards for each time step and item.
    - costs (numpy.ndarray): Array of shape (T, n, m) representing the costs for each time step, item, and constraint.
    - OPT_FD (float): The optimal fractional decision value.

    Returns:
    - worst_FD (numpy.ndarray): Array of shape (n,) representing the worst-case fractional decision values for each item.
    - obj_val (float): The objective value of the optimization problem.
    """
    n = rewards.shape[1]
    T = rewards.shape[0]
    m = costs.shape[2]

    # Define the model and its variables
    model = gp.Model("lp")
    xi = model.addVars(n, lb=0, ub=1, name="xi")
    model.update()

    # Do not log to console
    model.Params.LogToConsole = 0

    # Define the objective function
    obj = gp.quicksum(gp.quicksum(rewards[:, i] * xi[i]) for i in range(n))
    model.setObjective(obj, gp.GRB.MINIMIZE)
    model.addConstr(gp.quicksum(xi[i] for i in range(n)) == 1)
   
    # Add constraints
    for j in range(m):    
        model.addConstr(gp.quicksum(gp.quicksum(costs[:, i, j] * xi[i]) for i in range(n)) >= 0.95*B)
        model.addConstr(gp.quicksum(gp.quicksum(costs[:, i, j] * xi[i]) for i in range(n)) <= B)
    
    # Optimize the model
    model.optimize()
    return np.array([model.getVars()[i].X for i in range(n)]), model.ObjVal

def generate_report(game, optimal_reward, tvd):
    """
    Generate a report containing various metrics and statistics for the given game.

    Parameters:
    - game: The game object containing the data for the experiment.
    - optimal_reward: The optimal reward for the game.
    - tvd: The total variation distance for the game.

    Returns:
    - report_df: A pandas DataFrame containing the report data.

    """
    # Create a dictionary with the desired fields
    report_data = {
        #"Total Pseudo Regret": [game.cumulative_pseudo_regret.copy()[-1]],
        "Total Reward": [game.expected_cumulative_reward.copy()[-1]],
        "Cumulative Cost": [game.cumulative_cost.copy()[-1][0]]
    }

    last_iteration = None
    for i, action in enumerate(game.vector_of_actions):
        if action >= 0.0 and action < game.n:
            last_iteration = i

    report_data["Stopping Iteration"] = [last_iteration]
    report_data["Optimal Reward"] = [optimal_reward]
    report_data["TVD"] = [tvd]

    # Create a DataFrame from the dictionary
    report_df = pd.DataFrame(report_data)

    # Return the DataFrame
    return report_df

def get_average_result(results, T, B, bandit):
    """
    Calculate the average result of the given results.

    Args:
        results (pandas.DataFrame): The results of the experiments.
        T (int): The horizon of the experiments.
        B (int): The budget of the experiments.
        bandit (str): The type of bandit used in the experiments.

    Returns:
        pandas.Series: The average result, including the mean total reward, standard deviation of total reward,
        horizon, budget, and bandit type.
    """
    average_result = results.mean() 
    std_result = results.std()
    average_result["Standard Deviation"] = std_result["Total Reward"]
    average_result["Horizon"] = T
    average_result["Budget"] = B
    average_result["Bandit"] = bandit
    
    return average_result
    
def show_average_result(average_result):
    """
    Display the average result in a tabular format.

    Args:
        average_result (pandas.DataFrame): A DataFrame containing the average result.

    Returns:
        None
    """
    headers = average_result.index.tolist()
    values = average_result.values.tolist()
    table = zip(headers, values)
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

def generate_data(seeds, number_of_experiments, data_generator, budget, mean_rewards, sigma_rewards, mean_costs, sigma_costs, rate=1000, store=True, test=False):
    """
    Generate data for adversarial bandit with knapsacks experiments.

    Args:
        seeds (list): List of random seeds for data generation.
        number_of_experiments (int): Number of experiments to run.
        data_generator (object): Object responsible for generating data.
        budget (float): Budget for the knapsack problem.
        mean_rewards (float): Mean value for reward generation.
        sigma_rewards (float): Standard deviation for reward generation.
        mean_costs (float): Mean value for cost generation.
        sigma_costs (float): Standard deviation for cost generation.
        rate (int, optional): Rate parameter for data generation. Defaults to 1000.
        store (bool, optional): Flag to indicate whether to store the generated data. Defaults to True.

    Returns:
        tuple: A tuple containing the generated data, best FDs, best FDs rewards, and worst FDs.
    """
    data = dict()
    best_FDs = []
    best_FDs_rewards = []
    worst_FDs = []

    for i in range(number_of_experiments):
        np.random.seed(seeds[i])
        data_generator.generate_data_lognormal(mean_rewards, sigma_rewards, mean_costs, sigma_costs)
        rewards = data_generator.data[0].copy()
        costs = data_generator.data[1].copy()
        # np.savetxt(f"experiment_data//rewards_{i}_{T}_{B}.csv", rewards, delimiter=",")
        best_FD_results = compute_best_FD(rewards, costs, budget)
        best_FD = best_FD_results[0]
        best_FDs.append(best_FD)
        best_FDs_rewards.append(best_FD_results[1])
        data[f"experiment_{i}"] = (data_generator.data[0].copy(), data_generator.data[1].copy())
        
        worst_FD_result = compute_worst_FD(rewards, costs, budget)
        worst_FD = worst_FD_result[0]
        worst_FDs.append(worst_FD)

        data_generator.reset()

    if store:
        file_name = 'data.pkl' if not test else 'data_test.pkl'
        with open(file_name, 'wb') as file:
            file.truncate(0)
            pickle.dump((data, best_FDs, best_FDs_rewards, worst_FDs), file)
        
    return data, best_FDs, best_FDs_rewards, worst_FDs

def run_experiment(parameters, data, best_FD, best_FD_reward, worst_FD, noise_vector, scenario, verbose_game=False):
    """
    Runs a single experiment for the Adversarial Bandit with Knapsacks problem.

    Args:
        num_experiments (int): Number of experiments to run.
        generate_data (bool): Whether to generate new data for each experiment.
        best_FDs (list): List to store the best feasible solutions for each experiment.
        return_best_FD (bool): Whether to return the best feasible solutions.

    Returns:
        list: List of dictionaries containing the results of each experiment.
        list: List of best feasible solutions for each experiment (if return_best_FD is True).
    """
    T = parameters['T']
    B = parameters['B']
    n = parameters['n']
    m = parameters['m']

    rewards = data[0]
    costs = data[1]

    if scenario == "best":
        parameters["mixed_action_predicted"] = best_FD.copy()
        # We add noise to the prediction
        value = np.abs(parameters["mixed_action_predicted"] + noise_vector)
        value = value / np.sum(value)
        parameters["mixed_action_predicted"] = value
    elif scenario == "worst":
        parameters["mixed_action_predicted"] = worst_FD.copy()

    tvd = np.sum(np.abs(parameters["mixed_action_predicted"] - best_FD)) / 2
    # print(parameters["mixed_action_predicted"])
    # Run experiment
    game = G.Game(T, B, n, m, bandit=parameters["bandit"])
    game.run(rewards, costs, parameters, best_FD, verbose=verbose_game)

    res_row = generate_report(game, best_FD_reward, tvd)
    # print("Results: ", results)

    return res_row, game

def run(seeds, num_experiments, parameters, data_dict, best_FDs, best_FDs_rewards, worst_FDs, noise=None, scenario="best", verbose_results=False, verbose_game=False):
    """
    Run the experiments for the adversarial bandit with knapsacks problem.

    Args:
        seeds (list): List of random seeds for reproducibility.
        num_experiments (int): Number of experiments to run.
        parameters (dict): Dictionary of algorithm parameters.
        data_dict (dict): Dictionary of experiment data.
        best_FDs (list): List of best feasible decisions for each experiment.
        best_FDs_rewards (list): List of rewards for the best feasible decisions for each experiment.
        worst_FDs (list): List of worst feasible decisions for each experiment.
        noise (ndarray, optional): Array of noise vectors for each experiment. Defaults to None.
        scenario (str, optional): Scenario to run the experiments in. Defaults to "best".
        verbose_results (bool, optional): Flag to print verbose results. Defaults to False.
        verbose_game (bool, optional): Flag to print verbose game information. Defaults to False.

    Returns:
        tuple: A tuple containing the algorithm name, average results, and average sequences.
    """
    T = parameters['T']
    B = parameters['B']
    n = parameters['n']
    m = parameters['m']

    results = pd.DataFrame()
    cum_rew_seq = np.empty((num_experiments, parameters["T"]))
    cum_cost_seq = np.empty((num_experiments, parameters["T"]))
    mul_seq = np.empty((num_experiments, parameters["T"]))

    if noise is None:
        noise = np.array([np.zeros(parameters["n"]) for _ in range(num_experiments)])
    
    for i in range(num_experiments):
        np.random.seed(seeds[i])  # Set seed for reproducibility
        print("  Experiment:", i)
        res_row, game = run_experiment(parameters, data_dict[f"experiment_{i}"], best_FDs[i], best_FDs_rewards[i], worst_FDs[i], noise[i], scenario, verbose_game=verbose_game)
        results = pd.concat([results, pd.DataFrame(res_row)], ignore_index=True)
        index_cost = np.argmax(game.cumulative_cost[-1, :])
        cum_rew_seq[i, :] = game.expected_cumulative_reward.copy()
        cum_cost_seq[i, :] = game.cumulative_cost[:, index_cost].copy()
        mul_seq[i, :] = game.vector_of_lambdas[:, index_cost].copy()

        # Reset algorithm
        parameters["RP"].reset()
        parameters["RD"].reset()
        parameters.pop("B_current_A", None)
        parameters.pop("B_current_WC", None)
        parameters.pop("B_aux_WC", None)
        game.reset()

    cum_rew_seq_avg = np.mean(cum_rew_seq, axis=0)
    cum_cost_seq_avg = np.mean(cum_cost_seq, axis=0)
    mul_seq_avg = np.nanmean(mul_seq, axis=0)  # Calculate mean while ignoring None values
    cum_rew_seq_std = np.std(cum_rew_seq, axis=0)
    cum_cost_seq_std = np.std(cum_cost_seq, axis=0)
    mul_seq_std = np.nanstd(mul_seq, axis=0)  # Calculate std while ignoring None values
       
    average_results = get_average_result(results, parameters["T"], parameters["B"], parameters["bandit"]).copy()
    average_sequences = {"avg_rewards":cum_rew_seq_avg, "std_rewards":cum_rew_seq_std, "avg_costs":cum_cost_seq_avg, "std_costs":cum_cost_seq_std, "avg_mult":mul_seq_avg, "std_mult":mul_seq_std}


    if verbose_results:
        show_average_result(average_results)

    return parameters["algorithm_name"], average_results, average_sequences