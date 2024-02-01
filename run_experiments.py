"""
This script runs experiments for the Adversarial Bandit with Knapsacks problem.
It defines the necessary variables, algorithms, and data generation functions.
The main function, run_experiments, generates data, runs the experiments, and returns the results.
"""

import RegretMinimerzModule.RegretMinimizer as R
import DataGeneratorModule.DataGenerator as DG
import AlgorithmsModule.Algorithms as A
import matplotlib.pyplot as plt
import gurobipy as gp
import pandas as pd
import numpy as np
import Game as G
import itertools
import os

from tabulate import tabulate
from datetime import datetime

# Define structural variables
T = 10000  # Number of time steps
n = 5  # Number of actions
m = 1  # Number of knapsacks
B = 5000  # Knapsack capacity # Already at 50000-4000 the algorithm outperforms ours

# Define the algorithm to run
rp_starting_point = np.ones(n)/n
rd_starting_point = np.ones(m)/(2*(m*B/T))
learning_rate_primal = 1/np.sqrt(T)

# Define parameters for the algorithm
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


parameters_augmented_ff = {
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
    "p": 0.5,
    "nu": 0.0,
    "mu": 0.0,
    "mixed_action_predicted": np.array([0.85, 0.05, 0.05, 0.05, 0.0])
}

parameters_augmented_bandit = {
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
    "p": 0.5,
    "nu": 0.0,
    "mu": 0.0,
    "mixed_action_predicted": np.array([0.85, 0.05, 0.05, 0.05, 0.0])
}

parameters_stochastic_ff = {
    "algorithm_name": "stochastic_with_prediction_bandit",
    "algorithm": A.stochastic_with_prediction,
    "state": "prediction",
    "Delta": np.sqrt(T),
    "delta": 1/np.sqrt(T),
    "bandit": False,
    "learning_rate": 1/np.sqrt(T),
    "RP": R.Hedge(starting_point=rp_starting_point, learning_rate=learning_rate_primal, nActions=n),
    "RD": R.DualRegretMinimizer(starting_point=rd_starting_point, learning_rate=1/np.sqrt(T), rho=B/T),
    "T": T,
    "n": n,
    "m": m,
    "t": 0,
    "B": B,
    "rho": B/T,
    "total_rewaerd": 0.0,
    "mixed_action_predicted": np.array([0.85, 0.05, 0.05, 0.05, 0.0]),
    "lambda_value_predicted": np.zeros(m)
}

parameters_stochastic_bandit = {
    "algorithm_name": "stochastic_with_prediction_bandit",
    "algorithm": A.stochastic_with_prediction,
    "state": "prediction",
    "delta": np.sqrt(T),
    "delta": 1/np.sqrt(T),
    "bandit": True,
    "learning_rate": 1/np.sqrt(T),
    "RP": R.Hedge(starting_point=rp_starting_point, learning_rate=learning_rate_primal, nActions=n),
    "RD": R.DualRegretMinimizer(starting_point=rd_starting_point, learning_rate=1/np.sqrt(T), rho=B/T),
    "T": T,
    "n": n,
    "m": m,
    "B": B,
    "rho": B/T,
    "total_rewaerd": 0.0,
    "mixed_action_predicted": np.array([0.85, 0.05, 0.05, 0.05, 0.0]),
    "lambda_value_predicted": np.zeros(m)
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

    model.Params.LogToConsole = 1

    obj = gp.quicksum(gp.quicksum(rewards[:, i] * xi[i]) for i in range(n))
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    model.addConstr(gp.quicksum(xi[i] for i in range(n)) == 1)
    for j in range(m):
        model.addConstr(gp.quicksum(gp.quicksum(costs[:, i, j] * xi[i]) for i in range(n)) <= B)
    model.optimize()
    return np.array([model.getVars()[i].X for i in range(n)]), model.ObjVal, model

def generate_report(game, optimal_reward):
    # Create a dictionary with the desired fields
    report_data = {
        "Total Pseudo Regret": [game.cumulative_pseudo_regret.copy()[-1]],
        "Total Reward": [game.expected_cumulative_reward.copy()[-1]],
        "Cumulative Cost": [game.cumulative_cost.copy()[-1][0]]
    }

    last_iteration = None
    for i, action in enumerate(game.vector_of_actions):
        if action >= 0.0 and action < n:
            last_iteration = i

    report_data["Stopping Iteration"] = [last_iteration]
    report_data["Optimal Reward"] = [optimal_reward]

    # Create a DataFrame from the dictionary
    report_df = pd.DataFrame(report_data)

    # Return the DataFrame
    return report_df

def get_average_result(results, T, B, bandit):
    average_result = results.mean() 
    average_result["Horizon"] = T
    average_result["Budget"] = B
    average_result["Bandit"] = bandit
    return average_result
    
def show_average_result(average_result):
    headers = average_result.index.tolist()
    values = average_result.values.tolist()
    table = zip(headers, values)
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

def generate_data(seeds, number_of_experiments, data_generator, budget, mean_rewards, sigma_rewards, mean_costs, sigma_costs, rate=1000):
    data = dict()
    best_FDs = []
    best_FDs_rewards = []

    for i in range(number_of_experiments):
        np.random.seed(seeds[i])
        # data_generator.generate_data_lognormal(mean_rewards, sigma_rewards, mean_costs, sigma_costs)
        # data_generator.generate_data_lognormal_adversarial(mean_rewards, sigma_rewards, mean_costs, sigma_costs)
        data_generator.generate_data_lognormal_adversarial_v2(mean_rewards, sigma_rewards, mean_costs, sigma_costs, rate=1000)
        rewards = data_generator.data[0].copy()
        costs = data_generator.data[1].copy()
        # np.savetxt(f"experiment_data//rewards_{i}_{T}_{B}.csv", rewards, delimiter=",")
        # TODO: Fix cause being 3-D "savetxt" does not work. np.savetxt(f"experiment_data//costs_{i}_{T}_{B}.csv", costs, delimiter=",")a
        best_FD_results = compute_best_FD(rewards, costs, budget)
        best_FD = best_FD_results[0]
        best_FDs.append(best_FD)
        best_FDs_rewards.append(best_FD_results[1])
        data[f"experiment_{i}"] = (data_generator.data[0].copy(), data_generator.data[1].copy())
        data_generator.reset()
    return data, best_FDs, best_FDs_rewards, best_FD_results

def run_experiments(parameters, data, best_FD, best_FD_reward, best_scenario=False, verbose_game=False):
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
    rewards = data[0]
    costs = data[1]

    if best_scenario:
        parameters["mixed_action_predicted"] = best_FD.copy()
    # print(parameters["mixed_action_predicted"])
    # Run experiment
    game = G.Game(T, B, n, m, bandit=parameters["bandit"])
    game.run(rewards, costs, parameters, best_FD, verbose=verbose_game)

    res_row = generate_report(game, best_FD_reward)
    # print("Results: ", results)

    return res_row, game

def run(seeds, num_experiments, parameters, data_dict, best_FDs, best_FDs_rewards, best_scenario=False, verbose_results=False, verbose_game=False):
    print(parameters["T"])
    results = pd.DataFrame()
    sequences = pd.DataFrame()
    cum_rew_seq = np.empty((num_experiments, parameters["T"]))
    cum_cost_seq = np.empty((num_experiments, parameters["T"]))
    mul_seq = np.empty((num_experiments, parameters["T"]))
    
    for i in range(num_experiments):
        np.random.seed(seeds[i])  # Set seed for reproducibility
        print("Experiment:", i)
        # print("seed:", seeds[i])
        res_row, game = run_experiments(parameters, data_dict[f"experiment_{i}"], best_FDs[i], best_FDs_rewards[i], best_scenario, verbose_game)
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

def compose_results_table(names_list, results_list):
    results_table = pd.DataFrame(results_list)
    results_table.index = names_list
    results_table = results_table.reindex(columns=["Total Reward", "Optimal Reward", 
                                   "Total Pseudo Regret", "Cumulative Cost", 
                                   "Stopping Iteration", "Horizon", "Budget", 
                                   "Bandit"])
    return results_table

def show_results_table(results_table):
    print(tabulate(results_table, results_table.columns.tolist(), tablefmt="fancy_grid"))

# Choose the parameters to run experiments
num_experiments = 1  # Number of experiments
best_FDs = []  # List of best Fixed Distributions. If the data is not generated it will compute automatically
best_FDs_rewards = [] # List of corresponding total rewards of the best Fixed Distributions.
best_scenario = True  # Whether to use the best Fixed Distribution as prediction or to use the fed prediction.
verbose_results = False  # prints the values of the results in a table for each single experiment.
verbose_game = True  # gives a complete understanding of each iteration of the game.

seeds = np.random.randint(0, 10000, size=num_experiments)
seeds2 = [10]#, 20 , 30] #, 40, 50, 60, 70, 80, 90, 100] #, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 211, 220, 230, 240, 250, 260, 270, 280, 290, 300]

# Generate the data
data_generator = DG.DataGenerator(n, m, T, B)
data, best_FDs, best_FDs_rewards, a = generate_data(seeds2, num_experiments, data_generator, B, mean_rewards, sigma_rewards, mean_costs, sigma_costs)

# Run the experiments 
# name1, res1, _ = run(seeds2, num_experiments, parameters_pd_ff, data, best_FDs, best_FDs_rewards, best_scenario, verbose_results=verbose_results, verbose_game=verbose_game)
# name2, res2, _ = run(seeds2, num_experiments, parameters_pd_bandit, data, best_FDs, best_FDs_rewards, best_scenario, verbose_results=verbose_results, verbose_game=verbose_game)
# name3, res3, seq = run(seeds2, num_experiments, parameters_augmented_ff, data, best_FDs, best_FDs_rewards, best_scenario, verbose_results=verbose_results, verbose_game=verbose_game)
name4, res4, _ = run(seeds2, num_experiments, parameters_augmented_bandit, data, best_FDs, best_FDs_rewards, best_scenario, verbose_results=verbose_results, verbose_game=verbose_game)

# Define which experiments to show
names_list = [name4]#, name2, name3, name4]
results_list = [res4]#, res2, res3, res4]

# Compute the aggregated table of results
results_table = compose_results_table(names_list, results_list)

# Show the results of the experiments
show_results_table(results_table)

# Make the plots
run_build = {"seeds": seeds2, "num_experiments": num_experiments, "data": data, "best_FDs": best_FDs, "best_FDs_rewards": best_FDs_rewards, "best_scenario": best_scenario, "verbose_results": verbose_results, "verbose_game": verbose_game}


# plot_results(results_list, names_list, T, B, num_experiments, bandit=True)
def plot_across_p(exp_name, run_build, parameters_test, parameters_benchmark):
    reward_dict = dict()
    parameters_test_aux = parameters_test.copy()
    for p in np.arange(0.0, 1.1, 0.1):
        # Update the value of p in the parameters dictionary
        print("Starting experiments for p:", p)
        parameters_test_aux["p"] = p
        _, res, _ = run(run_build["seeds"], run_build["num_experiments"], parameters_test_aux, run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["best_scenario"], run_build["verbose_results"], run_build["verbose_game"])
        
        reward_dict[p] = res["Total Reward"]

    p_values = list(reward_dict.keys())
    total_rewards = list(reward_dict.values())

    plt.bar(p_values, total_rewards, width=0.08)
    plt.xlabel("p")
    plt.ylabel("Total Reward")
    plt.ylim(top=plt.ylim()[1] + 0.15 * plt.ylim()[1])  # Add 15% more space on top of the y-axis
    plt.title("Total Reward vs. p")

    min_p, max_p, step_p = 0.0, 1.0, 0.1  # Define the range and step of p values for ticks
    plt.xticks(np.arange(min_p, max_p + step_p, step_p))

    plt.xlim(min_p - step_p/2, max_p + step_p/2) 

    optimal_reward = res["Optimal Reward"]  # Get the optimal reward from the results
    reward_dict["Optimal"] = optimal_reward  # Add the optimal reward to the dictionary
    plt.axhline(y=optimal_reward, color='b', linestyle='--', label='Optimal Reward')  # Add a horizontal line for the optimal reward
    
    print("Running benchmark experiments")
    _, res, _ = run(run_build["seeds"], run_build["num_experiments"], parameters_benchmark, run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["best_scenario"], run_build["verbose_results"], run_build["verbose_game"])
    benchmark_reward = res["Total Reward"]
    reward_dict["Benchmark"] = benchmark_reward
    plt.axhline(y=benchmark_reward, color='r', linestyle='--', label='Benchmark Reward')

    plt.legend()  # Show the legend

    now = datetime.now()
    fname = f"c:/Users/david/Desktop/AdversarialBanditwithKnapsacks_code/AdversarialKnapsacksCode/imgs/{exp_name}/comparision_" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S") + "reward.pdf"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname + "mult.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    plt.show()

    return reward_dict

def plot_experiments(exp_name, run_build, parameters_dict):

    algorithms = parameters_dict.keys()

    avg_rewards = {k: [] for k in algorithms}
    std_rewards = {k: [] for k in algorithms}
    avg_costs = {k: [] for k in algorithms}
    std_costs = {k: [] for k in algorithms}
    avg_mult = {k: [] for k in algorithms}
    std_mult = {k: [] for k in algorithms}

    for k in algorithms:
        _, res, avg_sequences =   run(run_build["seeds"], run_build["num_experiments"], parameters_dict[k], run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["best_scenario"], run_build["verbose_results"], run_build["verbose_game"])
        avg_rewards[k] = avg_sequences["avg_rewards"]
        std_rewards[k] = avg_sequences["std_rewards"]
        avg_costs[k] = avg_sequences["avg_costs"]
        std_costs[k] = avg_sequences["std_costs"]
        avg_mult[k] = avg_sequences["avg_mult"]
        std_mult[k] = avg_sequences["std_mult"]

    now = datetime.now()
    fname = f"c:/Users/david/Desktop/AdversarialBanditwithKnapsacks_code/AdversarialKnapsacksCode/imgs/{exp_name}/comparision_" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S") + "reward.pdf"
    x = np.arange(0, T, 1)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    linestyles = itertools.cycle(['-', '--', '-.', ':'])
    algstyles = {k: next(linestyles) for k in algorithms}

    f_final = plt.figure(figsize=(6, 4))  
    ax_final = f_final.add_subplot(111)  

    for k in algorithms:
        rewards_avgs = avg_rewards[k]
        rewards_stds = std_rewards[k]

        ax_final.fill_between(
                x,
                (rewards_avgs - rewards_stds),
                (rewards_avgs + rewards_stds),
                alpha=0.5,
                )
        ax_final.plot(rewards_avgs, label=k, linestyle=algstyles[k])

    ax_final.axhline(y=res["Optimal Reward"], color='r', linestyle='-', label='Optimal Reward')  # Add a horizontal line for the optimal reward

    ax_final.legend(fontsize=11)
    ax_final.set_xlabel(r"$Time$", fontsize=11)
    ax_final.set_ylabel("Cumulative Reward", fontsize=11)
    
    f_final.suptitle(f'Comparision for B={B}', fontsize=11)

    plt.savefig(fname + "reward.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    plt.show()

    f_final = plt.figure(figsize=(6, 4))  
    ax_final = f_final.add_subplot(111)  
    for k in algorithms:
        cost_avgs = avg_costs[k]
        cost_stds = std_costs[k]

        ax_final.fill_between(
                x,
                (cost_avgs - cost_stds),
                (cost_avgs + cost_stds),
                alpha=0.5,
                )
        ax_final.plot(cost_avgs, label=k, linestyle=algstyles[k])

    ax_final.legend(fontsize=11)
    ax_final.set_xlabel(r"$Time$", fontsize=11)
    ax_final.set_ylabel("Cumulative Costs", fontsize=11)

    f_final.suptitle(f'Comparision for B={B}', fontsize=11)

    plt.savefig(fname + "costs.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    plt.show()

    f_final = plt.figure(figsize=(6, 4))  
    ax_final = f_final.add_subplot(111)  
    for k in algorithms:
        if len(avg_mult[k]) > 0:
            ax_final.fill_between(
                x,
                (avg_mult[k] - std_mult[k]),
                (avg_mult[k] + std_mult[k]),
                alpha=0.5,
                )
            ax_final.plot(avg_mult[k], label=k, linestyle=algstyles[k])

    ax_final.legend(fontsize=11)
    ax_final.set_xlabel(r"$Time$", fontsize=11)
    ax_final.set_ylabel("Dual multiplier", fontsize=11)

    f_final.suptitle(f'Comparision for B={B}', fontsize=11)

    plt.savefig(fname + "mult.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    plt.show()

parameters_to_run_ff = {"primal_dual_ff": parameters_pd_ff, "adversarial_with_prediction_ff": parameters_augmented_ff}         
parameters_to_run_bandit = {"primal_dual_bandit": parameters_pd_bandit,  "adversarial_with_prediction_bandit": parameters_augmented_bandit}

# plot_experiments("ff05", run_build, parameters_to_run_ff)
# plot_experiments("bandit05", run_build, parameters_to_run_bandit)
# plot_experiments("ff01", run_build, parameters_to_run_ff)
# plot_experiments("bandit01", run_build, parameters_to_run_bandit)
# plot_across_p("ff_across_p", run_build, parameters_augmented_ff, parameters_pd_ff)
# plot_across_p("bandit_across_p", run_build, parameters_augmented_bandit, parameters_pd_bandit)
