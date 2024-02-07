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
import pickle
import os

from tabulate import tabulate
from datetime import datetime

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

    model.Params.LogToConsole = 0

    obj = gp.quicksum(gp.quicksum(rewards[:, i] * xi[i]) for i in range(n))
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    model.addConstr(gp.quicksum(xi[i] for i in range(n)) == 1)
    for j in range(m):
        model.addConstr(gp.quicksum(gp.quicksum(costs[:, i, j] * xi[i]) for i in range(n)) <= B)
    model.optimize()
    return np.array([model.getVars()[i].X for i in range(n)]), model.ObjVal

def compute_worst_FD(rewards, costs, OPT_FD):
    n = rewards.shape[1]
    T = rewards.shape[0]
    m = costs.shape[2]

    # Define the model and its variables
    model = gp.Model("lp")
    xi = model.addVars(n, lb=0, ub=1, name="xi")
    model.update()

    model.Params.LogToConsole = 0

    obj = gp.quicksum(gp.quicksum(rewards[:, i] * xi[i]) for i in range(n))
    # obj = gp.quicksum(gp.quicksum(gp.quicksum(costs[:, i, j] * xi[i]) for i in range(n)) for j in range(m))
    
    model.setObjective(obj, gp.GRB.MINIMIZE)
    # model.setObjective(obj, gp.GRB.MAXIMIZE) 
    model.addConstr(gp.quicksum(xi[i] for i in range(n)) == 1)
    # model.addConstr(gp.quicksum(gp.quicksum(rewards[:, i] * xi[i]) for i in range(n)) <= OPT_FD)
    for j in range(m):    
        model.addConstr(gp.quicksum(gp.quicksum(costs[:, i, j] * xi[i]) for i in range(n)) >= 0.95*B)
        model.addConstr(gp.quicksum(gp.quicksum(costs[:, i, j] * xi[i]) for i in range(n)) <= B)
    model.optimize()
    return np.array([model.getVars()[i].X for i in range(n)]), model.ObjVal

def generate_report(game, optimal_reward, tvd):
    # Create a dictionary with the desired fields
    report_data = {
        #"Total Pseudo Regret": [game.cumulative_pseudo_regret.copy()[-1]],
        "Total Reward": [game.expected_cumulative_reward.copy()[-1]],
        "Cumulative Cost": [game.cumulative_cost.copy()[-1][0]]
    }

    last_iteration = None
    for i, action in enumerate(game.vector_of_actions):
        if action >= 0.0 and action < n:
            last_iteration = i

    report_data["Stopping Iteration"] = [last_iteration]
    report_data["Optimal Reward"] = [optimal_reward]
    report_data["TVD"] = [tvd]

    # Create a DataFrame from the dictionary
    report_df = pd.DataFrame(report_data)

    # Return the DataFrame
    return report_df

def get_average_result(results, T, B, bandit):
    average_result = results.mean() 
    std_result = results.std()
    average_result["Standard Deviation"] = std_result["Total Reward"]
    average_result["Horizon"] = T
    average_result["Budget"] = B
    average_result["Bandit"] = bandit
    
    return average_result
    
def show_average_result(average_result):
    headers = average_result.index.tolist()
    values = average_result.values.tolist()
    table = zip(headers, values)
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

def generate_data(seeds, number_of_experiments, data_generator, budget, mean_rewards, sigma_rewards, mean_costs, sigma_costs, rate=1000, store=True):
    data = dict()
    best_FDs = []
    best_FDs_rewards = []
    worst_FDs = []

    for i in range(number_of_experiments):
        np.random.seed(seeds[i])
        data_generator.generate_data_lognormal(mean_rewards, sigma_rewards, mean_costs, sigma_costs)
        # data_generator.generate_data_lognormal_adversarial(mean_rewards, sigma_rewards, mean_costs, sigma_costs)
        # data_generator.generate_data_lognormal_adversarial_v2(mean_rewards, sigma_rewards, mean_costs, sigma_costs, rate=1000)
        rewards = data_generator.data[0].copy()
        costs = data_generator.data[1].copy()
        # np.savetxt(f"experiment_data//rewards_{i}_{T}_{B}.csv", rewards, delimiter=",")
        # TODO: Fix cause being 3-D "savetxt" does not work. np.savetxt(f"experiment_data//costs_{i}_{T}_{B}.csv", costs, delimiter=",")a
        best_FD_results = compute_best_FD(rewards, costs, budget)
        best_FD = best_FD_results[0]
        best_FDs.append(best_FD)
        best_FDs_rewards.append(best_FD_results[1])
        data[f"experiment_{i}"] = (data_generator.data[0].copy(), data_generator.data[1].copy())
        
        worst_FD_result = compute_worst_FD(rewards, costs, best_FD_results[1])
        worst_FD = worst_FD_result[0]
        worst_FDs.append(worst_FD)

        data_generator.reset()

    if store:
        with open('data.pkl', 'wb') as file:
            with open('data.pkl', 'wb') as file:
                file.truncate(0)
                pickle.dump((data, best_FDs, best_FDs_rewards, worst_FDs), file)
        
    return data, best_FDs, best_FDs_rewards, worst_FDs

def run_experiments(parameters, data, best_FD, best_FD_reward, worst_FD, noise_vector, scenario, verbose_game=False):
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
    results = pd.DataFrame()
    cum_rew_seq = np.empty((num_experiments, parameters["T"]))
    cum_cost_seq = np.empty((num_experiments, parameters["T"]))
    mul_seq = np.empty((num_experiments, parameters["T"]))
    tvds = np.empty(num_experiments)

    if noise is None:
        noise = np.array([np.zeros(parameters["n"]) for _ in range(num_experiments)])
    
    for i in range(num_experiments):
        np.random.seed(seeds[i])  # Set seed for reproducibility
        print("Experiment:", i)
        # print("seed:", seeds[i])
        res_row, game = run_experiments(parameters, data_dict[f"experiment_{i}"], best_FDs[i], best_FDs_rewards[i], worst_FDs[i], noise[i], scenario, verbose_game=verbose_game)
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

def compose_results_table(names_list, results_list):
    results_table = pd.DataFrame(results_list)
    results_table.index = names_list
    results_table = results_table.reindex(columns=["Total Reward", "Optimal Reward", 
                                                   "TVD", "Cumulative Cost", 
                                                   "Stopping Iteration", "Standard Deviation", 
                                                   "Horizon", "Budget", "Bandit"])
    return results_table

def show_results_table(results_table):
    print(tabulate(results_table, results_table.columns.tolist(), tablefmt="fancy_grid"))
    return

def run_experiment_table(run_build, parameters_dict):
    name_list = []
    res_list = []
    for parameters_key in parameters_dict.keys():
        parameters_test_aux = parameters_dict[parameters_key].copy()
        name, res, _ = run(run_build["seeds"], run_build["num_experiments"], parameters_test_aux, run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["worst_FDs"], run_build["noise"], run_build["scenario"], run_build["verbose_results"], run_build["verbose_game"])
        name_list.append(name)
        res_list.append(res)
    results_table = compose_results_table(name_list, res_list)
    show_results_table(results_table)
    return results_table

    # name1, res1, seq = run(seeds, num_experiments, parameters_pd_ff, data, best_FDs, best_FDs_rewards, worst_FDs, noise, scenario=scenario, verbose_results=verbose_results, verbose_game=verbose_game)
    # name2, res2, seq = run(seeds, num_experiments, parameters_pd_bandit, data, best_FDs, best_FDs_rewards, worst_FDs, noise, scenario=scenario, verbose_results=verbose_results, verbose_game=verbose_game)
    # name3, res3, seq = run(seeds, num_experiments, parameters_augmented_ff, data, best_FDs, best_FDs_rewards, worst_FDs, noise, scenario=scenario, verbose_results=verbose_results, verbose_game=verbose_game)
    # name4, res4, seq = run(seeds, num_experiments, parameters_augmented_bandit, data, best_FDs, best_FDs_rewards, worst_FDs, noise, scenario=scenario, verbose_results=verbose_results, verbose_game=verbose_game)
    # parameters_augmented_bandit["p"] = 0.5
    # parameters_augmented_ff["p"] = 0.5
    # noise = np.array([np.random.normal(0, 0.01, n) for _ in range(num_experiments)])
    # name5, res5, seq = run(seeds, num_experiments, parameters_augmented_ff, data, best_FDs, best_FDs_rewards, worst_FDs, noise, scenario=scenario, verbose_results=verbose_results, verbose_game=verbose_game)
    # name6, res6, seq = run(seeds, num_experiments, parameters_augmented_bandit, data, best_FDs, best_FDs_rewards, worst_FDs, noise, scenario=scenario, verbose_results=verbose_results, verbose_game=verbose_game)

    # noise = np.array([np.random.normal(0, 0.1, n) for _ in range(num_experiments)])
    # name7, res7, seq = run(seeds, num_experiments, parameters_augmented_ff, data, best_FDs, best_FDs_rewards, worst_FDs, noise, scenario=scenario, verbose_results=verbose_results, verbose_game=verbose_game)
    # name8, res8, seq = run(seeds, num_experiments, parameters_augmented_bandit, data, best_FDs, best_FDs_rewards, worst_FDs, noise, scenario=scenario, verbose_results=verbose_results, verbose_game=verbose_game)


    # Define which experiments to show
    # names_list = [name3, name4, name5, name6, name7, name8]#, name1, name2, name3, name4]
    # results_list = [res3, res4, res5, res6, res7, res8]#, res1, res2, res3, res4]

    # Compute the aggregated table of results
    # results_table = compose_results_table(names_list, results_list)

# Show the results of the experiments
# show_results_table(results_table)

# plot_results(results_list, names_list, T, B, num_experiments, bandit=True)
def plot_across_p(exp_name, run_build, parameters_test, parameters_benchmark):
    reward_dict = dict()
    parameters_test_aux = parameters_test.copy()
    for p in np.arange(0.0, 1.1, 0.1):
        # Update the value of p in the parameters dictionary
        print("Starting experiments for p:", p)
        parameters_test_aux["p"] = p
        _, res, _ = run(run_build["seeds"], run_build["num_experiments"], parameters_test_aux, run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["worst_FDs"], run_build["noise"], run_build["scenario"], run_build["verbose_results"], run_build["verbose_game"])
        
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
    _, res, _ = run(run_build["seeds"], run_build["num_experiments"], parameters_benchmark, run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["worst_FDs"], run_build["noise"], run_build["scenario"], run_build["verbose_results"], run_build["verbose_game"])
    benchmark_reward = res["Total Reward"]
    reward_dict["Benchmark"] = benchmark_reward
    plt.axhline(y=benchmark_reward, color='r', linestyle='--', label='Primal-Dual Reward')

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
        _, res, avg_sequences =   run(run_build["seeds"], run_build["num_experiments"], parameters_dict[k], run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["worst_FDs"], run_build["noise"], run_build["scenario"], run_build["verbose_results"], run_build["verbose_game"])
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

    ax_final.legend(fontsize=20, loc='upper left')
    ax_final.set_xlabel(r"$Time$", fontsize=20)
    ax_final.set_ylabel("Cumulative Reward", fontsize=20)
    
    f_final.suptitle(f'Comparision for B={B}', fontsize=20)

    plt.savefig(fname + "reward.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    plt.show()

    return

    # f_final = plt.figure(figsize=(6, 4))  
    # ax_final = f_final.add_subplot(111)  
    # for k in algorithms:
    #     cost_avgs = avg_costs[k]
    #     cost_stds = std_costs[k]

    #     ax_final.fill_between(
    #             x,
    #             (cost_avgs - cost_stds),
    #             (cost_avgs + cost_stds),
    #             alpha=0.5,
    #             )
    #     ax_final.plot(cost_avgs, label=k, linestyle=algstyles[k])

    # ax_final.legend(fontsize=20)
    # ax_final.set_xlabel(r"$Time$", fontsize=20)
    # ax_final.set_ylabel("Cumulative Costs", fontsize=20)

    # f_final.suptitle(f'Comparision for B={B}', fontsize=20)

    # plt.savefig(fname + "costs.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    # plt.show()

    # f_final = plt.figure(figsize=(6, 4))  
    # ax_final = f_final.add_subplot(111)  
    # for k in algorithms:
    #     if len(avg_mult[k]) > 0:
    #         ax_final.fill_between(
    #             x,
    #             (avg_mult[k] - std_mult[k]),
    #             (avg_mult[k] + std_mult[k]),
    #             alpha=0.5,
    #             )
    #         ax_final.plot(avg_mult[k], label=k, linestyle=algstyles[k])

    # ax_final.legend(fontsize=20)
    # ax_final.set_xlabel(r"$Time$", fontsize=20)
    # ax_final.set_ylabel("Dual multiplier", fontsize=20)

    # f_final.suptitle(f'Comparision for B={B}', fontsize=20)

    # plt.savefig(fname + "mult.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    # plt.show()

def plot_across_noise(exp_name, run_build, parameters_test, parameters_benchmark, min_noise=0.0, max_noise=0.1, step_noise=0.01):
    reward_dict = dict()
    total_rewards = []
    # Iterate over noise levels
    sd_noise_levels = np.arange(min_noise, max_noise + step_noise, step_noise)
    for noise_level in sd_noise_levels:
        # Set the noise level
        print("Starting experiments for noise level:", noise_level)
        # signs = np.array([[1 if run_build["best_FDs"][i][j] < 0.05 else -1 for j in range(n)] for i in range(num_experiments)])
        run_build["noise"] = np.array([np.abs(np.random.normal(0, noise_level, n)) for _ in range(num_experiments)]) # * signs
        # run_build["noise"] = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        _, res, _ = run(run_build["seeds"], run_build["num_experiments"], parameters_test, run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["worst_FDs"], run_build["noise"], run_build["scenario"], run_build["verbose_results"], run_build["verbose_game"])
        
        reward_dict[str(noise_level)[0:4]] = res["Total Reward"]
        reward_dict[str(noise_level)[0:4] + "_std"] = res["Standard Deviation"]
        total_rewards.append(res["Total Reward"])
    noise_values = sd_noise_levels

    plt.bar(noise_values, total_rewards, width=step_noise/2)
    plt.xlabel("gaussian noise std", fontsize=20)
    plt.ylabel("Total Reward", fontsize=20)
    plt.ylim(top=plt.ylim()[1] + 0.15 * plt.ylim()[1])  # Add 15% more space on top of the y-axis
    plt.title("Total Reward vs. noise level", fontsize=20)

    plt.xticks(np.arange(min_noise, max_noise + step_noise, step_noise)[::2], fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim(min_noise - step_noise/2, max_noise + step_noise/2) 

    optimal_reward = res["Optimal Reward"]  # Get the optimal reward from the results
    reward_dict["Optimal"] = optimal_reward  # Add the optimal reward to the dictionary
    plt.axhline(y=optimal_reward, color='b', linestyle='--', label='Optimal Reward')  # Add a horizontal line for the optimal reward
    
    print("Running benchmark experiments")
    _, res, _ = run(run_build["seeds"], run_build["num_experiments"], parameters_benchmark, run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["worst_FDs"], run_build["noise"], run_build["scenario"], run_build["verbose_results"], run_build["verbose_game"])
    benchmark_reward = res["Total Reward"]
    reward_dict["Benchmark"] = benchmark_reward
    plt.axhline(y=benchmark_reward, color='r', linestyle='--', label='Primal-Dual Reward')

    plt.legend(fontsize=20, loc='lower right')  # Show the legend

    now = datetime.now()
    fname = f"c:/Users/david/Desktop/AdversarialBanditwithKnapsacks_code/AdversarialKnapsacksCode/imgs/{exp_name}/comparision_" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S") + "reward.pdf"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    
    # Adjust y-axis limits to include the optimal reward
    plt.ylim(bottom=0, top=max(optimal_reward, benchmark_reward) + 0.15 * plt.ylim()[1])
    
    plt.savefig(fname + "mult.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    plt.show()

    return reward_dict

# Define structural variables
T = 50000  # Number of time steps
n = 5  # Number of actions
m = 1  # Number of knapsacks
B = 5000 # Long term budget

# Define the algorithm to run
rp_starting_point = np.ones(n)/n
rd_starting_point = np.ones(m)/(2*(m*B/T))
learning_rate_primal = 1/np.sqrt(T)

# Define parameters for the algorithms
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
    "p": 0.9,
    "mu": 0.0,
    "mixed_action_predicted": np.array([0.85, 0.05, 0.05, 0.05, 0.0])
}
# (2*np.sqrt(2*np.log(1/0.1)))/(0.1*np.sqrt(T))

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
    "p": 0.9,
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

parameters_augmented_ff09 = parameters_augmented_ff.copy()
parameters_augmented_ff09["algorithm_name"] = "adversarial_with_prediction_ff09"
parameters_augmented_bandit09 = parameters_augmented_bandit.copy()
parameters_augmented_bandit09["algorithm_name"] = "adversarial_with_prediction_bandit09"
parameters_augmented_ff05 = parameters_augmented_ff.copy()
parameters_augmented_ff05["p"] = 0.5
parameters_augmented_ff05["algorithm_name"] = "adversarial_with_prediction_ff05"
parameters_augmented_bandit05 = parameters_augmented_bandit.copy()
parameters_augmented_bandit05["p"] = 0.5
parameters_augmented_bandit05["algorithm_name"] = "adversarial_with_prediction_bandit05"
parameters_augmented_ff09_mulow = parameters_augmented_ff.copy()
parameters_augmented_ff09_mulow["mu"] = 1/np.sqrt(T)
parameters_augmented_ff09_mulow["algorithm_name"] = "adversarial_with_prediction_ff09_mulow"
parameters_augmented_bandit09_mulow = parameters_augmented_bandit.copy()
parameters_augmented_bandit09_mulow["mu"] = 1/np.sqrt(T)
parameters_augmented_bandit09_mulow["algorithm_name"] = "adversarial_with_prediction_bandit09_mulow"
parameters_augmented_ff09_muhigh = parameters_augmented_ff.copy()
parameters_augmented_ff09_muhigh["mu"] = (2*np.sqrt(2*np.log(1/0.1)))/(0.1*np.sqrt(T))
parameters_augmented_ff09_muhigh["algorithm_name"] = "adversarial_with_prediction_ff09_muhigh"
parameters_augmented_bandit09_muhigh = parameters_augmented_bandit.copy()
parameters_augmented_bandit09_muhigh["mu"] = (2*np.sqrt(2*np.log(1/0.1)))/(0.1*np.sqrt(T))
parameters_augmented_bandit09_muhigh["algorithm_name"] = "adversarial_with_prediction_bandit09_muhigh"

# Define action means
mean_rewards = [0.9, 1.2, 0.1, 0.4]
mean_costs = [0.8, 0.6, 1.2, 0.5]
sigma_rewards = [0.5, 0.5, 0.5, 0.5]
sigma_costs = [0.5, 0.5, 0.5, 0.5]
# np.exp(np.array([0.8, 0.9, 0.3, 0.5]) + (np.array([0.5, 0.5, 0.5, 0.5])**2)/2)
# np.mean(data["experiment_0"][0], axis=0)

# Choose the parameters to run experiments
num_experiments = 10  # Number of experiments
best_FDs = []  # List of best Fixed Distributions. If the data is not generated it will compute automatically
best_FDs_rewards = [] # List of corresponding total rewards of the best Fixed Distributions.
worst_FDs = []  # List of worst Fixed Distributions. If the data is not generated it will compute automatically
scenario = "best"  # Whether to use the best Fixed Distribution as prediction or to use the fed prediction.
verbose_results = False  # prints the values of the results in a table for each single experiment.
verbose_game = False  # gives a complete understanding of each iteration of the game.
store = "Done"  # Whether to store the data or not. If the data is already stored it will load it.

seeds = [20, 10 , 30, 40, 51, 60, 70, 80, 90, 100] #, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 211, 220, 230, 240, 250, 260, 270, 280, 290, 300]

# Generate the data
if store == "Done":
    with open('data.pkl', 'rb') as file:
        data, best_FDs, best_FDs_rewards, worst_FDs = pickle.load(file)
else:
    data_generator = DG.DataGenerator(n, m, T, B)
    data, best_FDs, best_FDs_rewards, worst_FDs = generate_data(seeds, num_experiments, data_generator, B, mean_rewards, sigma_rewards, mean_costs, sigma_costs, store=store)


# Define the noise to use in the experiments
noise = None
# noise = np.array([np.random.normal(0, 0.01, n) for _ in range(num_experiments)])

################### MAKE THE PLOTS ###################
# Define the parameters to run the plots
run_build = {"seeds": seeds, "num_experiments": num_experiments, "data": data, "best_FDs": best_FDs, "best_FDs_rewards": best_FDs_rewards, "worst_FDs": worst_FDs, "noise": noise, "scenario": scenario, "verbose_results": verbose_results, "verbose_game": verbose_game}
parameters_to_run_table = {# "primal_dual_ff": parameters_pd_ff.copy(), 
                           # "primal_dual_bandit": parameters_pd_bandit.copy(), 
                           # "adversarial_with_prediction_ff05": parameters_augmented_ff05.copy(), 
                           # "adversarial_with_prediction_bandit05": parameters_augmented_bandit05.copy(),
                           # "adversarial_with_prediction_ff09": parameters_augmented_ff09.copy(), 
                           # "adversarial_with_prediction_bandit09": parameters_augmented_bandit09.copy(),
                           # "adversarial_with_prediction_ff09_mulow": parameters_augmented_ff09_mulow.copy(), 
                           # "adversarial_with_prediction_bandit09_mulow": parameters_augmented_bandit09_mulow.copy(),
                           "adversarial_with_prediction_ff09_muhigh": parameters_augmented_ff09_muhigh.copy(), 
                           "adversarial_with_prediction_bandit09_muhigh": parameters_augmented_bandit09_muhigh.copy(),
                           }
parameters_to_run_across_noise_ff = {"benchmark": parameters_pd_ff.copy(), "test": parameters_augmented_ff.copy()}         
parameters_to_run_across_noise_bandit = {"benchmark": parameters_pd_bandit.copy(),  "test": parameters_augmented_bandit.copy()}
parameters_to_run_across_noise_ffmulow = {"benchmark": parameters_pd_ff.copy(), "test": parameters_augmented_ff09_mulow.copy()}
parameters_to_run_across_noise_banditmulow = {"benchmark": parameters_pd_bandit.copy(),  "test": parameters_augmented_bandit09_mulow.copy()}
parameters_to_run_across_noise_ffmuhigh = {"benchmark": parameters_pd_ff.copy(), "test": parameters_augmented_ff09_muhigh.copy()}
parameters_to_run_across_noise_banditmuhigh = {"benchmark": parameters_pd_bandit.copy(),  "test": parameters_augmented_bandit09_muhigh.copy()}
all_params_noise_experiments = {# "ff_across_noise": parameters_to_run_across_noise_ff, 
                                # "bandit_across_noise": parameters_to_run_across_noise_bandit, 
                                # "ff_across_noise_muhigh": parameters_to_run_across_noise_ffmuhigh,
                                # "bandit_across_noise_muhigh": parameters_to_run_across_noise_banditmuhigh,
                                "ff_across_noise_mulow": parameters_to_run_across_noise_ffmulow,
                                "bandit_across_noise_mulow": parameters_to_run_across_noise_banditmulow,
                                }
######### PLOTS TO SHOW IN THE MAIN TEXT #########

# Plot the table of results
print("Running experiments for the table")
results_table = run_experiment_table(run_build, parameters_to_run_table)

now = datetime.now ()
file_name = f"c:/Users/david/Desktop/AdversarialBanditwithKnapsacks_code/AdversarialKnapsacksCode/results/table/" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S")
os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
results_table.to_csv(file_name + "table.csv")
# results_table = pd.read_csv("c:/Users/david/Desktop/AdversarialBanditwithKnapsacks_code/AdversarialKnapsacksCode/results/table/" + "07022024_124843" + "table.csv", index_col=0)

# Plot the results across noise
aux = [[str(x)[0:4], str(x)[0:4] + "_std"] for x in np.arange(0.0, 0.11, 0.01)]
aux_bad_case = [[str(x)[0:4], str(x)[0:4] + "_std"] for x in np.arange(0.5, 2.15, 0.15)]
columns_noise = ["Name"] + [item for sublist in aux for item in sublist] + ["Optimal", "Benchmark"]
columns__noise_bad_case = ["Name"] + [item for sublist in aux_bad_case for item in sublist] + ["Optimal", "Benchmark"]
results_noise_dataframe = pd.DataFrame(columns=columns_noise)
results_noise_bad_case_dataframe = pd.DataFrame(columns=columns__noise_bad_case)
for parameters_across_noise_key in all_params_noise_experiments.keys():
    name = parameters_across_noise_key
    print(f"Running experiments for {name}")
    params = all_params_noise_experiments[name]
    experiment_row = plot_across_noise(name, run_build.copy(), 
                     params["test"], params["benchmark"],
                     min_noise=0.0, max_noise=0.1, step_noise=0.01
                     )
    experiment_row_bad_case = plot_across_noise(name + "_bad_case", run_build.copy(),
                     params["test"], params["benchmark"],
                     min_noise=0.5, max_noise=2.0, step_noise=0.15
                     )
    row = {"Name": parameters_across_noise_key}
    row.update(experiment_row)
    row = pd.Series(row).to_frame().T.set_index("Name")
    row_bad = {"Name": parameters_across_noise_key}
    row_bad.update(experiment_row_bad_case)
    row_bad = pd.Series(row_bad).to_frame().T.set_index("Name")
    results_noise_dataframe = pd.concat([results_noise_dataframe, row])
    results_noise_bad_case_dataframe = pd.concat([results_noise_bad_case_dataframe, row_bad])

now = datetime.now()

file_name = f"c:/Users/david/Desktop/AdversarialBanditwithKnapsacks_code/AdversarialKnapsacksCode/results/noise/" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S")
os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
results_noise_dataframe.to_csv(file_name + "reward.csv")
results_noise_bad_case_dataframe.to_csv(file_name + "reward_bad_case.csv")
# plot_across_noise("ff_across_noise", run_build, parameters_augmented_ff, parameters_pd_ff, min_noise=0.0, max_noise=0.1, step_noise=0.01)
# plot_across_noise("bandit_across_noise", run_build, parameters_augmented_bandit, parameters_pd_bandit, min_noise=0.0, max_noise=0.1, step_noise=0.01)
# plot_across_noise("ff_across_noise_muhigh", run_build, parameters_augmented_ff09_muhigh, parameters_pd_ff, min_noise=0.0, max_noise=0.1, step_noise=0.01)
# plot_across_noise("bandit_across_noise_muhigh", run_build, parameters_augmented_bandit09_muhigh, parameters_pd_bandit, min_noise=0.0, max_noise=0.1, step_noise=0.01)

# Plot the results across noise for large noise (bad case)
# plot_across_noise("ff_across_noise_bad_case", run_build, parameters_augmented_ff, parameters_pd_ff, min_noise=0.5, max_noise=2.0, step_noise=0.15)
# plot_across_noise("bandit_across_noise_bad_case", run_build, parameters_augmented_bandit, parameters_pd_bandit, min_noise=0.5, max_noise=2.0, step_noise=0.15)
# plot_across_noise("ff_across_noise_muhigh", run_build, parameters_augmented_ff09_muhigh, parameters_pd_ff, min_noise=0.5, max_noise=2.0, step_noise=0.15)
# plot_across_noise("bandit_across_noise_muhigh", run_build, parameters_augmented_bandit09_muhigh, parameters_pd_bandit, min_noise=0.5, max_noise=2.0, step_noise=0.15)


# plot_experiments("ff01_09", run_build, parameters_to_run_ff)
# plot_experiments("bandit01_09", run_build, parameters_to_run_bandit)
# parameters_augmented_bandit["p"] = 0.5
# parameters_augmented_ff["p"] = 0.5
# plot_experiments("ff01_05", run_build, parameters_to_run_ff)
# plot_experiments("bandit01_05", run_build, parameters_to_run_bandit)
# plot_across_p("ff_across_p", run_build, parameters_augmented_ff, parameters_pd_ff)
# plot_across_p("bandit_across_p", run_build, parameters_augmented_bandit, parameters_pd_bandit)





# Utility for Interactive plots

# noise_values = list(reward_dict.keys())[:-2]
# total_rewards = list(reward_dict.values())[:-2]
# plt.bar(noise_values, total_rewards, width=0.005)
# plt.xlabel("gaussian noise std", fontsize=20)
# plt.ylabel("Total Reward", fontsize=20)
# plt.ylim(top=plt.ylim()[1] + 1.2 * plt.ylim()[1])  # Add 15% more space on top of the y-axis
# plt.title("Total Reward vs. noise level", fontsize=20)

# min_noise, max_noise, step_noise = 0.0, 0.1, 0.01  # Define the range and step of p values for ticks
# plt.xticks(np.arange(min_noise, max_noise + step_noise, step_noise)[::2], fontsize=20)
# plt.yticks(fontsize=20)

# plt.xlim(min_noise - step_noise/2, max_noise + step_noise/2) 

# plt.axhline(y=reward_dict["Optimal"], color='b', linestyle='--', label='Optimal Reward')  # Add a horizontal line for the optimal reward

# plt.axhline(y=reward_dict["Benchmark"], color='r', linestyle='--', label='Benchmark Reward')

# legend = plt.legend(fontsize=20)  # Show the legend
# legend.get_frame().set_alpha(0.3)

# now = datetime.now()
# fname = f"c:/Users/david/Desktop/AdversarialBanditwithKnapsacks_code/AdversarialKnapsacksCode/imgs/ff_across_noise/comparision_" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S") + "reward.pdf"
# os.makedirs(os.path.dirname(fname), exist_ok=True)
    
# Adjust y-axis limits to include the optimal reward
# plt.ylim(bottom=0, top=max(reward_dict["Optimal"], reward_dict["Benchmark"]) + 0.15 * plt.ylim()[1])

# plt.savefig(fname + "mult.pdf", format='pdf', bbox_inches='tight', dpi=1200)
# plt.show()
