import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from tabulate import tabulate

from Runner.run_experiments_script import run

def compose_results_table(names_list, results_list):
    """
    Composes a results table using the given names list and results list.

    Args:
        names_list (list): List of names for the rows of the table.
        results_list (list): List of results for the columns of the table.

    Returns:
        pandas.DataFrame: The composed results table.
    """
    results_table = pd.DataFrame(results_list)
    results_table.index = names_list
    results_table = results_table.reindex(columns=["Total Reward", "Optimal Reward", 
                                                   "TVD", "Cumulative Cost", 
                                                   "Stopping Iteration", "Standard Deviation", 
                                                   "Horizon", "Budget", "Bandit"])
    return results_table

def show_results_table(results_table):
    """
    Displays the given results table.

    Args:
        results_table (pandas.DataFrame): The results table to be displayed.
    """
    print(tabulate(results_table, results_table.columns.tolist(), tablefmt="fancy_grid"))

def compute_experiment_table(run_build, parameters_dict, show=False):
    """
    Runs experiments using the given run build and parameters dictionary, and displays the results table.

    Args:
        run_build (dict): Dictionary containing the necessary parameters for running the experiments.
        parameters_dict (dict): Dictionary containing the parameters for each algorithm.

    Returns:
        pandas.DataFrame: The results table.
    """
    name_list = []
    res_list = []
    for parameters_key in parameters_dict.keys():
        print(" Running algorithm " + parameters_dict[parameters_key]["algorithm_name"])
        parameters_test_aux = parameters_dict[parameters_key].copy()
        name, res, _ = run(run_build["seeds"], run_build["num_experiments"], parameters_test_aux, run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["worst_FDs"], run_build["noise"], run_build["scenario"], run_build["verbose_results"], run_build["verbose_game"])
        name_list.append(name)
        res_list.append(res)
    results_table = compose_results_table(name_list, res_list)
    if show:
        show_results_table(results_table)
    return results_table


def plot_experiments(exp_name, run_build, parameters_dict, show=False):
    """
    Plots the cumulative reward, cost, and multiplier for the given experiment name, run build, and parameters dictionary.

    Args:
        exp_name (str): Name of the experiment.
        run_build (dict): Dictionary containing the necessary parameters for running the experiments.
        parameters_dict (dict): Dictionary containing the parameters for each algorithm.
    """
    algorithms = parameters_dict.keys()
    B = parameters_dict[list(algorithms)[0]]["B"]
    T = parameters_dict[list(algorithms)[0]]["T"]

    avg_rewards = {k: [] for k in algorithms}
    std_rewards = {k: [] for k in algorithms}
    avg_costs = {k: [] for k in algorithms}
    std_costs = {k: [] for k in algorithms}
    avg_mult = {k: [] for k in algorithms}
    std_mult = {k: [] for k in algorithms}

    for k in algorithms:
        print("  Running for algorithm" + k)
        _, res, avg_sequences =   run(run_build["seeds"], run_build["num_experiments"], parameters_dict[k], run_build["data"], run_build["best_FDs"], run_build["best_FDs_rewards"], run_build["worst_FDs"], run_build["noise"], run_build["scenario"], run_build["verbose_results"], run_build["verbose_game"])
        avg_rewards[k] = avg_sequences["avg_rewards"]
        std_rewards[k] = avg_sequences["std_rewards"]
        avg_costs[k] = avg_sequences["avg_costs"]
        std_costs[k] = avg_sequences["std_costs"]
        avg_mult[k] = avg_sequences["avg_mult"]
        std_mult[k] = avg_sequences["std_mult"]

    now = datetime.now()
    current_directory = os.getcwd()
    fname = current_directory + f"/imgs/{exp_name}/comparison_" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S")
    
    x = np.arange(0, T, 1)

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

    ax_final.legend(fontsize=14, loc='upper left')
    ax_final.set_xlabel(r"$Time$", fontsize=16)
    ax_final.set_ylabel("Cumulative Reward", fontsize=16)
    
    f_final.suptitle(f'Comparision for B={B}', fontsize=16)

    plt.savefig(fname + "_reward.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    if show:
        plt.show()
    else:
        plt.close()


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

    ax_final.legend(fontsize=14, loc='upper left')
    ax_final.set_xlabel(r"$Time$", fontsize=16)
    ax_final.set_ylabel("Cumulative Cost", fontsize=16)
    
    f_final.suptitle(f'Comparision for B={B}', fontsize=16)

    plt.savefig(fname + "_cost.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    if show:
        plt.show()
    else:
        plt.close()
    return

def plot_across_noise(exp_name, run_build, parameters_test, parameters_benchmark, min_noise=0.0, max_noise=0.1, step_noise=0.01, show=False):
    """
    Plots the total reward vs. noise level for a given experiment. 
    The noise level is expressed as the standard deviation of the Gaussian noise.

    Args:
        exp_name (str): Name of the experiment.
        run_build (dict): Dictionary containing the necessary parameters for running the experiment.
        parameters_test (dict): Parameters for the test phase of the experiment.
        parameters_benchmark (dict): Parameters for the benchmark phase of the experiment.
        min_noise (float, optional): Minimum noise level. Defaults to 0.0.
        max_noise (float, optional): Maximum noise level. Defaults to 0.1.
        step_noise (float, optional): Step size for noise levels. Defaults to 0.01.

    Returns:
        dict: Dictionary containing the total reward and standard deviation for each noise level.
    """
    reward_dict = dict()
    total_rewards = []
    # Iterate over noise levels
    sd_noise_levels = np.arange(min_noise, max_noise + step_noise, step_noise)
    for noise_level in sd_noise_levels:
        # Set the noise level
        print(f" Starting {exp_name} experiments for noise level:", noise_level)
        run_build["noise"] = np.array([np.abs(np.random.normal(0, noise_level, parameters_benchmark["n"])) for _ in range(run_build["num_experiments"])]) # * signs
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
    current_directory = os.getcwd()
    fname = current_directory + f"/imgs/{exp_name}/comparison_" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    
    # Adjust y-axis limits to include the optimal reward
    plt.ylim(bottom=0, top=max(optimal_reward, benchmark_reward) + 0.15 * plt.ylim()[1])
    
    plt.savefig(fname + "_noise.pdf", format='pdf', bbox_inches='tight', dpi=1200)
    if show:
        plt.show()
    else:
        plt.close()

    return reward_dict