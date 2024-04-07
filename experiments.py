#%% Importing the libraries
from datetime import datetime
import os
import numpy as np
import pickle

import pandas as pd

from Runner import run_experiments_script as R
from Plotter import plot_experiments_script as P
from DataGeneratorModule import DataGenerator as DG
#%% Import constants
from constants import (

    # Import seeds
    seeds,

    test, 

    # Import parameters for the experiments
    num_experiments, 
    best_FDs, best_FDs_rewards, 
    worst_FDs,
    stored, scenario,
    store_new_data,
    show_plots,
    verbose_results, 
    verbose_game, 

    # Import structural variables
    n, m, T, B,

    # Import parameters for data generation
    mean_rewards, sigma_rewards, 
    mean_costs, sigma_costs,
    
    # Import parameters for the algorithms
    parameters_pd_ff, 
    parameters_pd_bandit, 
    parameters_augmented_ff, 
    parameters_augmented_bandit, 
    
)

#%% Define all the sets of parameters to run the experiments on

# p=0.9 - Define the full feedback and bandit feedback with p=0.9
parameters_augmented_ff_pi09 = parameters_augmented_ff.copy()
parameters_augmented_ff_pi09["algorithm_name"] = "adversarial_with_prediction_ff_pi09"
parameters_augmented_bandit_pi09 = parameters_augmented_bandit.copy()
parameters_augmented_bandit_pi09["algorithm_name"] = "adversarial_with_prediction_bandit_pi09"

# p=0.5 - Define the full feedback and bandit feedback with p=0.5
parameters_augmented_ff_pi05 = parameters_augmented_ff.copy()
parameters_augmented_ff_pi05["p"] = 0.5
parameters_augmented_ff_pi05["algorithm_name"] = "adversarial_with_prediction_ff_pi05"
parameters_augmented_bandit_pi05 = parameters_augmented_bandit.copy()
parameters_augmented_bandit_pi05["p"] = 0.5
parameters_augmented_bandit_pi05["algorithm_name"] = "adversarial_with_prediction_bandit_pi05"

# mu_low - Define the full feedback and bandit feedback with mu=1/sqrt(T) [low], and default p=0.9
parameters_augmented_ff_pi09_mulow = parameters_augmented_ff.copy()
parameters_augmented_ff_pi09_mulow["mu"] = 1/np.sqrt(T)
parameters_augmented_ff_pi09_mulow["algorithm_name"] = "adversarial_with_prediction_ff_pi09_mulow"
parameters_augmented_bandit_pi09_mulow = parameters_augmented_bandit.copy()
parameters_augmented_bandit_pi09_mulow["mu"] = 1/np.sqrt(T)

# mu_high - Define the full feedback and bandit feedback with mu=2*sqrt(2*log(1/0.1)/(0.1*sqrt(T))) [high], and default p=0.9
parameters_augmented_bandit_pi09_mulow["algorithm_name"] = "adversarial_with_prediction_bandit_pi09_mulow"
parameters_augmented_ff_pi09_muhigh = parameters_augmented_ff.copy()
parameters_augmented_ff_pi09_muhigh["mu"] = (2*np.sqrt(2*np.log(1/0.1)))/(0.1*np.sqrt(T))
parameters_augmented_ff_pi09_muhigh["algorithm_name"] = "adversarial_with_prediction_ff_pi09_muhigh"
parameters_augmented_bandit_pi09_muhigh = parameters_augmented_bandit.copy()
parameters_augmented_bandit_pi09_muhigh["mu"] = (2*np.sqrt(2*np.log(1/0.1)))/(0.1*np.sqrt(T))
parameters_augmented_bandit_pi09_muhigh["algorithm_name"] = "adversarial_with_prediction_bandit_pi09_muhigh"

#%% Generate or Import the data
if stored:
    try:
        file_to_open = 'data.pkl' if not test else 'data_test.pkl'
        with open(file_to_open, 'rb') as file:  
            data, best_FDs, best_FDs_rewards, worst_FDs = pickle.load(file)
        
        # Check the data is stored in the right format
        assert len(data) == num_experiments
        for i in range(num_experiments):
            assert data[f"experiment_{i}"][0].shape == (T, n)
            assert data[f"experiment_{i}"][1].shape == (T, n, m)
        assert np.array(best_FDs).shape == (num_experiments, n)
        assert np.array(best_FDs_rewards).shape == (num_experiments, )
        assert np.array(worst_FDs).shape == (num_experiments, n)

    except FileNotFoundError:
        raise Exception("The data is not stored in the file")
    except AssertionError:
        raise Exception("The data is not stored in the right format")
else:
    data_generator = DG.DataGenerator(n, m, T, B)
    data, best_FDs, best_FDs_rewards, worst_FDs = R.generate_data(
                                                    seeds, num_experiments, 
                                                    data_generator, B, 
                                                    mean_rewards, sigma_rewards, 
                                                    mean_costs, sigma_costs, 
                                                    store=store_new_data,
                                                    test=test
                                                )


################### RUN THE EXPERIMENTS ###################
#%% Results with perfect predictions - outputs a table 
### Corresponds to the results table in the main text and the results table in the appendix

# Define the parameters to use in the experiments
run_build = {
    "seeds": seeds, 
    "num_experiments": num_experiments, 
    "data": data, 
    "best_FDs": best_FDs, 
    "best_FDs_rewards": best_FDs_rewards, 
    "worst_FDs": worst_FDs, 
    "noise": None,  # We are considering perfect predictions
    "scenario": scenario, 
    "verbose_results": verbose_results, 
    "verbose_game": verbose_game
}
parameters_to_run_table = {
    # MAIN TEXT
    "primal_dual_ff": parameters_pd_ff.copy(),
    "primal_dual_bandit": parameters_pd_bandit.copy(),
    "adversarial_with_prediction_ff_pi09": parameters_augmented_ff_pi09.copy(),
    "adversarial_with_prediction_bandit_pi09": parameters_augmented_bandit_pi09.copy(),
    
    # APPENDIX
    "adversarial_with_prediction_ff_pi09_mulow": parameters_augmented_ff_pi09_mulow.copy(),
    "adversarial_with_prediction_bandit_pi09_mulow": parameters_augmented_bandit_pi09_mulow.copy(),
    "adversarial_with_prediction_ff_pi09_muhigh": parameters_augmented_ff_pi09_muhigh.copy(),
    "adversarial_with_prediction_bandit_pi09_muhigh": parameters_augmented_bandit_pi09_muhigh.copy(),
    
    # OMITTED
    # "adversarial_with_prediction_ff_pi05": parameters_augmented_ff_pi05.copy(), 
    # "adversarial_with_prediction_bandit_pi05": parameters_augmented_bandit_pi05.copy(),
}

# Perform the experiments and save the results in a table
def run_table(run_build, parameters_to_run_table, show_plots=False, test=False):
    print("Running experiments with perfect prediction...")
    results_table = P.compute_experiment_table(run_build, parameters_to_run_table, show=show_plots)

    # Save the results in a csv file
    now = datetime.now ()
    current_directory = os.getcwd()
    if test:
        file_name = current_directory + "/results/test/table/" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S")
    else:
        file_name = current_directory + "/results/table/" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
    results_table.to_csv(file_name + "table.csv")

#%% Results ACROSS NOISE - outputs a table, saves corresponding bar plots
### Corresponds to the results ACROSS NOISE in the main text and the results across noise in the appendix
### Both for the Low and High noise levels sections

# Each of the following dictionaries defines an algorithm to run across noise against a benchmark algorithm.
# It is an inefficient implementation considering that we are not storing the values for the benchmark but we are rerunning it for each experiment.
parameters_to_run_across_noise_ff = {"benchmark": parameters_pd_ff.copy(), "test": parameters_augmented_ff.copy()}         
parameters_to_run_across_noise_bandit = {"benchmark": parameters_pd_bandit.copy(),  "test": parameters_augmented_bandit.copy()}
parameters_to_run_across_noise_ffmulow = {"benchmark": parameters_pd_ff.copy(), "test": parameters_augmented_ff_pi09_mulow.copy()}
parameters_to_run_across_noise_banditmulow = {"benchmark": parameters_pd_bandit.copy(),  "test": parameters_augmented_bandit_pi09_mulow.copy()}
parameters_to_run_across_noise_ffmuhigh = {"benchmark": parameters_pd_ff.copy(), "test": parameters_augmented_ff_pi09_muhigh.copy()}
parameters_to_run_across_noise_banditmuhigh = {"benchmark": parameters_pd_bandit.copy(),  "test": parameters_augmented_bandit_pi09_muhigh.copy()}
all_params_noise_experiments = {"ff_across_noise": parameters_to_run_across_noise_ff, 
                                "bandit_across_noise": parameters_to_run_across_noise_bandit, 
                                "ff_across_noise_muhigh": parameters_to_run_across_noise_ffmuhigh,
                                "bandit_across_noise_muhigh": parameters_to_run_across_noise_banditmuhigh,
                                "ff_across_noise_mulow": parameters_to_run_across_noise_ffmulow,
                                "bandit_across_noise_mulow": parameters_to_run_across_noise_banditmulow,
                                }

# Perform the experiments and save the results in a table and bar plots
def run_across_noise(run_build, all_params_noise_experiments, show_plots=False, test=False):
    # Define all the values on noise that must stored
    aux = [[str(x)[0:4], str(x)[0:4] + "_std"] for x in np.arange(0.0, 0.11, 0.01)]
    aux_bad_case = [[str(x)[0:4], str(x)[0:4] + "_std"] for x in np.arange(0.5, 2.15, 0.15)]
    

    columns_noise = ["Name"] + [item for sublist in aux for item in sublist] + ["Optimal", "Benchmark"]
    columns_noise_bad_case = ["Name"] + [item for sublist in aux_bad_case for item in sublist] + ["Optimal", "Benchmark"]
    results_noise_dataframe = pd.DataFrame(columns=columns_noise)
    results_noise_bad_case_dataframe = pd.DataFrame(columns=columns_noise_bad_case)
    for parameters_across_noise_key in all_params_noise_experiments.keys():
        name = parameters_across_noise_key
        params = all_params_noise_experiments[name]
        if test:
            name = "test/" + name
        print(f"Running experiments for {name}...")
        

        # Store the results in pdf files and outputs tabular results
        experiment_row = P.plot_across_noise(name, run_build.copy(), 
                            params["test"], params["benchmark"],
                            min_noise=0.0, max_noise=0.1, step_noise=0.01,
                            show=show_plots
                        )
        experiment_row_bad_case = P.plot_across_noise(name + "_bad_case", run_build.copy(),
                                    params["test"], params["benchmark"],
                                    min_noise=0.5, max_noise=2.0, step_noise=0.15,
                                    show=show_plots
                                )
        
        # Store the results in the dataframe
        row = {"Name": parameters_across_noise_key}
        row.update(experiment_row)
        row = pd.Series(row).to_frame().T
        row_bad = {"Name": parameters_across_noise_key}
        row_bad.update(experiment_row_bad_case)
        row_bad = pd.Series(row_bad).to_frame().T
        results_noise_dataframe = pd.concat([results_noise_dataframe, row])
        results_noise_bad_case_dataframe = pd.concat([results_noise_bad_case_dataframe, row_bad])

    # Save the tabular results in a csv file
    now = datetime.now()
    current_directory = os.getcwd()
    if test:
        file_name = current_directory + "/results/test/noise/" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S")
    else:
        file_name = current_directory + "/results/noise/" + now.strftime("%d") + now.strftime("%m") + now.strftime("%Y") + "_" + now.strftime("%H%M%S")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
    results_noise_dataframe.to_csv(file_name + "reward.csv", index=False)
    results_noise_bad_case_dataframe.to_csv(file_name + "reward_bad_case.csv", index=False)

#%% Results ACROSS TIME - outputs line plots
### Corresponds to the results ACROSS TIME in the appendix
parameters_to_run_ff_pi09 = {"Primal Dual": parameters_pd_ff.copy(), "Prediction Algorithm": parameters_augmented_ff_pi09.copy()}
parameters_to_run_bandit_pi09 = {"Primal Dual": parameters_pd_bandit.copy(), "Prediction Algorithm": parameters_augmented_bandit_pi09.copy()}
parameters_to_run_ff_pi05 = {"Primal Dual": parameters_pd_ff.copy(), "Prediction Algorithm": parameters_augmented_ff_pi05.copy()}
parameters_to_run_bandit_pi05 = {"Primal Dual": parameters_pd_bandit.copy(), "Prediction Algorithm": parameters_augmented_bandit_pi05.copy()}

parameters_to_run_across_time = {
    "ff_rho01_pi09": parameters_to_run_ff_pi09,
    "bandit_rho01_pi09": parameters_to_run_bandit_pi09,
    "ff_rho01_pi05": parameters_to_run_ff_pi05,
    "bandit_rho01_pi05": parameters_to_run_bandit_pi05,
}

def run_across_time(run_build, parameters_to_run_across_time, show_plots=False, test=False):
    print("Running experiments across time horizon...")

    for parameters_across_time_key in parameters_to_run_across_time.keys():
        print(" Running" + parameters_across_time_key)
        name = parameters_across_time_key
        if test:
            name = "test/" + name
        params = parameters_to_run_across_time[parameters_across_time_key]
        P.plot_experiments(name, run_build.copy(), params, show=show_plots)

#%% Main
if __name__ == "__main__":

    #%% Run table
    run_table(run_build, parameters_to_run_table, show_plots, test)

    #%% Run across noise
    run_across_noise(run_build, all_params_noise_experiments, show_plots, test)

    #%% Run across time
    run_across_time(run_build, parameters_to_run_across_time, show_plots, test)

# %%
