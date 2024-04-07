# Adversarial Bandit with Knapsacks

This project implements the experimental analysis of the paper "Bandits with Knapsacks and Predictions". 

The novel adversarial algorithms introudced in the paper are presented in AlgorithmsModule. To add new algorithms, code the algorithm in the module and add the parameters for the algorithm to the file `constants.py`.

The GameModule introduces a class to easily run and store the restults of a Multi-armed bandit experiment.

The Regret Minimisers in the RegretMinimisersModule correspond to the ones suggested in the paper Castiglioni et al. (2022).

The DataGenerator module is used to generate the lognormal data. It is enough to introduce a new data generation process there to test the algorithms on new data.

The Plotter and the Runner are scripts that link `experiments.py` to the Modules, to have a cleaner run of the experiments.

## Getting Started

To run the project, follow these steps:

1. Clone the repository to your local machine.
2. Open the `constants.py` file and set the parameters according to your requirements. The parameters are already defaulted to the values mentioned in the paper.
3. Run the `experiments.py` script to start the experiments.

## Results

The tabular results of the experiments will be saved in the `results` directory. 
The plots are saved in the `imgs` directory.