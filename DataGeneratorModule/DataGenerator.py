import numpy as np


def normalize(x, x_max):
    return (x) / (x_max)

normilize_vectorized = np.vectorize(normalize)

class DataGenerator:
    """
    A class for generating data for the Adversarial Bandit with Knapsacks problem.
    """

    def __init__(self, n, m, T, B):
        """
        Initialize a DataGenerator object.

        Args:
            n (int): Number of actions.
            m (int): Number of resources.
            T (int): Number of iterations.
            B (float): Budget.

        Returns:
            None
        """
        self.n = n  # Number of actions
        self.m = m  # Number of resources
        self.T = T  # Number of iterations
        self.B = B  # Budget
        self.data = None  # Updated with generated data
        return

    def reset(self):
        """
        Reset the data of the DataGenerator Object.
        """
        self.data = None
        return

    def generate_data_lognormal(self, mean_rewards, sigma_rewards, mean_costs, sigma_costs):
        """
        Populates the field self.data with lognormal data.

        Args:
            mean_rewards (list): List of means to pass to the lognormal distribution of the costs.
            sigma_rewards (list): List of sigmas to pass to the lognormal distribution of the rewards.
            mean_costs (list): List of means to pass to the lognormal distribution of the costs.
            sigma_costs (list): List of sigmas to pass to the lognormal distribution of the costs.

        Returns:
            None
        """

        rewards = np.zeros((self.T, self.n - 1))
        costs = np.zeros((self.T, self.n - 1, self.m))
        for a in range(self.n - 1):
            rewards[:, a] = np.random.lognormal(mean_rewards[a], sigma_rewards[a], size=self.T)
            costs[:, a, :] = np.random.lognormal(mean_costs[a], sigma_costs[a], size=(self.T, self.m))

        rewards[rewards >= 5] = 5  # Discard all outlier values greater than 5
        costs[costs >= 5] = 5  # Discard all outlier values greater than 5
        rewards = normilize_vectorized(rewards, np.max(rewards))  # Normalize to cast the rewards in the 0, 1 range
        costs = normilize_vectorized(costs, np.max(costs))  # Normalize to cast the costs to the 0, 1 range

        rewards = np.hstack((rewards, np.zeros((self.T, 1))))
        costs = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        # Populate the field self.data
        self.data = (rewards.copy(), costs.copy())
        return

    def get_means(self):
        """
        Computes the means of the data in self.data

        Returns:
            tuple: Tuple containing the mean of the rewards and the mean of the costs
        """
        mean_rewards = np.mean(self.data[0], axis=0)
        mean_costs = np.mean(np.mean(self.data[1], axis=2), axis=0)
        return mean_rewards, mean_costs
    