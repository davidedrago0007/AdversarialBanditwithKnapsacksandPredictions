import AlgorithmsModule.Algorithms as A
import numpy as np

class Game:
    """
    Represents a game with adversarial bandit with knapsacks setting.

    Attributes:
    - T (int): Total number of iterations.
    - B (int): Budget per resource.
    - n (int): Number of actions.
    - m (int): Number of resources.
    - bandit (bool): Flag indicating whether the game is a bandit game.

    Methods:
    - __init__(self, T:int, B:int, n:int, m:int, bandit:bool = False): Initializes a Game object.
    - copy(self): Creates a copy of the current Game object.
    - reset(self): Resets the game to its initial state.
    - update(self, reward_vector, cost_vector, mixed_action, action, lambda_value, B_current, source, best_FD, verbose=False): Updates the game state based on the given inputs.
    - run_step(self, reward_vector, cost_vector, parameters, verbose=False): Runs a step of the designated algorithm.
    - compute_best_mixed_action(self, rewards_partial, costs_partial, lambda_value, rho): Computes the best mixed action based on partial rewards and costs.
    - run(self, rewards, costs, parameters, best_FD, verbose=False): Runs the game.

    """

    def __init__(self, T:int, B:int, n:int, m:int, bandit:bool = False):
        """
        Initializes a Game object.

        Parameters:
        - T (int): Total number of iterations.
        - B (int): Budget per resource.
        - n (int): Number of actions.
        - m (int): Number of resources.
        - bandit (bool, optional): Flag indicating whether the game is a bandit game. Defaults to False.
        """
        self.bandit = bandit

        self.T = T  # Total number of iterations
        self.t = 0  # Current iteration

        self.B = B  # Budget per resource
        self.m = m  # Number of resources
        self.B_current = np.ones(self.m) * B  # Initialising the current budget
        self.rho = self.B / self.T  # Budget per iteration

        self.n = n  # Number of actions
        self.actions = np.arange(self.n)

        # Instantiate the memory for the experiment data
        self.vector_of_actions = np.zeros(self.T)
        self.vector_of_strategies = np.zeros((self.T, self.n))
        self.vector_of_lambdas = np.zeros((self.T, self.m))
        self.vector_of_sources = np.array([None for _ in range(self.T)])

        self.best_FD_B_current = np.ones(self.m) * B
        self.cumulative_reward_per_action = np.zeros((self.T, self.n))
        self.expected_cumulative_reward = np.zeros(self.T)
        self.cumulative_cost_per_action = np.zeros((self.T, self.n, self.m))
        self.cumulative_reward = np.zeros(self.T)
        self.cumulative_cost = np.zeros((self.T, self.m))
        self.best_action = np.zeros(self.T, int)
        self.regret_per_iteration = np.zeros(self.T)
        self.cumulative_regret = np.zeros(self.T)
        self.pseudo_regret_per_iteration = np.zeros(self.T)
        self.cumulative_pseudo_regret = np.zeros(self.T)

        return

    def copy(self):
        """
        Creates a copy of the current Game object.

        Returns:
            A new Game object with the same attribute values as the current object.
        """
        game = Game(self.T, self.B, self.n, self.m, bandit=self.bandit)

        game.B_current = self.B_current
        game.vector_of_lambdas = self.vector_of_lambdas.copy()
        game.vector_of_sources = self.vector_of_sources.copy()
        game.vector_of_actions = self.vector_of_actions.copy()
        game.vector_of_strategies = self.vector_of_strategies.copy()

        game.best_FD_B_current = self.best_FD_B_current.copy()
        game.cumulative_reward_per_action = self.cumulative_reward_per_action.copy()
        game.expected_cumulative_reward = self.expected_cumulative_reward.copy()
        game.cumulative_cost_per_action = self.cumulative_cost_per_action.copy()
        game.cumulative_reward = self.cumulative_reward.copy()
        game.cumulative_cost = self.cumulative_cost.copy()
        game.best_action = self.best_action.copy()
        game.regret_per_iteration = self.regret_per_iteration.copy()
        game.cumulative_regret = self.cumulative_regret.copy()
        game.pseudo_regret_per_iteration = self.pseudo_regret_per_iteration.copy()
        game.cumulative_pseudo_regret = self.cumulative_pseudo_regret.copy()

        return game

    def reset(self):
        """
        Resets the game to its initial state.
        """
        self.B_current = np.ones(self.m) * self.B
        self.vector_of_actions = np.zeros(self.T, dtype=np.int32)
        self.vector_of_strategies = np.zeros((self.T, self.n))
        self.vector_of_lambdas = np.zeros((self.T, self.m))
        self.vector_of_sources = np.array([None for _ in range(self.T)])

        self.best_FD_B_current = np.ones(self.m) * self.B
        self.cumulative_reward_per_action = np.zeros((self.T, self.n))
        self.expected_cumulative_reward = np.zeros(self.T)
        self.cumulative_cost_per_action = np.zeros((self.T, self.n, self.m))
        self.cumulative_reward = np.zeros(self.T)
        self.cumulative_cost = np.zeros((self.T, self.m))
        self.best_action = np.zeros(self.T, int)
        self.regret_per_iteration = np.zeros(self.T)
        self.cumulative_regret = np.zeros(self.T)
        self.pseudo_regret_per_iteration = np.zeros(self.T)
        self.cumulative_pseudo_regret = np.zeros(self.T)

        return

    def update(self, reward_vector, cost_vector, mixed_action, action, lambda_value, B_current, source, best_FD, verbose=False):
        """
        Updates the game state based on the given inputs.

        Parameters:
        - reward_vector (numpy.ndarray): Vector of rewards for each action.
        - cost_vector (numpy.ndarray): Matrix of costs for each action and resource.
        - mixed_action (numpy.ndarray): Vector of mixed actions.
        - action (int): Chosen action.
        - lambda_value (float): Lambda value.
        - B_current (numpy.ndarray): Current budget per resource.
        - source (object): Source of the update.
        - best_FD (numpy.ndarray): Best feasible decision budget per resource.
        - verbose (bool, optional): Flag indicating whether to print verbose output. Defaults to False.
        """
        self.B_current = B_current
        self.vector_of_lambdas[self.t] = lambda_value
        self.vector_of_sources[self.t] = source
        self.vector_of_actions[self.t] = action
        self.vector_of_strategies[self.t] = mixed_action

        self.cumulative_reward_per_action[self.t] = self.cumulative_reward_per_action[self.t-1] + reward_vector
        self.cumulative_cost_per_action[self.t] = self.cumulative_cost_per_action[self.t-1] + cost_vector

        if isinstance(action, (np.int32, int)):
            reward = reward_vector[action]
            cost = cost_vector[action, :]
            exp_reward = np.sum(reward_vector * mixed_action)
        else:
            reward = 0
            cost = 0
            exp_reward = 0

        self.expected_cumulative_reward[self.t] = self.expected_cumulative_reward[self.t - 1] + exp_reward
        self.cumulative_reward[self.t] = self.cumulative_reward[self.t-1] + reward
        self.cumulative_cost[self.t, :] = self.cumulative_cost[self.t-1, :] + cost
        list_rewards = self.cumulative_reward_per_action[self.t, :]
        self.best_action[self.t] = np.argmax(list_rewards)

        self.best_FD_B_current -= np.sum(cost_vector.T*best_FD, axis=1)
        if all(self.best_FD_B_current > 1):
            reward_FD = np.sum(reward_vector * best_FD)
        else:
            reward_FD = 0

        self.regret_per_iteration[self.t] = reward_FD - reward
        self.pseudo_regret_per_iteration[self.t] = reward_FD - exp_reward

        self.cumulative_regret[self.t] = self.cumulative_regret[self.t-1] + self.regret_per_iteration[self.t]
        self.cumulative_pseudo_regret[self.t] = self.cumulative_pseudo_regret[self.t - 1] + self.pseudo_regret_per_iteration[self.t]

        if verbose:
            print("   mixed_action:", mixed_action)
            print("   lambda:", lambda_value)
            print("   action:", action)
            print("   reward:", reward)
            print("   cumulative_reward:", self.cumulative_reward[self.t])
            print("   cumulative cost:", self.cumulative_cost[self.t, :])
            print("   best_FD_reward:", reward_FD)
            print("   regret in this iteration:", self.regret_per_iteration[self.t])
            print("   cumulative_regret:", self.cumulative_regret[self.t])
        return

    def run_step(self, reward_vector, cost_vector, parameters, verbose=False):
        """
        Runs a step of the designated algorithm.

        Parameters:
        - reward_vector (numpy.ndarray): Vector of rewards for each action.
        - cost_vector (numpy.ndarray): Matrix of costs for each action and resource.
        - parameters (dict): Dictionary of algorithm parameters.
        - verbose (bool, optional): Flag indicating whether to print verbose output. Defaults to False.

        Returns:
        - The result of the algorithm step.
        """
        # Run a step of the designated algorithm
        algorithm = parameters["algorithm"]
        return algorithm(self.actions, self.B_current, self.rho, reward_vector, cost_vector, parameters, self.bandit, verbose=verbose)

    def compute_best_mixed_action(self, rewards_partial, costs_partial, lambda_value, rho):
        """
        Computes the best mixed action based on partial rewards and costs.

        Parameters:
        - rewards_partial (numpy.ndarray): Partial vector of rewards.
        - costs_partial (numpy.ndarray): Partial matrix of costs.
        - lambda_value (float): Lambda value.
        - rho (float): Budget per iteration.

        Returns:
        - The best mixed action.
        """
        average_rewards = np.mean(rewards_partial)
        average_costs = np.mean(costs_partial)
        x = np.zeros(self.n)
        return x

    def run(self, rewards, costs, parameters, best_FD, verbose=False):
        """
        Runs the game.

        Parameters:
        - rewards (numpy.ndarray): Matrix of rewards for each iteration and action.
        - costs (numpy.ndarray): 3D matrix of costs for each iteration, action, and resource.
        - parameters (dict): Dictionary of algorithm parameters.
        - best_FD (numpy.ndarray): Best feasible decision budget per resource.
        - verbose (bool, optional): Flag indicating whether to print verbose output. Defaults to False.

        Returns:
        - The game object and the updated parameters.
        """
        # Implementation of the game logic
        pass
        for t in range(self.T):
            if verbose:
                print(f"ITERATION {t} ")
            mixed_action, action, lambda_value, B_current, parameters, source = self.run_step(
                rewards[t], costs[t], parameters, verbose=verbose)
            self.update(rewards[t], costs[t], mixed_action, action, lambda_value, B_current, source, best_FD, verbose=verbose)
            self.t += 1
        return self, parameters

    def results_pseudo(self):
        """
        Returns the cumulative pseudo regret of the game.

        Returns:
        - The cumulative pseudo regret.
        """
        return self.cumulative_pseudo_regret[-1]
    
    def results(self):
        """
        Returns the cumulative regret of the game.

        Returns:
        - The cumulative regret.
        """
        return self.cumulative_regret[-1]
    
    def results_cost(self):
        """
        Returns the cumulative cost of the game.

        Returns:
        - The cumulative cost.
        """
        return self.cumulative_cost
    
