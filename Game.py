import numpy as np
import AlgorithmsModule.Algorithms as A


class Game:

    def __init__(self, T, B, n, m, bandit=False):
        self.bandit = bandit

        self.T = T  # Total number of iterations
        self.t = 0  # Current iteration

        self.B = B  # Budget per resource
        self.m = m
        self.B_current = np.ones(self.m) * B  # Initialising the current budget
        self.rho = self.B / self.T

        self.n = n
        self.actions = np.arange(self.n)

        # self.GP = 0
        # self.GD = 0
        # if delta:
        #     self.HP = 2*np.sqrt(2*self.delta*np.log(8*self.n*self.T**2)) + \
        #           (2/self.rho)*np.sqrt(2*self.delta*np.log(8*self.n*(self.T**2)/self.rho))
        #     self.HD = 2*np.sqrt((2*self.delta/self.rho)*np.log(4*self.n*(self.T**2)))

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
        # Run a step of the designated algorithm
        algorithm = parameters["algorithm"]
        return algorithm(self.actions, self.B_current, self.rho, reward_vector, cost_vector, parameters, self.bandit, verbose=verbose)

    def compute_best_mixed_action(self, rewards_partial, costs_partial, lambda_value, rho):
        average_rewards = np.mean(rewards_partial)
        average_costs = np.mean(costs_partial)
        x = np.zeros(self.n)
        return x
        
    


    def run(self, rewards, costs, parameters, best_FD, verbose=False):
        for t in range(self.T):
            if parameters["algorithm"] == A.stochastic_with_prediction:
                parameters["empirical_action"] = self.compute_best_mixed_action(rewards[:t+1, :], costs[:t+1, :, :], parameters["lambda_value_predicted"], self.rho)
            if verbose:
                print(f"ITERATION {t} ")
            mixed_action, action, lambda_value, B_current, parameters, source = self.run_step(
                rewards[t], costs[t], parameters, verbose=verbose)
            self.update(rewards[t], costs[t], mixed_action, action, lambda_value, B_current, source, best_FD, verbose=verbose)
            self.t += 1
        return self, parameters

    def results_pseudo(self):
        return self.cumulative_pseudo_regret[-1]
    
    def results(self):
        return self.cumulative_regret[-1]
    
    def results_cost(self):
        return self.cumulative_cost
    
