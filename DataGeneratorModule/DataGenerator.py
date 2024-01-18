import AlgorithmsModule.Algorithms as A
import numpy as np
import Game as G


def func(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def func2(x):
    return min(x, 1)

v_func = np.vectorize(func)
v_func2 = np.vectorize(func2)


class DataGenerator:
    def __init__(self, n, m, T, B):
        self.n = n  # Number of actions
        self.m = m  # Number of resources
        self.T = T  # Number of iterations
        self.B = B
        self.data = None
        return

    # def generate_data(self):
    #     valuation_primal_player = np.random.choice([i for i in range(1, self.n+1)], size=self.T)  # We make each item be valued
    #     valuation_dual_player = np.random.lognormal(size=(self.T, 1))
    #     self.data = np.column_stack((valuation_primal_player, valuation_dual_player))
    #     return

    def reset(self):
        self.data = None
        return


    def generate_data_lognormal(self, mean_rewards, sigma_rewards, mean_costs, sigma_costs):
        rewards = np.zeros((self.T, self.n))
        costs = np.zeros((self.T, self.n, self.m))
        for a in range(self.n):
            rewards[:, a] = np.random.lognormal(mean_rewards[a], sigma_rewards[a], size=self.T)
            costs[:, a, :] = np.random.lognormal(mean_costs[a], sigma_costs[a], size=(self.T, self.m))

        rewards_max = np.percentile(rewards, 75)
        costs_max = np.percentile(costs, 75)
        rewards[rewards >= rewards_max] = rewards_max
        costs[costs >= costs_max] = costs_max
        rewards = v_func(rewards, np.min(rewards), np.max(rewards))
        costs = v_func(costs, np.min(costs), np.max(costs))

        # rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
        # costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        # self.data = (rewards_aux.copy(), costs_aux.copy())
        self.data = (rewards.copy(), costs.copy())
        return 
    
    def exchange_highest_lowest(self, mean_rewards):
            max_index = np.argmax(mean_rewards)
            min_index = np.argmin(mean_rewards)

            # Swap the highest and lowest values
            mean_rewards[max_index], mean_rewards[min_index] = mean_rewards[min_index], mean_rewards[max_index]

            # Shuffle the rest of the values
            other_indices = [i for i in range(len(mean_rewards)) if i != min_index]
            np.random.shuffle(other_indices)
            shuffled_mean_rewards = [mean_rewards[i] for i in other_indices]
            shuffled_mean_rewards.insert(min_index, mean_rewards[min_index])

            return shuffled_mean_rewards
    
    def generate_data_lognormal_adversarial(self, mean_rewards, sigma_rewards, mean_costs, sigma_costs):
        final_rewards = np.zeros((self.T, self.n))
        final_costs = np.zeros((self.T, self.n, self.m))
        for i in range(self.n):
            rewards = np.zeros((self.T//self.n, self.n))
            costs = np.zeros((self.T//self.n, self.n, self.m))
            for a in range(self.n):
                rewards[:, a] = np.random.lognormal(mean_rewards[a], sigma_rewards[a], size=self.T//self.n)
                costs[:, a, :] = np.random.lognormal(mean_costs[a], sigma_costs[a], size=(self.T//self.n, self.m))

            rewards_max = np.percentile(rewards, 75)
            costs_max = np.percentile(costs, 75)
            rewards[rewards >= rewards_max] = rewards_max
            costs[costs >= costs_max] = costs_max
            rewards = v_func(rewards, np.min(rewards), np.max(rewards))
            costs = v_func(costs, np.min(costs), np.max(costs))

            final_rewards[i*self.T//self.n:(i+1)*self.T//self.n] = rewards.copy()
            final_costs[i*self.T//self.n:(i+1)*self.T//self.n] = costs.copy()

            mean_rewards = self.exchange_highest_lowest(mean_rewards)

        self.data = (final_rewards.copy(), final_costs.copy())
        return 

    def generate_data_lognormal_adversarial_v2(self, mean_rewards, sigma_rewards, mean_costs, sigma_costs, rate=1000):
        final_rewards = np.zeros((self.T, self.n))
        final_costs = np.zeros((self.T, self.n, self.m))
        for i in range(self.T//rate):
            rewards = np.zeros((rate, self.n))
            costs = np.zeros((rate, self.n, self.m))
            for a in range(self.n):
                rewards[:, a] = np.random.lognormal(mean_rewards[a], sigma_rewards[a], size=rate)
                costs[:, a, :] = np.random.lognormal(mean_costs[a], sigma_costs[a], size=(rate, self.m))

            rewards_max = np.percentile(rewards, 75)
            costs_max = np.percentile(costs, 75)
            rewards[rewards >= rewards_max] = rewards_max
            costs[costs >= costs_max] = costs_max
            rewards = v_func(rewards, np.min(rewards), np.max(rewards))
            costs = v_func(costs, np.min(costs), np.max(costs))

            final_rewards[i*rate:(i+1)*rate] = rewards.copy()
            final_costs[i*rate:(i+1)*rate] = costs.copy()

            mean_rewards = self.exchange_highest_lowest(mean_rewards)
            mean_costs = self.exchange_highest_lowest(mean_costs)

        # rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
        # costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        # self.data = (rewards_aux.copy(), costs_aux.copy())
        self.data = (final_rewards.copy(), final_costs.copy())
        return 

    def generate_data_beta(self, alpha_rewards, beta_rewards, alpha_costs, beta_costs):
        rewards = np.zeros((self.T, self.n))
        costs = np.zeros((self.T, self.n, self.m))
        for a in range(self.n):
            rewards[:, a] = np.random.beta(alpha_rewards[a], beta_rewards[a], self.T)
            costs[:, a, :] = np.random.beta(alpha_costs[a], beta_costs[a], (self.T, self.m))

        rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
        costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        self.data = (rewards_aux.copy(), costs_aux.copy())
        return

    def generate_adversarial_random_data(self, corruption):
        indices = np.random.random(self.T) < corruption

        alphas_reward = np.random.uniform(4.8, 5.2, self.n)
        alphas_cost = np.random.uniform(1.9, 2.1, self.n)
        rewards = np.hstack([np.random.beta(alpha, 5, size=(self.T, 1)) for alpha in alphas_reward])
        costs = np.hstack([np.random.beta(alpha, 1.5, size=(self.T, 1, self.m)) for alpha in alphas_cost])

        rewards[indices] = np.random.random(size=(self.T, self.n))[indices]
        costs[indices] = np.random.random(size=(self.T, self.n, self.m))[indices]

        rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
        costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        self.data = (rewards_aux.copy(), costs_aux.copy())
        return

    def generate_adversarial_min_data(self, parameters, corruption):
        indices = np.random.random(self.T) < corruption

        alphas_reward = np.random.uniform(4.8, 5.2, self.n)
        alphas_cost = np.random.uniform(1.9, 2.1, self.n)
        rewards = np.hstack([np.random.beta(alpha, 1, size=(self.T, 1)) for alpha in alphas_reward])
        costs = np.hstack([np.random.beta(alpha, 1, size=(self.T, 1, self.m)) for alpha in alphas_cost])

        best_FD = np.repeat(1 / self.n, self.n)  # We use a fake distribution, as it is not useful in our case

        game = G.Game(self.T, self.B, self.n, self.m)
        game.run(rewards, costs, A.primal_dual, parameters, best_FD)

        for t in range(self.T):
            if indices[t]:
                rewards[t, abs(game.vector_of_strategies[t] - max(game.vector_of_strategies[t])) < 1e-5] = 0.0
                costs[t, abs(game.vector_of_strategies[t] - max(game.vector_of_strategies[t])) < 1e-5, :] = np.repeat(1.0, self.m)

        rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
        costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        self.data = (rewards_aux.copy(), costs_aux.copy())
        return

    def generate_adversarial_switch_data(self, corruption):
        indeces_rewards = []
        indeces_costs = []
        for i in range(int(np.floor(corruption*np.sqrt(self.T)))):
            indeces_rewards.append(np.random.choice(np.arange(1, self.T, dtype=np.int64)))
            indeces_costs.append(np.random.choice(np.arange(1, self.T, dtype=np.int64)))
        np.sort(indeces_rewards)
        np.sort(indeces_costs)

        rewards_beta = np.random.beta(5, 1, size=(self.T, self.n))
        costs_beta = np.random.beta(2, 2, size=(self.T, self.n, self.m))

        rewards_lognormal = np.random.lognormal(1, 1, size=(self.T, self.n))
        costs_lognormal = np.random.lognormal(1, 1, size=(self.T, self.n, self.m))

        rewards_max = np.percentile(rewards_lognormal, 75)
        costs_max = np.percentile(costs_lognormal, 75)
        rewards_lognormal[rewards_lognormal >= rewards_max] = rewards_max
        costs_lognormal[costs_lognormal >= costs_max] = costs_max
        rewards_lognormal = v_func(rewards_lognormal, np.min(rewards_lognormal), np.max(rewards_lognormal))
        costs_lognormal = v_func(costs_lognormal, np.min(costs_lognormal), np.max(costs_lognormal))

        rewards = rewards_beta
        costs = costs_beta
        for i in range(0, len(indeces_rewards) - 1, 2):
            for a in range(self.n):
                r = np.random.random()
                if r < 0.5:
                    rewards[indeces_rewards[max(0, i)]:indeces_rewards[i + 1], a] = \
                        rewards_lognormal[indeces_rewards[max(0, i)]:indeces_rewards[i + 1], a]
                    costs[indeces_costs[max(0, i)]:indeces_costs[i + 1], a, :] = \
                        costs[indeces_costs[max(0, i)]:indeces_costs[i + 1], a, :]

        rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
        costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        self.data = (rewards_aux.copy(), costs_aux.copy())
        return

    # def generate_adversarial_FGSM_data(self, B, parameters, corruption_probabiity, corruption_level):
    #     indices = np.random.random(self.T) < corruption_probabiity
    #
    #     alphas_reward = np.random.uniform(4.8, 5.2)
    #     alphas_cost = np.random.uniform(1.9, 2.1)
    #     rewards = np.hstack([np.random.beta(alpha, 1, size=(self.T, 1)) for alpha in alphas_reward])
    #     costs = np.hstack([np.random.beta(alpha, 1, size=(self.T, 1, self.m)) for alpha in alphas_cost])
    #
    #     last_action = np.repeat(1 / self.n, self.n)
    #
    #     best_FD = np.repeat(1 / self.n, self.n)  # We use a fake distribution, as it is not useful in our case
    #
    #     game = G.Game(self.T, self.n, B)
    #     game.run(rewards, costs, A.primal_dual, parameters, best_FD)
    #     perturbation = np.zeros(self.n)
    #     for t in self.T:
    #         reward_vector = game.cumulative_reward_per_action[t] - game.cumulative_reward_per_action[t-1]
    #         cost_vector = game.cumulative_cost_per_action[t] - game.cumulative_cost_per_action[t-1]
    #         mixed_action = game.vector_of_strategies[t-1]
    #         lambda_value = game.vector_of_lambdas[t-1]
    #         last_loss = reward_vector * mixed_action - \
    #                     np.sum(lambda_value * cost_vector * mixed_action.reshape(self.n, 1), axis=1)
    #         perturbation += corruption_level*last_loss
    #
    #         reward_adversarial = reward_vector + perturbation
    #
    #         if indices[t]:
    #             rewards[t, :] += reward_adversarial
    #
    #     rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
    #     costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))
    #
    #     self.data = (rewards_aux.copy(), costs_aux.copy())
    #     return

    def generate_adversarial_mixed_case(self):
        rewards = np.random.beta(2, 2, (self.T, self.n))
        rewards[:int(0.5*self.T), 0] = 0.8
        rewards[int(0.5*self.T):, -1] = 1

        costs = np.random.beta(2, 1, (self.T, self.n, self.m))
        costs[:int(0.5*self.T), 0] = self.B/(0.6*self.T)
        costs[int(0.5*self.T):, -1] = self.B/(0.6*self.T)

        rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
        costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        self.data = (rewards_aux.copy(), costs_aux.copy())
        return

    def generate_adversarial_corner_case(self, length):
        rewards = np.zeros((self.T, self.n))
        rewards[0:length] = 0.001
        rewards[0:length, self.n-1] = 0.0
        rewards[length:self.T, self.n-1] = 1.0

        costs = np.zeros((self.T, self.n, self.m))
        costs[0:length] = self.B/length
        costs[0:length, self.n-1] = 0.0
        costs[length:self.T, self.n-1] = self.B/(self.T-length)

        rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
        costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        self.data = (rewards_aux.copy(), costs_aux.copy())
        return

    def generate_adversarial_good_case(self, mean_rewards, sigma_rewards):
        rewards = np.zeros((self.T, self.n))
        costs = np.zeros((self.T, self.n, self.m))

        for a in range(self.n):
            rewards[:, a] = np.random.lognormal(mean_rewards[a], sigma_rewards[a], self.T)

        rewards_max = np.percentile(rewards, 75)
        rewards[rewards >= rewards_max] = rewards_max
        rewards = v_func(rewards, np.min(rewards), np.max(rewards))

        alpha_costs = np.random.uniform(2, 4, self.n)

        for j in range(1, int(np.sqrt(self.T))):
            for a in range(self.n):
                costs[int(np.sqrt(self.T)*(j-1)):int(np.sqrt(self.T)*j), a, :] = \
                    np.random.beta(alpha_costs[a]*(1 - (j/np.sqrt(self.T))), 2, (int(np.sqrt(self.T)), self.m))

        rewards_aux = np.hstack((rewards, np.zeros((self.T, 1))))
        costs_aux = np.hstack((costs, np.zeros((self.T, 1, self.m))))

        self.data = (rewards_aux.copy(), costs_aux.copy())
        return


    def get_means(self):
        mean_rewards = np.mean(self.data[0], axis=0)
        mean_costs = np.mean(np.mean(self.data[1], axis=2), axis=0)
        return mean_rewards, mean_costs
    
    # import sys
    # sys.path.append('C://Users//david//Desktop//AdversarialBanditwithKnapsacks_code//AdversarialKnapsacksCode//DG//DataGenerator.py')
    # import DataGenerator.DataGenerator as DG