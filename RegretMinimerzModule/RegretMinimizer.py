import numpy as np


def div_by_0(x, y):
    if y!=0:
        return x/y
    else:
        return 0


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


v_div_by_0 = np.vectorize(div_by_0)


class DualRegretMinimizer:
    def __init__(self, starting_point, learning_rate, rho):
        self.starting_point = starting_point.copy()
        self.mixed_action_current = starting_point.copy()
        self.learning_rate = learning_rate
        self.rho = rho
        self.last_loss = np.zeros(np.shape(starting_point)[-1])

    def next_element(self):
        self.mixed_action_current = np.clip(self.mixed_action_current, 0.0, 1/self.rho)
        return self.mixed_action_current

    def observe_utility(self, loss):
        self.last_loss = loss
        gradient = -self.last_loss
        self.mixed_action_current -= self.learning_rate * gradient

    def reset(self):
        self.last_loss = np.zeros(np.shape(self.starting_point)[-1])
        self.mixed_action_current = self.starting_point.copy()

class Hedge:
    def __init__(self, starting_point, learning_rate, nActions):
        self.learning_rate = learning_rate
        self.nActions = nActions
        self.w = np.ones(nActions)
        self.starting_point = starting_point.copy()
        self.p = starting_point.copy()

    #     self.acc_loss = []
    #     self.acc_cost = []

    def next_element(self, context=None):
        self.p = (self.w / np.sum(self.w))
        return self.p.copy()

    def observe_utility(self, loss, cost=None):
    #     self.acc_loss.append(loss[self.action][0] + (self.acc_loss[-1] if self.acc_loss else 0.0))
    #     if cost is not None:
    #         self.acc_cost.append(cost[self.action][0] +  (self.acc_cost[-1] if self.acc_cost else 0.0))

        self.w = self.w * np.exp(-self.learning_rate * loss) 

    def reset(self):
        self.w = np.ones(self.nActions)
        self.p = self.starting_point.copy()
    
    #     self.acc_loss = []
    #     self.acc_cost = []

    # def results(self):
    #     return self.acc_loss, self.acc_cost, None

class EXP3(Hedge):
    def __init__(self, starting_point, learning_rate, nActions):
        super().__init__(starting_point, learning_rate, nActions)

    def next_element(self, context=None):
        return super().next_element(context=context)
    
    def observe_utility(self, loss, action, cost=None):
        ell_hat = np.zeros(self.nActions)
        ell_hat[action] = loss[action] / self.p[action]
        super().observe_utility(ell_hat)

