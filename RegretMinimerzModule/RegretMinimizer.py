import numpy as np
import torch

class DualRegretMinimizer:
    def __init__(self, starting_point, learning_rate, rho):
        self.starting_point = starting_point.copy()
        self.mixed_action_current = starting_point.copy()
        self.learning_rate = learning_rate
        self.rho = rho
        self.last_loss = np.zeros(np.shape(starting_point)[-1])
        self.verbose = False

    def next_element(self):
        if self.verbose:
            print("        DUAL: running next_element...")
            print("         Original lambda value:", self.mixed_action_current)
        if np.sum(np.abs(self.mixed_action_current)) > 1/self.rho:
            self.mixed_action_current /= np.sum(np.abs(self.mixed_action_current))
            self.mixed_action_current *= 1/self.rho
        if self.verbose:
            print("         Actual Lambda Used:", self.mixed_action_current)
        return self.mixed_action_current.copy()

    def observe_utility(self, loss):
        self.last_loss = loss
        gradient = -self.last_loss
        if self.verbose:
            print("        DUAL: running observe_utility...")
            print("         Loss:", loss)
            print("         Loss used:", self.last_loss)
            print("         Gradient:", gradient)
            print("         Past lambda value:", self.mixed_action_current)
            print("         Change:", -1*self.learning_rate * gradient)
        self.mixed_action_current -= self.learning_rate * gradient
        if self.verbose:
            print("         New lambda value:", self.mixed_action_current)

    def reset(self):
        self.last_loss = np.zeros(np.shape(self.starting_point)[-1])
        self.mixed_action_current = self.starting_point.copy()

    def set_verbose(self, verbose):
        self.verbose = verbose

class Hedge:
    def __init__(self, starting_point, learning_rate, nActions):
        self.learning_rate = learning_rate
        self.nActions = nActions
        self.w = np.ones(nActions)
        self.starting_point = starting_point.copy()
        self.p = starting_point.copy()
        self.verbose = False

    #     self.acc_loss = []
    #     self.acc_cost = []

    def next_element(self, context=None):
        if self.verbose:
            print("         Runnning RP next element") 
            print("             w:", self.w)
        self.p = (self.w / np.sum(self.w))
        if self.verbose:
            print("             p:", self.p)
        return self.p.copy()

    def observe_utility(self, loss, cost=None):
    #     self.acc_loss.append(loss[self.action][0] + (self.acc_loss[-1] if self.acc_loss else 0.0))
    #     if cost is not None:
    #         self.acc_cost.append(cost[self.action][0] +  (self.acc_cost[-1] if self.acc_cost else 0.0))
        if self.verbose:
            print("         Runnning HEDGE RP Observe utility") 
            print("             multiplier: ", np.exp(-self.learning_rate * loss))
        x = -self.learning_rate * loss
        # print("             x: ", x)
        x_tensor = torch.from_numpy(x)  # Convert numpy array to PyTorch tensor
        exp = torch.where(x_tensor > 50, torch.log1p(torch.exp(x_tensor)), 13.45678)
        # print("             exp: ", exp)
        exp = exp.numpy()
        exp[exp==13.45678] = np.exp(x)[exp==13.45678]
        # print("             exp2: ", exp)
        self.w = self.w * exp  # Convert back to numpy array
        # self.w = self.w * np.exp(-self.learning_rate * loss)
         
        if self.verbose:
            print("             new w:", self.w)

    def reset(self):
        self.w = np.ones(self.nActions)
        self.p = self.starting_point.copy()
    
    #     self.acc_loss = []
    #     self.acc_cost = []

    # def results(self):
    #     return self.acc_loss, self.acc_cost, None
        
    def set_verbose(self, verbose):
        self.verbose = verbose

class EXP3(Hedge):
    def __init__(self, starting_point, learning_rate, nActions):
        super().__init__(starting_point, learning_rate, nActions)

    def next_element(self, context=None):
        return super().next_element(context=context)
    
    def observe_utility(self, loss, action, cost=None):
        if self.verbose:
            print("         Runnning EXP3 RP Observe utility") 
        ell_hat = np.zeros(self.nActions)
        ell_hat[action] = loss[action] / self.p[action]
        if self.verbose:
            print("             ell_hat:", ell_hat)
        super().observe_utility(ell_hat)

    def set_verbose(self, verbose):
        return super().set_verbose(verbose)
    
    def reset(self):
        return super().reset()
