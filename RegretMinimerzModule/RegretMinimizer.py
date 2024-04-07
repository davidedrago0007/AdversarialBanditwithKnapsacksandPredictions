import numpy as np
import torch

class DualRegretMinimizer:
    """
    A class representing a dual regret minimizer.

    Attributes:
        starting_point (list): The starting point for the mixed action.
        learning_rate (float): The learning rate for updating the mixed action.
        rho (float): The budget per iteration.
        last_loss (numpy.ndarray): The last loss observed.
        verbose (bool): Flag indicating whether to print verbose output.

    Methods:
        next_element(): Returns the next mixed action element.
        observe_utility(loss): Observes the utility loss and updates the mixed action.
        reset(): Resets the last loss and mixed action to their initial values.
        set_verbose(verbose): Sets the verbose flag.

    """

    def __init__(self, starting_point, learning_rate, rho):
        self.starting_point = starting_point.copy()
        self.mixed_action_current = starting_point.copy()
        self.learning_rate = learning_rate
        self.rho = rho
        self.last_loss = np.zeros(np.shape(starting_point)[-1])
        self.verbose = False

    def next_element(self):
        """
        Returns the next element of the mixed action.

        Returns:
            numpy.ndarray: The next element of the mixed action.

        """
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
        """
        Observes the utility loss and updates the mixed action.

        Args:
            loss (numpy.ndarray): The utility loss.

        """
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
        """
        Resets the last loss and mixed action to their initial values.

        """
        self.last_loss = np.zeros(np.shape(self.starting_point)[-1])
        self.mixed_action_current = self.starting_point.copy()

    def set_verbose(self, verbose):
        """
        Sets the verbose flag.

        Args:
            verbose (bool): Flag indicating whether to print verbose output.

        """
        self.verbose = verbose

class Hedge:
    """
    Hedge class for implementing the Hedge algorithm.

    Parameters:
    - starting_point (numpy.ndarray): The starting point for the weights.
    - learning_rate (float): The learning rate for updating the weights.
    - nActions (int): The number of actions.

    Attributes:
    - learning_rate (float): The learning rate for updating the weights.
    - nActions (int): The number of actions.
    - w (numpy.ndarray): The weights for each action.
    - starting_point (numpy.ndarray): The starting point for the weights.
    - p (numpy.ndarray): The probability distribution over actions.
    - verbose (bool): Flag indicating whether to print verbose output.

    Methods:
    - next_element(context=None): Computes the next probability distribution over actions.
    - observe_utility(loss, cost=None): Updates the weights based on the observed utility.
    - reset(): Resets the weights and probability distribution to their starting values.
    - set_verbose(verbose): Sets the verbose flag.

    """

    def __init__(self, starting_point, learning_rate, nActions):
        self.learning_rate = learning_rate
        self.nActions = nActions
        self.w = np.ones(nActions)
        self.starting_point = starting_point.copy()
        self.p = starting_point.copy()
        self.verbose = False

    def next_element(self):
        """
        Computes the next probability distribution over actions.

        Parameters:

        Returns:
        - p (numpy.ndarray): The probability distribution over actions.

        """
        if self.verbose:
            print("         Running RP next element") 
            print("             w:", self.w)
        self.p = (self.w / np.sum(self.w))
        if self.verbose:
            print("             p:", self.p)
        return self.p.copy()

    def observe_utility(self, loss):
        """
        Updates the weights based on the observed utility.

        Parameters:
        - loss (numpy.ndarray): The loss values for each action.

        """
        if self.verbose:
            print("         Running HEDGE RP Observe utility") 
            print("             multiplier: ", np.exp(-self.learning_rate * loss))
        x = -self.learning_rate * loss
        x_tensor = torch.from_numpy(x)  # Convert numpy array to PyTorch tensor
        exp = torch.where(x_tensor > 50, torch.log1p(torch.exp(x_tensor)), 13.45678)
        exp = exp.numpy()
        exp[exp==13.45678] = np.exp(x)[exp==13.45678]
        self.w = self.w * exp  # Convert back to numpy array
         
        if self.verbose:
            print("             new w:", self.w)

    def reset(self):
        """
        Resets the weights and probability distribution to their starting values.

        """
        self.w = np.ones(self.nActions)
        self.p = self.starting_point.copy()

    def set_verbose(self, verbose):
        """
        Sets the verbose flag.

        Parameters:
        - verbose (bool): Flag indicating whether to print verbose output.

        """
        self.verbose = verbose

class EXP3(Hedge):
    """
    The EXP3 class represents an implementation of the EXP3 algorithm for adversarial bandit problems.

    Parameters:
    - starting_point: The starting point for the algorithm.
    - learning_rate: The learning rate for the algorithm.
    - nActions: The number of available actions.

    Methods:
    - next_element(): Returns the next action to take based on the current context.
    - observe_utility(loss, action, cost=None): Observes the utility of a chosen action.
    - set_verbose(verbose): Sets the verbosity level of the algorithm.
    - reset(): Resets the algorithm to its initial state.
    """

    def __init__(self, starting_point, learning_rate, nActions):
        super().__init__(starting_point, learning_rate, nActions)

    def next_element(self):
        """
        Returns the next action to take based on the current context.

        Parameters:
        - context: The current context (optional).

        Returns:
        - The next action to take.
        """
        return super().next_element()
    
    def observe_utility(self, loss, action):
        """
        Observes the utility of a chosen action.

        Parameters:
        - loss: The loss vector.
        - action: The chosen action.
        """
        if self.verbose:
            print("         Running EXP3 RP Observe utility") 
        ell_hat = np.zeros(self.nActions)
        ell_hat[action] = loss[action] / self.p[action]
        if self.verbose:
            print("             ell_hat:", ell_hat)
        super().observe_utility(ell_hat)

    def set_verbose(self, verbose):
        """
        Sets the verbosity level of the algorithm.

        Parameters:
        - verbose: True to enable verbose output, False otherwise.
        """
        return super().set_verbose(verbose)
    
    def reset(self):
        """
        Resets the algorithm to its initial state.
        """
        return super().reset()
