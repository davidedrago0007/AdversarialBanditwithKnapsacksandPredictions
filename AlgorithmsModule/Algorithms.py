import RegretMinimisersModule.RegretMinimizer as R
import numpy as np


def primal_dual(actions, B_current, rho, reward_vector, cost_vector, parameters, bandit, verbose=False):
    """
    Implements the primal-dual algorithm from Castiglioni et al. (2021).

    Args:
        actions (list): List of possible actions.
        B_current (numpy.ndarray): Current budget vector.
        rho (float): Dual parameter.
        reward_vector (numpy.ndarray): Vector of rewards for each action.
        cost_vector (numpy.ndarray): Matrix of costs for each action and resource.
        parameters (dict): Dictionary of algorithm parameters.
        bandit (bool): Flag indicating whether the problem is a bandit problem or full feedback problem.
        verbose (bool, optional): Flag indicating whether to print verbose output. Defaults to False.

    Returns:
        tuple: Tuple containing the mixed action, chosen action, dual variable, updated budget, updated parameters, and source.

    """
    if verbose:
        print("Running the primal-dual algorithm...")
        print(" reward vector:", reward_vector)
        print(" cost vector:", cost_vector)
    source = "WC_algorithm"
    RP = parameters.get("RP")
    RD = parameters.get("RD")
    T = parameters.get("T")
    learning_rate = parameters["learning_rate"]
    n = parameters["n"]
    m = parameters["m"]
    mixed_action = RP.p.copy()
    if verbose:
        print("     Initial mixed_action:", mixed_action)

    # Initialise RP and RD
    if not RP:
        print("       Initialising RP...")
        if bandit:
            RP = R.EXP3(starting_point=np.ones(n)/n, nActions=n, learning_rate=learning_rate)
            RP.learning_rate_primal = learning_rate
        else:
            RP = R.Hedge(starting_point=np.ones(n)/n, nActions=n, learning_rate=learning_rate)
            RP.learning_rate_primal = learning_rate
    if not RD:
        if verbose:
            print("     Initialising RD...")
        RD = R.DualRegretMinimizer(starting_point=np.ones(m)/(m*rho), learning_rate=learning_rate, rho=rho)
        RD.learning_rate_dual = learning_rate
    RP.set_verbose(verbose)
    RD.set_verbose(verbose)

    # Run step of primal
    if all(B_current > 1):
        if verbose:
            print("     There is enough budget.")
        mixed_action = RP.next_element()
        action = np.random.choice(actions, p=mixed_action)
    else:
        if verbose:
            print("     The budget is depleted.")
        action = None
        mixed_action = np.repeat(None, n)
        lambda_value = np.repeat(None, m)
        return mixed_action, action, lambda_value, B_current, parameters, source

    # Run step of dual
    if verbose:
        print("     Running step of dual...")
    lambda_value = RD.next_element()

    # Observe request and update available budget
    if isinstance(action, (np.int32, int)):
        if verbose:
            print("     A valid action was played")
        if bandit:
            if verbose:
                print("     Bandit- getting the lossess...")
            B_current -= cost_vector[action, :]
            primal_loss = np.zeros(n)
            primal_loss[action] = -(reward_vector[action] - np.sum(lambda_value * cost_vector[action, :]))
            dual_loss = - np.sum(lambda_value * (rho - cost_vector[action]))
            if verbose:
                print("         Primal Loss:", primal_loss)
                print("         Dual Loss:", dual_loss)
            RP.observe_utility(loss=primal_loss, action=action)
        else:
            if verbose:
                print("     Full Feedback- getting the lossess...")

            B_current -= cost_vector[action]

            if verbose:
                print("     The mixed action used is:", mixed_action)
                print("    The cost vector used is:", cost_vector)
                print("    The rho used is:", rho)
                print("    The lambda value used is:", lambda_value)
            primal_loss = -(reward_vector*mixed_action - np.sum(lambda_value * cost_vector*mixed_action.reshape(n, 1), axis=1))
            dual_loss = - np.sum(lambda_value * (rho - np.sum(cost_vector * mixed_action.reshape(n, 1), axis=0)))
            if verbose:
                print("         Primal Loss:", primal_loss)
                print("         Dual Loss:", dual_loss)
            RP.observe_utility(loss=primal_loss)
        RD.observe_utility(loss=dual_loss)


    parameters["RP"] = RP
    parameters["RD"] = RD
    return mixed_action, action, lambda_value, B_current, parameters, source


def adversarial_with_prediction(actions, B_current, rho, reward_vector, cost_vector, parameters, bandit, verbose=False):
    """
    Run the adversarial algorithm with prediction.

    Args:
        actions (list): List of possible actions.
        B_current (numpy.ndarray): Current budget vector.
        rho (float): Learning rate.
        reward_vector (numpy.ndarray): Vector of rewards for each action.
        cost_vector (numpy.ndarray): Matrix of costs for each action and resource.
        parameters (dict): Dictionary of algorithm parameters.
        bandit (bool): Flag indicating whether it's a bandit setting or not.
        verbose (bool, optional): Flag indicating whether to print verbose output. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - mixed_action (numpy.ndarray): Mixed action vector.
            - action (int or None): Selected action.
            - lambda_value (float): Dual variable value.
            - B_current (numpy.ndarray): Updated budget vector.
            - parameters (dict): Updated algorithm parameters.
            - source (str): Source of the action selection.

    """
    # Retrieve the parameters from the dictionary
    p = parameters["p"]
    mu = parameters["mu"]
    m = parameters["m"]
    B_current_A = parameters.get("B_current_A", np.repeat(parameters["B"]*p, m)).copy()
    B_current_WC = parameters.get("B_current_WC", np.repeat(parameters["B"]*(1-p), m)).copy()
    B_aux_WC = parameters.get("B_aux_WC", np.repeat(parameters["B"]*1.0, m)).copy()
    if verbose:
        print("Running the adversarial algorithm...")
        print("starting B_current_WC:", B_current_WC)
        print("starting B_current_A:", B_current_A)

    mixed_action_predicted = parameters["mixed_action_predicted"]
    prob = np.random.random()

    # with prob p-mu
    if prob < p-mu:
        source = "Prediction"
        if verbose:
            print("  Entering source:", source)
        if all(B_current_A > 1):
            if verbose:
                print("     There is enough budget for the prediction")
            mixed_action = mixed_action_predicted.copy()
            action = np.random.choice(actions, p=mixed_action)

        else:
            source = "Void"
            action = None
            mixed_action = np.zeros(len(actions))

        if not bandit:
            # Draw the action from the predicted strategy, only if the budget is enough
            if verbose:
                print("FF CASE")
            if isinstance(action, (np.int32, int)):
                if verbose:
                    print("cost_vector:", cost_vector)
                    print("Cost to remove", cost_vector[action, :])
                B_current_A -= cost_vector[action, :]
                parameters["B_current_A"] = B_current_A.copy()
                if verbose:
                    print("B_current_A new:", B_current_A)
        else:
            if verbose:
                print("BANDIT CASE")
            if isinstance(action, (np.int32, int)):
                if verbose:
                    print("cost_vector:", cost_vector)
                    print("Cost to remove", cost_vector[action, :])
            
                B_current_A -= cost_vector[action, :]
                parameters["B_current_A"] = B_current_A.copy()
                if verbose:
                    print("B_current_A new:", B_current_A)
    
    # with prob 1-p-mu
    elif (p-mu <= prob) and (prob < 1-2*mu):
        if verbose:
            print("Entering source:", "WC_algorithm")
        source = "Void"
        if all(B_current_WC > 1):
            if not bandit:
                B_aux_WC_previous = B_aux_WC.copy()
                mixed_action, action, lambda_value, B_aux_WC, parameters, source = primal_dual(actions, B_aux_WC, rho,
                                                                                            reward_vector, cost_vector,
                                                                                            parameters, bandit, verbose=verbose
                                                                                            )
                if verbose:
                    print("mixed_action:", mixed_action)
                    B_current_WC -= (B_aux_WC_previous - B_aux_WC)
                parameters["B_current_WC"] = B_current_WC.copy()
                parameters["B_aux_WC"] = B_aux_WC.copy()
                if verbose:
                    print("B_current_WC new:", B_current_WC)
            else:
                mixed_action, action, lambda_value, B_current_WC, parameters, source = primal_dual(actions, B_current_WC, rho,
                                                                                            reward_vector, cost_vector,
                                                                                            parameters, bandit, verbose=verbose
                                                                                            )
                if verbose:
                    print("mixed_action:", mixed_action)
                parameters["B_current_WC"] = B_current_WC.copy()
                if verbose:
                    print("B_current_WC new:", B_current_WC)
        else:
            action = None
            mixed_action = np.zeros(len(actions))

    # with prob 2mu
    else:
        
        mixed_action = np.zeros(len(actions))
        action = None
        source = "Void"
        if verbose:
            print("Entering source:", source)

    if source != "WC_algorithm":  # update WC algorithm with the available feedback
        if bandit:
            reward_vector_aux, cost_vector_aux = np.zeros(len(reward_vector)), np.zeros(cost_vector.shape)
            B_temp = B_current_WC.copy()
            if verbose:
                print("     UPDATING wc in bandit setting...")
            _, _, lambda_value, _, parameters, _ = primal_dual(actions, B_temp, rho,
                                                               reward_vector_aux, cost_vector_aux,
                                                               parameters, bandit, verbose=verbose
                                                               )
        else:
            reward_vector_aux, cost_vector_aux = reward_vector, cost_vector
            if verbose:
                print("     UPDATING wc in ff setting...")
            _, _, lambda_value, B_aux_WC, parameters, _ = primal_dual(actions, B_aux_WC, rho,
                                                                reward_vector_aux, cost_vector_aux,
                                                                parameters, bandit, verbose=verbose
                                                                )
            parameters["B_aux_WC"] = B_aux_WC.copy()
    return mixed_action, action, lambda_value, B_current, parameters, source
