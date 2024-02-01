import RegretMinimerzModule.RegretMinimizer as R
import numpy as np


def primal_dual(actions, B_current, rho, reward_vector, cost_vector, parameters, bandit, verbose=False):
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
    p = parameters["p"]
    nu = parameters["nu"]
    mu = parameters["mu"]
    m = parameters["m"]
    B_current_A = parameters.get("B_current_A", np.repeat(parameters["B"]*p, m)).copy()
    B_current_WC = parameters.get("B_current_WC", np.repeat(parameters["B"]*(1-p), m)).copy()
    if verbose:
        print("Running the adversarial algorithm...")
        print("starting B_current_WC:", B_current_WC)
        print("starting B_current_A:", B_current_A)

    mixed_action_predicted = parameters["mixed_action_predicted"]
    prob = np.random.random()

    if prob < p-nu:  # with prob p-nu
        source = "Prediction"
        if verbose:
            print("  Entering source:", source)
        if all(B_current_A > 1):
            if verbose:
                print("     There is enough budget for the prediction")
            mixed_action = mixed_action_predicted.copy()
            action = np.random.choice(actions, p=mixed_action)

        else:

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
                
    elif (p-nu <= prob) and (prob < 1-nu-mu):  # with prob 1-p-mu
        if verbose:
            print("Entering source:", "WC_algorithm")
        mixed_action, action, lambda_value, B_current_WC, parameters, source = primal_dual(actions, B_current_WC, rho,
                                                                                        reward_vector, cost_vector,
                                                                                        parameters, bandit, verbose=verbose
                                                                                        )
        if verbose:
            print("mixed_action:", mixed_action)
        parameters["B_current_WC"] = B_current_WC.copy()
        if verbose:
            print("B_current_WC new:", B_current_WC)
    else:  # with prob nu+mu
        
        mixed_action = np.zeros(len(actions))
        action = None
        source = "Void"
        if verbose:
            print("Entering source:", source)

    if source != "WC_algorithm":  # update WC algorithm with the available feedback
        if bandit:
            reward_vector_aux, cost_vector_aux = np.zeros(len(reward_vector)), np.zeros(cost_vector.shape)
            B_c_aux = B_current_WC.copy()
            if verbose:
                print("     UPDATING wc in bandit setting...")
            _, _, lambda_value, _, parameters, _ = primal_dual(actions, B_c_aux, rho,
                                                               reward_vector_aux, cost_vector_aux,
                                                               parameters, bandit, verbose=verbose
                                                               )
        else:
            B_c_aux = B_current_WC.copy()
            reward_vector_aux, cost_vector_aux = reward_vector, cost_vector
            if verbose:
                print("     UPDATING wc in ff setting...")
            _, _, lambda_value, _, parameters, _ = primal_dual(actions, B_c_aux, rho,
                                                                reward_vector_aux, cost_vector_aux,
                                                                parameters, bandit, verbose=verbose
                                                                )
    return mixed_action, action, lambda_value, B_current, parameters, source


def stochastic_with_prediction(actions, B_current, rho, reward_vector, cost_vector, parameters, bandit, verbose=False):
    t = parameters["t"]
    n = parameters["n"]
    rho = parameters["rho"]
    B_current = parameters["B_current"]
    total_reward = parameters["total_reward"]
    average_reward = (parameters["average_reward"]*(t-1) + reward_vector)*t
    average_cost = (parameters["average_cost"]*(t-1) + cost_vector)*t
    h = np.sqrt(t * np.log(((parameters["delta"]/4)*parameters["T"])**2))

    if t < parameters["Delta"]:
        source = "Prediction"
        if verbose:
            print("Entering source:", source)
        mixed_action = parameters["mixed_action_predicted"].copy()
        action = np.random.choice(actions, p=mixed_action)
        lambda_value = parameters["lambda_value_predicted"].copy()
        B_current -= cost_vector[action, :]
        
    if t >= parameters["Delta"]:
        # If the algorithm is in the primal-dual state, play the worst-case algorithm
        if parameters["state"] == "primal_dual":
            if verbose:
                print("Entering source:", "WC_algorithm")
            mixed_action, action, lambda_value, B_current, parameters, source = primal_dual(actions, B_current, rho,
                                                                                            reward_vector, cost_vector,
                                                                                            parameters, bandit, verbose=verbose
                                                                                            )
            if verbose:
                    print("B_current new:", B_current)
        # Otherwise check the performance of the prediction and decide whether to switch to the primal-dual state
        else:
            if verbose:
                print("Checking the performance...")
            # Get the empirical best reward for the current iteration
            empirical_reward = 0
            # Check if the empirical reward is close enough to the predicted reward
            if np.abs(total_reward - empirical_reward) <= 3*h:
                source = "Prediction"
                if verbose:
                    print("Entering source:", source)
                mixed_action = parameters["mixed_action_predicted"].copy()
                action = np.random.choice(actions, p=mixed_action)
                lambda_value = parameters["lambda_value_predicted"].copy()
                B_current -= cost_vector[action, :]
            else:
                parameters["state"] = "primal_dual"
                if verbose:
                    print("Entering source:", "WC_algorithm")
                mixed_action, action, lambda_value, B_current, parameters, source = primal_dual(actions, B_current, rho,
                                                                                            reward_vector, cost_vector,
                                                                                            parameters, bandit, verbose=verbose
                                                                                            )
                if verbose:
                    print("B_current new:", B_current)
    
    if bandit:
        parameters["total_loss"] += np.sum(reward_vector*mixed_action) - np.sum(lambda_value * (rho - np.sum(cost_vector*mixed_action.reshape(n, 1), axis=0)))
        parameters["B_current"] = B_current.copy()
    else:
        parameters["total_loss"] += np.sum(reward_vector * mixed_action)
        parameters["B_current"] = B_current.copy()
        
    parameters["t"] = t+1
    parameters["average_reward"] = average_reward
    parameters["average_cost"] = average_cost
    return mixed_action, action, lambda_value, B_current, parameters, source


