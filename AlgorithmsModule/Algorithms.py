import RegretMinimerzModule.RegretMinimizer as R
import numpy as np


def primal_dual(actions, B_current, rho, reward_vector, cost_vector, parameters, bandit):
    source = "WC_algorithm"
    RP = parameters.get("RP")
    RD = parameters.get("RD")
    T = parameters.get("T")
    learning_rate = parameters["learning_rate"]
    n = len(actions)
    m = len(B_current)
    mixed_action = RP.p.copy()

    # Initialise RP and RD
    if not RP:
        if bandit:
            RP = R.EXP3(starting_point=np.ones(n)/n, nActions=n, learning_rate=learning_rate)
            RP.learning_rate_primal = learning_rate
        else:
            RP = R.Hedge(starting_point=np.ones(n)/n, nActions=n, learning_rate=learning_rate)
            RP.learning_rate_primal = learning_rate
    if not RD:
        RD = R.DualRegretMinimizer(starting_point=np.ones(m)/(m*rho), learning_rate=learning_rate, rho=rho)
        RD.learning_rate_dual = learning_rate

    # Run step of primal
    if all(B_current > 1):
        mixed_action = RP.next_element()
        action = np.random.choice(actions, p=mixed_action)
    else:
        action = None
        mixed_action = np.repeat(None, n)
        lambda_value = np.repeat(None, m)
        return mixed_action, action, lambda_value, B_current, parameters, source

    # Run step of dual
    lambda_value = RD.next_element()

    # Observe request and update available budget
    if isinstance(action, (np.int32, int)):
        if bandit:
            B_current -= cost_vector[action]
            primal_loss = np.zeros(n)
            primal_loss[action] = reward_vector[action] - np.sum(lambda_value * cost_vector[action, :])
            dual_loss = - np.sum(lambda_value * (rho - cost_vector[action]))
            RP.observe_utility(loss=primal_loss, action=action)
        else:
            B_current -= cost_vector[action]

            primal_loss = reward_vector*mixed_action - np.sum(lambda_value * cost_vector*mixed_action.reshape(n, 1), axis=1)
            dual_loss = - np.sum(lambda_value * (rho - cost_vector * mixed_action.reshape(n, 1)), axis=0)

            RP.observe_utility(loss=primal_loss)
        RD.observe_utility(loss=dual_loss)

    parameters["RP"] = RP
    parameters["RD"] = RD
    return mixed_action, action, lambda_value, B_current, parameters, source


def adversarial_with_prediction(actions, B_current, rho, reward_vector, cost_vector, parameters, bandit):
    p = parameters["p"]
    nu = parameters["nu"]
    mu = parameters["mu"]
    m = parameters["m"]
    B_current_A = parameters.get("B_current_A", np.repeat(parameters["B"]*p, m))
    B_current_WC = parameters.get("B_current_WC", np.repeat(parameters["B"]*(1-p), m))
    # print("starting B_current_WC:", B_current_WC)
    # print("starting B_current_A:", B_current_A)

    mixed_action_predicted = parameters["mixed_action_predicted"]
    prob = np.random.random()

    if prob < p-nu:  # with prob p-nu
        source = "Prediction"
        # print("Entering source:", source)
        if all(B_current_A > 1):
            # print("There is enough budget for the prediction")
            mixed_action = mixed_action_predicted
            action = np.random.choice(actions, p=mixed_action)

        else:

            action = None
            mixed_action = np.zeros(len(actions))

        if not bandit:
            # Draw the action from the predicted strategy, only if the budget is enough
            # print("FF CASE")
            if isinstance(action, (np.int32, int)):
                # print("cost_vector:", cost_vector)
                # print("Cost to remove", cost_vector[action, :])
                B_current_A -= cost_vector[action, :]
                parameters["B_current_A"] = B_current_A.copy()
                # print("B_current_A new:", B_current_A)
        else:
            # print("BANDIT CASE")
            if isinstance(action, (np.int32, int)):
                # print("cost_vector:", cost_vector)
                # print("Cost to remove", cost_vector[action, :])
            
                B_current_A -= cost_vector[action, :]
                parameters["B_current_A"] = B_current_A.copy()
                # print("B_current_A new:", B_current_A)
                
    elif (p-nu <= prob) and (prob < 1-nu-mu):  # with prob 1-p-nu
        
        mixed_action, action, lambda_value, B_current_WC, parameters, source = primal_dual(actions, B_current_WC, rho,
                                                                                        reward_vector, cost_vector,
                                                                                        parameters, bandit
                                                                                        )
        # print("Entering source:", source)
        # print("mixed_action:", mixed_action)
        parameters["B_current_WC"] = B_current_WC.copy()
        # print("B_current_WC new:", B_current_WC)
    else:  # with prob nu+mu
        
        mixed_action = np.zeros(len(actions))
        action = None
        source = "Void"
        # print("Entering source:", source)

    if source != "WC_algorithm":  # update WC algorithm with the available feedback
        if bandit:
            reward_vector_aux, cost_vector_aux = np.zeros(len(reward_vector)), np.zeros(cost_vector.shape)
            B_c_aux = B_current_WC.copy()
            
            _, _, lambda_value, _, parameters, _ = primal_dual(actions, B_c_aux, rho,
                                                               reward_vector_aux, cost_vector_aux,
                                                               parameters, bandit
                                                               )
        else:
            B_c_aux = B_current_WC.copy()
            reward_vector_aux, cost_vector_aux = reward_vector, cost_vector

            _, _, lambda_value, _, parameters, _ = primal_dual(actions, B_c_aux, rho,
                                                                reward_vector_aux, cost_vector_aux,
                                                                parameters, bandit
                                                                )
    return mixed_action, action, lambda_value, B_current, parameters, source




