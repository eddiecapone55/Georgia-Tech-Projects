import numpy as np

class ValueIterationPlanner:
    """
    Planner that uses the Value Iteration algorithm to determine the optimal policy for a given environment.
    Args:
        env (gym.Env): The traffic environment.
        gamma (float): Discount factor for future rewards.
        theta (float): Threshold for stopping value iteration (determines convergence).
    """
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        # Use the shape of nvec for the observation space
        self.obs_shape = self.env.observation_space.nvec
        self.policy = np.zeros(np.prod(self.obs_shape), dtype=int)  # Initial policy
        self.value_function = np.zeros(np.prod(self.obs_shape))
        self.value_iteration()

    def state_to_index(self, state):
        """Converts a state (tuple) into an index for the value function/policy array."""
        return np.ravel_multi_index(state, self.obs_shape)

    def value_iteration(self):
        """
        Perform value iteration to compute the optimal value function and policy.
        """
        while True:
            delta = 0
            Q = np.zeros((np.prod(self.obs_shape), self.env.action_space.n), dtype=np.float64)
            # iterate over all possible states
            for state in np.ndindex(*self.obs_shape):
                state_index = self.state_to_index(state)
                # iterate over all actions
                for a in range(self.env.action_space.n):
                    # calculate the expected value of taking action `a` in state `state`
                    for prob, next_state, reward, done in self.env.P[state][a]:
                        next_state_index = self.state_to_index(next_state)
                        Q[state_index, a] += prob * (reward + self.gamma * self.value_function[next_state_index] * (not done))
                # update the value function with the best action value
                best_action_value = np.max(Q[state_index])
                delta = max(delta, np.abs(self.value_function[state_index] - best_action_value))
                self.value_function[state_index] = best_action_value
            # check for convergence
            if delta < self.theta:
                break
        # extract the policy from the Q-table
        for state in np.ndindex(*self.obs_shape):
            state_index = self.state_to_index(state)
            self.policy[state_index] = np.argmax(Q[state_index])

    def choose_action(self, state):
        """
        Select the action based on the learned policy.
        Args:
            state (tuple): The current state of the environment.
        Returns:
            int: The action chosen by the policy.
        """
        state_index = self.state_to_index(state)
        return self.policy[state_index]


class PolicyIterationPlanner:
    """
    Planner that uses the Policy Iteration algorithm to determine the optimal policy for a given environment.
    Args:
        env (gym.Env): The traffic environment.
        gamma (float): Discount factor for future rewards.
        theta (float): Threshold for stopping policy evaluation.
    """
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.obs_shape = self.env.observation_space.nvec
        self.policy = np.zeros(np.prod(self.obs_shape), dtype=int)  # Initial policy
        self.value_function = np.zeros(np.prod(self.obs_shape))
        self.policy_iteration()

    def state_to_index(self, state):
        """Converts a state (tuple) into an index for the value function/policy array."""
        return np.ravel_multi_index(state, self.obs_shape)

    def evaluate_policy(self):
        """
        Evaluate the current policy by computing the value function for all states.
        """
        while True:
            delta = 0
            for state in np.ndindex(*self.obs_shape):
                state_index = self.state_to_index(state)
                old_value = self.value_function[state_index]
                action = self.policy[state_index]
                # Calculate value of the current policy
                new_value = 0
                for prob, next_state, reward, done in self.env.P[state][action]:
                    next_state_index = self.state_to_index(next_state)
                    new_value += prob * (reward + self.gamma * self.value_function[next_state_index] * (not done))
                self.value_function[state_index] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < self.theta:
                break

    def improve_policy(self):
        """
        Improve the current policy by making it greedy with respect to the current value function.
        Returns:
            bool: True if the policy is stable (i.e., no changes were made), False otherwise.
        """
        policy_stable = True
        for state in np.ndindex(*self.obs_shape):
            state_index = self.state_to_index(state)
            old_action = self.policy[state_index]
            # Find the action that maximizes the value function
            action_values = np.zeros(self.env.action_space.n)
            for a in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[state][a]:
                    next_state_index = self.state_to_index(next_state)
                    action_values[a] += prob * (reward + self.gamma * self.value_function[next_state_index] * (not done))
            new_action = np.argmax(action_values)
            self.policy[state_index] = new_action
            if old_action != new_action:
                policy_stable = False
        return policy_stable

    def policy_iteration(self):
        """
        Perform policy iteration by alternately evaluating and improving the policy.
        """
        while True:
            self.evaluate_policy()
            if self.improve_policy():
                break

    def choose_action(self, state):
        """
        Select the action based on the learned policy.
        Args:
            state (tuple): The current state of the environment.
        Returns:
            int: The action chosen by the policy.
        """
        state_index = self.state_to_index(state)
        return self.policy[state_index]