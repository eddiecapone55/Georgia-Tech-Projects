import numpy as np
import matplotlib.pyplot as plt

class Agent:
    """
    Base RL agent class for Q-Learning and SARSA.

    Args:
        env (gym.Env): The environment to train in.
        gamma (float): Discount factor for future rewards.
        alpha (float): Learning rate.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Decay factor for epsilon.
        episodes (int): Number of training episodes.
    """

    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=1000):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.q_table = np.zeros((*self.env.observation_space.nvec, self.env.action_space.n))
        self.rewards_per_episode = []  # Store rewards for plotting
        self.steps_per_episode = []  # Store steps for plotting

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample() # Explore
        else:
            return np.argmax(self.q_table[state]) # Exploit

    def train(self):
        """
        Train the agent using the specified RL algorithm.

        Returns:
            rewards_per_episode, steps_per_episode: The updated rewards and steps after training.
        """
        for episode in range(self.episodes):
            state = self.env.reset() # Reset environment at the beginning of each episode
            total_reward = 0
            steps = 0
            done = False

            while not done:
                # Choose action based on epsilon-greedy policy
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.learn(state, action, reward, next_state, done)

                total_reward += reward
                steps += 1
                state = next_state

            self.rewards_per_episode.append(total_reward)
            self.steps_per_episode.append(steps)
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay) # Decay epsilon

        return self.rewards_per_episode, self.steps_per_episode

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table (to be implemented in subclasses).
        """
        pass


class QLearningAgent(Agent):
    """Q-Learning agent implementing the off-policy TD control."""

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-Learning algorithm.
        """
        state_idx = tuple(state)
        next_state_idx = tuple(next_state)

        # Q-Learning update rule
        best_next_action = np.argmax(self.q_table[next_state_idx])  # Max Q-value for the next state
        td_target = reward + self.gamma * self.q_table[next_state_idx][best_next_action] * (not done)
        td_error = td_target - self.q_table[state_idx][action]

        # Update the Q-value
        self.q_table[state_idx][action] += self.alpha * td_error


class SARSAgent(Agent):
    """SARSA agent implementing the on-policy TD control."""

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            action = self.choose_action(state) # Choose action based on current state

            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                next_action = self.choose_action(next_state) # Choose action for next state

                self.learn(state, action, reward, next_state, next_action, done)

                total_reward += reward
                steps += 1
                state = next_state
                action = next_action

            self.rewards_per_episode.append(total_reward)
            self.steps_per_episode.append(steps)
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay) # Decay epsilon

        return self.rewards_per_episode, self.steps_per_episode

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Update the Q-table using the SARSA algorithm.

        Args:
            state (tuple): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state of the environment.
            done (bool): Whether the episode is complete.

        Returns:
            None
        """
        state_idx = tuple(state)
        next_state_idx = tuple(next_state)

        # SARSA update rule
        target = reward + self.gamma * self.q_table[next_state_idx][next_action] * (not done)
        self.q_table[state_idx][action] += self.alpha * (target - self.q_table[state_idx][action])