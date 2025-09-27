import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, 
                 n_actions, n_agents, batch_size, debug=False):
        """
        Initialize the multi-agent replay buffer.
        
        Parameters:
            max_size (int): Maximum size of the replay buffer.
            critic_dims (int): Dimension of the critic state space (shared among agents).
            actor_dims (list): List of dimensions for each agent's actor state space.
            n_actions (int): Number of actions for each agent.
            n_agents (int): Number of agents in the environment.
            batch_size (int): Batch size for sampling.
            debug (bool): If True, enables debug logging for transitions.
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.debug = debug  # Enable debug logging if necessary

        # Initializing critic-specific memories (shared across all agents)
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        # Initializing actor-specific memories for each agent
        self.actor_state_memory = [np.zeros((self.mem_size, self.actor_dims[i])) for i in range(self.n_agents)]
        self.actor_new_state_memory = [np.zeros((self.mem_size, self.actor_dims[i])) for i in range(self.n_agents)]
        self.actor_action_memory = [np.zeros((self.mem_size, self.n_actions)) for _ in range(self.n_agents)]

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        """
        Stores a transition in the replay buffer, managing memory as a ring buffer.

        Parameters:
            raw_obs (list): List of current observations for each agent.
            state (ndarray): Joint state for the critic.
            action (list): List of actions taken by each agent.
            reward (list): List of rewards received by each agent.
            raw_obs_ (list): List of next observations for each agent.
            state_ (ndarray): Joint next state for the critic.
            done (ndarray): Boolean array indicating if the episode ended for each agent.
        """
        index = self.mem_cntr % self.mem_size  # Circular buffer for efficient memory usage

        # Storing each agent's individual states, actions, and next states
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        # Storing joint states and rewards for the critic
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done.astype(bool)  # Ensure 'done' is boolean

        self.mem_cntr += 1

        # Optional debug logging to verify transition storage
        if self.debug:
            print(f"Stored transition at index {index}, mem_cntr={self.mem_cntr}")

    def sample_buffer(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            actor_states (list): List of current states for each agent.
            states (ndarray): Batch of joint states for the critic.
            actions (list): List of actions for each agent.
            rewards (ndarray): Batch of rewards for each agent.
            actor_new_states (list): List of next states for each agent.
            states_ (ndarray): Batch of joint next states for the critic.
            terminal (ndarray): Batch of terminal flags for each agent.
        """
        max_mem = min(self.mem_cntr, self.mem_size)  # Limit sampling to available memory
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # Sampling critic-specific memories
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        # Sampling actor-specific memories for each agent
        actor_states = [np.asarray(self.actor_state_memory[agent_idx][batch]) for agent_idx in range(self.n_agents)]
        actor_new_states = [np.asarray(self.actor_new_state_memory[agent_idx][batch]) for agent_idx in range(self.n_agents)]
        actions = [np.asarray(self.actor_action_memory[agent_idx][batch]) for agent_idx in range(self.n_agents)]

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        """
        Checks if the buffer has enough samples for a batch.

        Returns:
            bool: True if buffer has enough samples, False otherwise.
        """
        return self.mem_cntr >= self.batch_size