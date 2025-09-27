import torch as T
import numpy as np
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01, initial_noise_scale=1.0):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.noise_scale = initial_noise_scale  # Added noise scale for action exploration
        self.agent_name = 'agent_%s' % agent_idx
        # Actor and Critic networks
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # Flatten and prepare observation for actor
        if isinstance(observation, np.ndarray):
            observation = observation.flatten()
        
        state = T.tensor(np.array(observation), dtype=T.float).to(self.actor.device)
        state = state.unsqueeze(0)
        
        self.actor.eval()
        actions = self.actor.forward(state)
        self.actor.train()
        
        # Adjusted noise addition with decay over time
        self.noise_scale = max(self.noise_scale * 0.99, 0.1)
        noise = self.noise_scale * T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise
        
        # Convert continuous action to discrete action
        discrete_action = self.map_continuous_to_discrete(action)
        
        return discrete_action

    def map_continuous_to_discrete(self, action):
        # Map continuous actions to discrete actions for Overcooked - 6 actions: 0=up, 1=down, 2=left, 3=right, 4=stay, 5=interact
        
        # Pick the highest value among the action outputs to decide direction
        action_idx = T.argmax(action).item()
        
        # Ensure the index corresponds to one of the six discrete actions
        if action_idx >= 6:
            action_idx = 5  # Fall back to "interact" if out of bounds

        return action_idx

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Actor network update
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Critic network update
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    # More detailed debug for parameter updates
    def log_network_parameters(self):
        print(f"Agent {self.agent_name} network parameters updated with tau={self.tau}")