import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                 n_agents, n_actions, name, chkpt_dir, use_batch_norm=True):
        """
        Critic Network for evaluating state-action pairs in multi-agent settings.
        Args:
            beta (float): Learning rate.
            input_dims (int): Dimension of the state space.
            fc1_dims (int): First hidden layer size.
            fc2_dims (int): Second hidden layer size.
            n_agents (int): Number of agents.
            n_actions (int): Number of actions.
            name (str): Name for the checkpoint file.
            chkpt_dir (str): Directory for saving checkpoints.
            use_batch_norm (bool): Whether to use batch normalization.
        """
        super(CriticNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)

        # Layers with optional batch normalization
        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.bn1 = nn.BatchNorm1d(fc1_dims) if use_batch_norm else nn.Identity()
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.BatchNorm1d(fc2_dims) if use_batch_norm else nn.Identity()
        self.q = nn.Linear(fc2_dims, 1)

        # Optimizer and scheduler setup
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        """
        Forward pass for the Critic network, evaluating Q-value of state-action pairs.
        Args:
            state (Tensor): State tensor.
            action (Tensor): Action tensor.
        Returns:
            Tensor: Q-value for the given state-action pair.
        """
        x = F.relu(self.bn1(self.fc1(T.cat([state, action], dim=1))))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        q = self.q(x)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir, use_batch_norm=True):
        """
        Actor Network for selecting actions in multi-agent settings.
        Args:
            alpha (float): Learning rate.
            input_dims (int): Dimension of the state space.
            fc1_dims (int): First hidden layer size.
            fc2_dims (int): Second hidden layer size.
            n_actions (int): Number of actions.
            name (str): Name for the checkpoint file.
            chkpt_dir (str): Directory for saving checkpoints.
            use_batch_norm (bool): Whether to use batch normalization.
        """
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)

        # Layers with optional batch normalization
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.bn1 = nn.BatchNorm1d(fc1_dims) if use_batch_norm else nn.Identity()
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.BatchNorm1d(fc2_dims) if use_batch_norm else nn.Identity()
        self.pi = nn.Linear(fc2_dims, n_actions)

        # Optimizer and scheduler setup
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass for the Actor network, outputting action probabilities.
        Args:
            state (Tensor): State tensor.
        Returns:
            Tensor: Action probabilities.
        """
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        pi = T.tanh(self.pi(x))
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))