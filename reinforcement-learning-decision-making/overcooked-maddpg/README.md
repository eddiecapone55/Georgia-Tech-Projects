# CS7642 Project 3: Multi-Agent Reinforcement Learning in Overcooked Environment

## Overview
This project applies **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** to solve the Overcooked environment, a multi-agent task where two agents collaborate to deliver as many onion soups as possible within a limited number of steps. The environment is based on the popular Overcooked video game and requires strategic cooperation to maximize the reward.

The primary goal is to develop an RL algorithm that achieves high performance across three different layouts (`cramped_room`, `asymmetric_advantages`, and `forced_coordination`) by delivering an average of at least 7 soups per episode.

## Code Structure
This repository contains the following files:

### 1. `agent.py`
Defines the **Agent** class, which implements the MADDPG agents' actor and critic networks along with methods for selecting actions and updating network parameters. Key components include:
- `choose_action`: Chooses an action based on the current observation and adds noise for exploration.
- `update_network_parameters`: Soft updates for target networks to ensure stability in training.
- `log_network_parameters`: Optional debug function for logging network updates.

### 2. `buffer.py`
Contains the **MultiAgentReplayBuffer** class, which stores experiences and allows for efficient sampling during training. Main features:
- `store_transition`: Stores experiences for each agent.
- `sample_buffer`: Retrieves a batch of samples for training.
- `ready`: Checks if the buffer has enough samples to start training.

### 3. `environment_setup.py`
Provides the environment setup for the Overcooked layouts. Functions include:
- `setup_environment`: Initializes the Overcooked environment with specific layout, horizon, and reward-shaping parameters.

### 4. `evaluate.py`
Evaluates the performance of trained agents across all layouts. It runs a specified number of episodes and records metrics such as average score and soups delivered. Key functions:
- `evaluate_agent`: Main function that sets up environments, loads trained models, and evaluates performance. Generates plots for analysis.

### 5. `maddpg.py`
Implements the **MADDPG** class that manages the interactions among agents, their learning processes, and coordinated actions. Key functions:
- `choose_action`: Executes each agent's action selection.
- `learn`: Trains the agents using multi-agent experience replay and collaborative reward shaping.
- `save_checkpoint` and `load_checkpoint`: Saves and loads models for each agent.

### 6. `main.py`
The main script for training the MADDPG agents in the Overcooked environment. It initializes the environment, agents, and memory, then executes the training loop. Key features:
- Processes observations and actions for compatibility with Overcooked’s discrete action space.
- Tracks performance metrics, including rewards and inter-agent distance.
- Saves plots for various metrics and training progress.

### 7. `networks.py`
Defines the **ActorNetwork** and **CriticNetwork** classes used by the agents. Each network uses batch normalization and an optional learning rate scheduler. Key functions:
- `forward`: Forward pass for the actor and critic networks.
- `save_checkpoint` and `load_checkpoint`: Saves and loads model checkpoints for reproducibility.

### 8. `plotting.py`
Contains helper functions for plotting key metrics to analyze agent performance, including:
- `plot_learning_curve`: Plots average rewards over episodes.
- `plot_soup_delivery`: Tracks the number of soups delivered over time.
- `plot_action_distribution`: Shows the distribution of actions taken by agents.
- `plot_avg_reward_per_step`: Visualizes average rewards per step.
- `plot_inter_agent_distance`: Displays the inter-agent distance over episodes to assess collaboration.

## Installation
The Overcooked environment requires Python 3.7. The following packages are needed:
```bash
pip install overcooked-ai torch numpy matplotlib gym
```

## Usage 
1. Training: Run `main.py` to train the MADDPG agents. The script trains the agents across three layouts and saves performance metrics.
```bash
python main.py
```

2. Evaluation: To evaluate the trained agents, run `evaluate.py`. This script loads saved models, runs evaluations across layouts, and generates plots.
```bash
python evaluate.py
```

## File output
- **Model Checkpoints:** Saved in `tmp/maddpg/<layout>/`
- **Plots:**
    - `learning_curve.png`: Average rewards over training episodes.
    - `soup_delivery.png`: Soups delivered across episodes.
    - `action_distribution_png`: Distribution of actions taken by agents.
    - `avg_reward_per_step.png`: Average reward per step.
    - `inter_agent_distance.png`: Average inter-agent distance.
 
## Hyperparameters
Key hyperparameters can be modified in `main.py`:
- `alpha`: Learning rate for actor networks
- `beta`: Learning rate for critic networks
- `gamma`: Discount factor
- `tau`: Soft update parameter for target networks
- `horizon`: Number of steps per episode
- `n_agents`: Number of agents (2 for overcooked environment)
- `n_actions`: Number of possible actions (6 for overcooked enviornment)
