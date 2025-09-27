# Lunar Lander - Reinforcement Learning Project

## Overview
This project implements a DDPG (Deep Deterministic Policy Gradient) agent to solve the **Lunar Lander Continuous** problem (`LunarLanderContinuous-v2`) in OpenAI Gym. The agent learns to control the lunar lander using continuous action spaces, where the actions represent the throttle of the main and side engines.

The code consists of several components, including the agent, environment setup, replay buffer, and utility functions for plotting. The agent uses PyTorch for neural network training and optimization.

## Requirements

To run the project, ensure that you have the following dependencies installed:

- Python 3.7+
- PyTorch
- Gymnasium (OpenAI Gym)
- NumPy
- Matplotlib

## Project Structure

The project contains the following Python files:

### 1. **ddpg_agent.py**
   This file defines the agent's architecture and the components used for training the DDPG algorithm. It contains the following key classes:

   - **OUActionNoise**: Implements the Ornstein-Uhlenbeck noise process for exploration in continuous action spaces.
   - **ReplayBuffer**: A memory buffer that stores and samples transitions (state, action, reward, next state) for the agent's training.
   - **CriticNetwork**: Defines the critic network architecture. The critic evaluates the Q-value for a given state-action pair.
   - **ActorNetwork**: Defines the actor network architecture. The actor outputs the optimal continuous actions given the state.
   - **Agent**: The DDPG agent, responsible for interacting with the environment, storing experiences in the replay buffer, and updating the actor and critic networks.

### 2. **lunar_lander_environment.py**
   This file sets up the Lunar Lander environment and trains the DDPG agent. The training loop performs the following tasks:
   
   - Initializes the `LunarLanderContinuous-v2` environment.
   - Runs the training loop for a specified number of episodes (e.g., 1000 episodes).
   - Chooses actions using the agent, performs the actions in the environment, and stores the results in the replay buffer.
   - The agent learns from the stored experiences and updates its networks.
   - The average score of the agent over time is printed and stored.

   The agent's model is saved every 25 episodes.

### 3. **utils.py**
   This file contains utility functions used throughout the project. It includes:

   - **plotLearning**: A function that plots the running average of the agent’s scores over episodes and saves the plot as a PNG image.

### 4. **Plotting Output**
   The `plotLearning` function in `utils.py` will generate a plot showing the running average of the rewards obtained by the agent during training. The image will be saved as `lunar-lander.png`.

## Running the Project

To run the project and train the agent, execute the following command:

\`\`\`bash
python lunar_lander_environment.py
\`\`\`

This will start the training process, and the agent will interact with the environment for 1000 episodes (or as configured). The results, including the running average score, will be printed to the console, and the trained models will be saved every 25 episodes.

The training rewards will be plotted and saved as `lunar-lander.png` once training is complete.

## Model Saving and Loading

The agent saves the trained models periodically. If you want to load previously saved models, you can uncomment the `load_models` method in the `lunar_lander_environment.py` file:

\`\`\`python
agent.load_models()  # Uncomment this line to load models
\`\`\`

This will load the actor and critic networks from the saved checkpoints and continue training from where you left off.

## Hyperparameters

The following hyperparameters can be modified in the `lunar_lander_environment.py` file to experiment with different settings:

- `alpha`: Learning rate for the actor network.
- `beta`: Learning rate for the critic network.
- `gamma`: Discount factor for the agent's future reward calculations.
- `tau`: Soft update parameter for updating the target networks.
- `batch_size`: Number of experiences sampled from the replay buffer at each learning step.

You can experiment with these parameters to evaluate their impact on the agent’s performance.
