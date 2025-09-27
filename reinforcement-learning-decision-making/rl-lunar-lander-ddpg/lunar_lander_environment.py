from ddpg_agent import Agent
import gymnasium as gym
import numpy as np
import os
from ddpg_agent import OUActionNoise
from utils import plotLearning, plot_consecutive_rewards, plot_hyperparameter_effects, plot_action_distribution, plot_trajectory

# Define the directory where you want to save the plots
save_dir = r'C:\GaTech\Fall Semester 2024\Reinforcement Learning (CS-7642)\project_2'

# Ensure the directory exists (if not, it creates it)
os.makedirs(save_dir, exist_ok=True)

# Initialize environment
env = gym.make('LunarLanderContinuous-v2')

# Initialize agent
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

# Track the score history during training
score_history = []
# Track actions and trajectory data for plotting
actions = []
trajectory = []

# Training loop for 5000 episodes
for i in range(5000):
    done = False
    score = 0
    obs, _ = env.reset()
    print(f"Episode {i} - Initial Observation Shape: {obs.shape}")
    
    # Reset action and trajectory tracking for the episode
    episode_actions = []
    episode_trajectory = []

    while not done:
        act = agent.choose_action(obs)
        episode_actions.append(act)  # Track the actions
        episode_trajectory.append(obs[:2])  # Track the x, y position (trajectory)
        
        step_result = env.step(act)
        if len(step_result) == 5:
            new_state, reward, done, truncated, info = env.step(act)
            done = done or truncated
        else:
            new_state, reward, done, info = step_result

        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
    
    score_history.append(score)
    actions.extend(episode_actions)  # Append episode actions to the full list
    trajectory.extend(episode_trajectory)  # Append episode trajectory to the full list
    
    print('episode', i, 'score %.2f' % score,
          '100 game average %.2f' % np.mean(score_history[-100:]))
    
    if i % 25 == 0:
        agent.save_models()

# Save training plot (reward for each training episode)
filename = os.path.join(save_dir, 'training_rewards.png')
plotLearning(score_history, filename, window=100)

# Check if actions and trajectory are populated
print(f"Number of actions collected: {len(actions)}")
print(f"Number of trajectory points collected: {len(trajectory)}")

# Save action distribution plot
if actions:
    action_filename = os.path.join(save_dir, 'action_distribution.png')
    print("Saving action distribution plot...")
    plot_action_distribution(actions, action_filename)
else:
    print("No actions were collected, skipping action distribution plot.")

# Save trajectory plot
if trajectory:
    trajectory_filename = os.path.join(save_dir, 'lander_trajectory.png')
    print("Saving lander trajectory plot...")
    plot_trajectory(trajectory, trajectory_filename)
else:
    print("No trajectory data collected, skipping trajectory plot.")

# Run 100 consecutive episodes with the trained agent (without learning)
consecutive_rewards = []
for i in range(100):
    done = False
    score = 0
    obs, _ = env.reset()

    while not done:
        act = agent.choose_action(obs)  # Use the trained agent
        step_result = env.step(act)
        if len(step_result) == 5:
            new_state, reward, done, truncated, info = env.step(act)
            done = done or truncated
        else:
            new_state, reward, done, info = step_result

        score += reward
        obs = new_state
    
    consecutive_rewards.append(score)
    print('Test episode', i, 'score %.2f' % score)

# Save plot for 100 consecutive episodes
consecutive_filename = os.path.join(save_dir, 'consecutive_rewards.png')
print("Saving consecutive rewards plot...")
plot_consecutive_rewards(consecutive_rewards, consecutive_filename)

# Hyperparameter tuning - Varying the learning rate (alpha)
alpha_values = [0.0001, 0.000025, 0.00001]
hyperparam_results = {}

for alpha in alpha_values:
    agent = Agent(alpha=alpha, beta=0.00025, input_dims=[8], tau=0.001, env=env,
                  batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)
    
    np.random.seed(0)
    score_history = []
    
    for i in range(500):  # Shorter training time for testing hyperparameters
        done = False
        score = 0
        obs, _ = env.reset()

        while not done:
            act = agent.choose_action(obs)
            step_result = env.step(act)
            if len(step_result) == 5:
                new_state, reward, done, truncated, info = env.step(act)
                done = done or truncated
            else:
                new_state, reward, done, info = step_result

            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)
    
    hyperparam_results[alpha] = score_history

# Save plot for the effect of varying alpha (learning rate)
hyperparam_filename = os.path.join(save_dir, 'alpha_effect.png')
print("Saving hyperparameter effect plot for alpha values...")
plot_hyperparameter_effects(hyperparam_results, 'Alpha (Learning Rate)', hyperparam_filename)

# Hyperparameter tuning - Varying the discount factor (gamma)
gamma_values = [0.9, 0.95, 0.99]
hyperparam_results_gamma = {}

for gamma in gamma_values:
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
                  batch_size=64, layer1_size=400, layer2_size=300, n_actions=2, gamma=gamma)

    np.random.seed(0)
    score_history = []

    for i in range(500):  # Shorter training time for testing hyperparameters
        done = False
        score = 0
        obs, _ = env.reset()

        while not done:
            act = agent.choose_action(obs)
            step_result = env.step(act)
            if len(step_result) == 5:
                new_state, reward, done, truncated, info = env.step(act)
                done = done or truncated
            else:
                new_state, reward, done, info = step_result

            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)

    hyperparam_results_gamma[gamma] = score_history

# Save plot for the effect of varying gamma
gamma_filename = os.path.join(save_dir, 'gamma_effect.png')
print("Saving hyperparameter effect plot for gamma values...")
plot_hyperparameter_effects(hyperparam_results_gamma, 'Gamma (Discount Factor)', gamma_filename)

# Hyperparameter tuning - Varying the noise scale
noise_values = [0.2, 0.3, 0.4]
hyperparam_results_noise = {}

for noise_scale in noise_values:
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
                  batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)
    agent.noise = OUActionNoise(mu=np.zeros(2), sigma=noise_scale, sigma_decay=0.995, sigma_min=0.05)

    np.random.seed(0)
    score_history = []

    for i in range(500):  # Shorter training time for testing hyperparameters
        done = False
        score = 0
        obs, _ = env.reset()

        while not done:
            act = agent.choose_action(obs)
            step_result = env.step(act)
            if len(step_result) == 5:
                new_state, reward, done, truncated, info = env.step(act)
                done = done or truncated
            else:
                new_state, reward, done, info = step_result

            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)

    hyperparam_results_noise[noise_scale] = score_history

# Save plot for the effect of varying noise scale
noise_filename = os.path.join(save_dir, 'noise_scale_effect.png')
print("Saving hyperparameter effect plot for noise scale values...")
plot_hyperparameter_effects(hyperparam_results_noise, 'Noise Scale', noise_filename)