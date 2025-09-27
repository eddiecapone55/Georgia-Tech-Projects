import time
import gymnasium as gym
from robot_environment import RobotEnv
import rl_agents
import matplotlib.pyplot as plt
import csv
import os

# Base directory for saving data
base_dir = r'C:\GaTech\Fall Semester 2024\Reinforcement Learning (CS-7642)\Project 1\gymRobot-templates'

# Define dynamic paths for saving data
def create_agent_folders(agent_name):
    graphs_folder = os.path.join(base_dir, f'Graphs_{agent_name}')
    csv_folder = os.path.join(base_dir, f'Trial_Data_{agent_name}')
    os.makedirs(graphs_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    return graphs_folder, csv_folder

# CSV file for storing all trial data
def create_csv_file(csv_folder, agent_name):
    csv_file_path = os.path.join(csv_folder, f'{agent_name}_trials_data.csv')
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Agent', 'Trial', 'Step', 'Reward', 'Cumulative Reward', 'Steps to Converge'])
    return csv_file_path

# Define function to log data to CSV
def log_data_to_csv(agent_name, trial, steps, rewards, cumulative_rewards, steps_to_converge, csv_file_path):
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for step, reward, cum_reward in zip(steps, rewards, cumulative_rewards):
            writer.writerow([agent_name, trial, step, reward, cum_reward, steps_to_converge])

# Function to plot and save graphs
def plot_results(trial, steps, rewards, cumulative_rewards, agent_name, graphs_folder):
    plt.figure(figsize=(10, 6))
    
    # Plot rewards and cumulative rewards
    plt.plot(steps, rewards, label="Rewards")
    plt.plot(steps, cumulative_rewards, label="Cumulative Rewards", linestyle='--')

    plt.title(f'{agent_name} - Trial {trial}')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.legend()

    # Save the plot
    plot_filename = f'{agent_name}_trial_{trial}_performance.png'
    plot_path = os.path.join(graphs_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved: {plot_path}")

# List of agents to test
agents = [
    {"name": "QLearningAgent", "instance": rl_agents.QLearningAgent},
    {"name": "SARSAgent", "instance": rl_agents.SARSAgent}
]

# Initialize environment
# Initialize environment with the proper rewards dictionary
rewards = {
    "wall": -10,
    "equipment": -20,
    "worker": -50,
    "target": 100,
    "move_closer": 5,  # Positive reward for getting closer to the goal
    "move_further": -1  # Small penalty for moving further from the goal
}
env = RobotEnv(rewards=rewards, max_steps=1000)

# Number of trials (episodes) for each agent
num_trials = 10
early_stopping_threshold = 100  # Stop training early if the agent converges in fewer than this many steps

# Loop through the agents and run experiments
for agent_info in agents:
    agent_name = agent_info["name"]
    agent_class = agent_info["instance"]

    # Create folders for this agent
    graphs_folder, csv_folder = create_agent_folders(agent_name)
    csv_file_path = create_csv_file(csv_folder, agent_name)

    # Initialize the agent
    agent = agent_class(env, gamma=0.9, alpha=0.1, epsilon=1.0, epsilon_decay=0.999, episodes=num_trials)

    # Load Q-table if needed (for resuming training)
    qtable_file = os.path.join(csv_folder, f'{agent_name}_qtable.npy')
    if os.path.exists(qtable_file):
        print(f"Loading Q-table for {agent_name}...")
        agent.load_qtable(qtable_file)

    # Run multiple trials for the agent
    for trial in range(1, num_trials + 1):
        observation = env.reset(seed=42)  # Reset returns the initial observation
        terminated, truncated = False, False
        steps = []
        rewards_list = []
        cumulative_rewards_list = []
        counter = 0
        cumulative_reward = 0
        convergence_threshold = 50
        steps_to_converge = None

        while not terminated and not truncated:
            # Agent chooses an action
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            # Update cumulative reward
            cumulative_reward += reward
            cumulative_rewards_list.append(cumulative_reward)

            # Track steps and rewards
            steps.append(counter)
            rewards_list.append(reward)

            # Check convergence
            if cumulative_reward >= convergence_threshold and steps_to_converge is None:
                steps_to_converge = counter
                print(f"Agent converged after {steps_to_converge} steps in trial {trial}.")

            # Print the current state for debugging purposes
            print(f"Trial: {trial}, Step: {counter}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            env.render()

            # Add a small delay for visualization
            time.sleep(0.1)

            counter += 1

            # Early stopping if the agent converges quickly
            if steps_to_converge is not None and steps_to_converge <= early_stopping_threshold:
                print(f"Early stopping: agent converged in {steps_to_converge} steps (threshold: {early_stopping_threshold})")
                break

            # If episode ends, break and continue next trial
            if terminated or truncated:
                print("\nEpisode ended, moving to next trial...\n")
                break

        # Log trial data to CSV
        log_data_to_csv(agent_name=agent_name, trial=trial, steps=steps, rewards=rewards_list, cumulative_rewards=cumulative_rewards_list, steps_to_converge=steps_to_converge, csv_file_path=csv_file_path)

        # Plot and save the results
        plot_results(trial=trial, steps=steps, rewards=rewards_list, cumulative_rewards=cumulative_rewards_list, agent_name=agent_name, graphs_folder=graphs_folder)

        # Decay epsilon (exploration factor) after each trial
        agent.epsilon *= agent.epsilon_decay

        # Save Q-table after each trial (for resuming or future analysis)
        agent.save_qtable(qtable_file)

# Close the environment
env.render(close=True)