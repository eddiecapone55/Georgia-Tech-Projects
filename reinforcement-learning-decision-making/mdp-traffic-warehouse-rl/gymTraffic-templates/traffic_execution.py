import time
import gymnasium as gym
from traffic_environment import TrafficEnv
import rl_planners
import rl_agents
import matplotlib.pyplot as plt
import csv
import os

# Define baseline traffic flow scenarios
scenarios = [
    {"lambda_ns": 3, "lambda_ew": 3, "description": "Uniform Traffic Flow"},
    {"lambda_ns": 5, "lambda_ew": 2, "description": "Unequal Traffic Flow (NS-heavy)"},
    {"lambda_ns": 1, "lambda_ew": 6, "description": "Unequal Traffic Flow (EW-heavy)"},
    {"lambda_ns": 6, "lambda_ew": 6, "description": "Heavy Traffic"},
    {"lambda_ns": 1, "lambda_ew": 1, "description": "Sparse Traffic"},
]

# Set light state variables 
RED, GREEN = 0, 1

# Initialize master CSV file path for all trials and agents
master_csv_file = r'C:\GaTech\Fall Semester 2024\Reinforcement Learning (CS-7642)\Project 1\gymTraffic-templates\Trial Data\all_agents_all_trials_results.csv'
if not os.path.exists(master_csv_file):
    with open(master_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header row
        writer.writerow(['Agent', 'Scenario', 'Trial', 'Step', 'Reward', 'Cumulative Discounted Reward', 'Steps to Converge'])

# Function to log results to the master CSV file
def log_results(agent_name, scenario_desc, trial_num, steps, rewards_list, cumulative_rewards_list, steps_to_converge):
    with open(master_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for step, reward, cum_reward in zip(steps, rewards_list, cumulative_rewards_list):
            writer.writerow([agent_name, scenario_desc, trial_num, step, reward, cum_reward])
        if steps_to_convergence:
            writer.writerow([agent_name, scenario_desc, trial_num, 'Steps to converge:', steps_to_converge])
        print(f"Results for {agent_name} - {scenario_desc} (Trial {trial_num}) logged in {master_csv_file}")

# Function to plot and save the results after a trial
def plot_results(trial_num, steps, rewards_list, cumulative_rewards_list, method_name, scenario_desc):
    plt.figure(figsize=(10,6))

    # Plot both rewards and cumulative discounted rewards
    plt.plot(steps, rewards_list, label="Rewards")
    plt.plot(steps, cumulative_rewards_list, label="Cumulative Discounted Rewards", linestyle='--')
    
    plt.title(f'Performance of {method_name} ({scenario_desc}) - Trial {trial_num}')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.legend()

    # Save plot in the "Graphs" folder
    graph_path = r'C:\GaTech\Fall Semester 2024\Reinforcement Learning (CS-7642)\Project 1\gymTraffic-templates\Graphs'
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    plot_filename = os.path.join(graph_path, f'{method_name}_trial_{trial_num}_{scenario_desc}_performance.png')
    plt.savefig(plot_filename)
    plt.close() 
    print(f"Plot saved as {plot_filename}")

# List of agents to test
agents = [
    {"name": "Value Iteration Planner", "instance": rl_planners.ValueIterationPlanner},
    {"name": "Policy Iteration Planner", "instance": rl_planners.PolicyIterationPlanner},
    #{"name": "Q-Learning Agent", "instance": rl_agents.QLearningAgent},
    
    #{"name": "SARSA Agent", "instance": rl_agents.SARSAgent}
]

# Loop through scenarios and run experiments
for scenario in scenarios:
    print(f"Running experiment for: {scenario['description']}")

    for agent_info in agents:
        agent_name = agent_info["name"]
        agent_class = agent_info["instance"]

        # Create the environment with the current scenario's parameters
        env = TrafficEnv(
            max_cars_dir=20, max_cars_total=30, 
            lambda_ns=scenario['lambda_ns'], lambda_ew=scenario['lambda_ew'],
            cars_leaving=5, rewards={"state": 0}, max_steps=1000
        )

        # Initialize the agent
        agent = agent_class(env)

        # Only run 1 trial for the current agent and scenario
        trial = 1

        # Reset the environment and get the initial observation
        observation, info = env.reset(seed=42), {}
        steps = []
        rewards_list = []
        terminated, truncated = False, False
        counter = 0

        discount_factor = 0.9  # arbitrary

        # Initialize list to store cumulative discounted rewards
        cumulative_rewards_list = []

        # Initialize cumulative rewards for this episode
        cumulative_discounted_reward = 0

        # Track steps to convergence
        convergence_threshold = 100  # adjust based on environment or desired threshold
        steps_to_convergence = None  # stores when the agent hits the threshold

        # Run environment until terminated or truncated
        while not terminated and not truncated:
            # Use the agent's policy to choose an action
            action = agent.choose_action(observation)
            # Step through the environment with the chosen action
            observation, reward, terminated, truncated, info = env.step(action)

            # Discounted reward calculation
            cumulative_discounted_reward += (discount_factor ** counter) * reward

            # Store cumulative discounted reward
            cumulative_rewards_list.append(cumulative_discounted_reward)

            if cumulative_discounted_reward >= convergence_threshold and steps_to_convergence is None:
                steps_to_convergence = counter
                print(f"Agent converged after {steps_to_convergence} steps.")

            # Unpack the state to get the number of cars and traffic light state
            ns, ew, light = tuple(observation)
            light_color = "GREEN" if light == GREEN else "RED"
            # Print the current state
            print(f"Step: {counter}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            # Render the environment at each step
            env.render()

            # Store results for plotting
            steps.append(counter)
            rewards_list.append(reward)

            # Update counter
            counter += 1

        # Log results for this trial to the master CSV file
        log_results(agent_name, scenario['description'], trial, steps, rewards_list, cumulative_rewards_list, steps_to_convergence)
        
        # Plot the results for this trial and save it in the "Graphs" folder
        plot_results(trial, steps, rewards_list, cumulative_rewards_list, agent_name, scenario['description'])

    # Close the environment after all trials for this agent and scenario
    env.render(close=True)
