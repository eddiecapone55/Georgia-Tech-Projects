import matplotlib.pyplot as plt
import numpy as np

# PlotLearning function to handle different types of plots
def plotLearning(scores, filename, title="Training Progress", x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    
    if x is None:
        x = [i for i in range(N)]
        
    plt.figure()
    plt.ylabel('Score')       
    plt.xlabel('Episode')                     
    plt.plot(x, running_avg)
    plt.title(title)
    plt.savefig(filename)
    print(f"Saved {filename}")  # Add this print statement
    plt.close()

# Function to plot the rewards for 100 consecutive episodes using the trained agent
def plot_consecutive_rewards(rewards, filename, title="Reward per Episode for 100 Consecutive Episodes"):
    plt.figure()
    plt.plot(range(len(rewards)), rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title(title)
    plt.savefig(filename)
    print(f"Saved {filename}")  # Add this print statement
    plt.close()

# Function to compare the effect of hyperparameters (learning rate, gamma, noise)
def plot_hyperparameter_effects(hyperparameter_results, hyperparameter_name, filename):
    plt.figure()
    for param_value, scores in hyperparameter_results.items():
        running_avg = np.empty(len(scores))
        for t in range(len(scores)):
            running_avg[t] = np.mean(scores[max(0, t-5):(t+1)])
        plt.plot(range(len(scores)), running_avg, label=f'{hyperparameter_name} = {param_value}')
    
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title(f'Effect of {hyperparameter_name} on Performance')
    plt.legend()
    plt.savefig(filename)
    print(f"Saved {filename}")  # Add this print statement
    plt.close()

# Function to plot action distribution
def plot_action_distribution(actions, filename, downsample_factor=10):
    # Downsample the actions for faster plotting
    actions = actions[::downsample_factor]
    actions = np.array(actions)
    
    plt.figure()
    plt.hist(actions, bins=50, density=True)
    plt.title("Action Distribution")
    plt.savefig(filename)
    print(f"Saved {filename}")  # Add this print statement
    plt.close()

# Function to plot lander trajectory
def plot_trajectory(trajectory, filename, downsample_factor=10):
    # Downsample the trajectory for faster plotting
    trajectory = trajectory[::downsample_factor]
    trajectory = np.array(trajectory)

    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], '.')
    plt.title("Lander Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.savefig(filename)
    print(f"Saved {filename}")  # Add this print statement
    plt.close()