# Warehouse Robot Navigation - Reinforcement Learning Project

## Introduction

This project simulates a warehouse robot tasked with navigating a grid-based environment to deliver boxes while avoiding obstacles such as workers and equipment. The robot uses reinforcement learning (Q-Learning and SARSA) to autonomously optimize its path and maximize successful deliveries. The robot must adapt to the probabilistic nature of its actions and the dynamic movement of warehouse workers.

## Requirements

### Dependencies

The following Python libraries are required to run the project:

- `numpy`
- `pygame`
- `gymnasium`
- `matplotlib`

To install all required dependencies, run:

```bash
pip install numpy pygame gymnasium matplotlib
```

### Installation

1. Clone or download this repository to your local machine.
2. Install the dependencies using the pip command provided above.
3. Navigate to the project directory in your terminal or preferred IDE.

## How to Run the Warehouse Robot Problem

### Running the Simulation

1. **Warehouse Simulation**:
   The simulation is defined in `robot_simulator.py` and rendered using `Pygame`. The robot navigates a grid-based warehouse while avoiding obstacles and workers. The objective is to deliver boxes to a target location.

2. **Environment**:
   The environment is built using `gymnasium` and is defined in `robot_environment.py`. It handles state transitions, rewards, and interactions between the robot and dynamic warehouse elements such as workers.

3. **Execution Script**:
   The main execution script, `robot_execution.py`, is used to run the simulation with different agents (Q-Learning and SARSA). This script also handles logging the results of each trial and plotting performance metrics.

To run the experiment, execute the following command:
```bash
python robot_execution.py
```

### Warehouse Layout and Actions

- The warehouse is represented as a grid of dimensions `n_rows` x `n_cols` (default: 10x10).
- The robot starts at a designated position and must navigate toward a target location while avoiding collisions with workers and equipment.
- The robot has four possible actions:
  - **Up**
  - **Right**
  - **Down**
  - **Left**

Each action has a probabilistic chance of success (`prob_success`), allowing for stochasticity in the robot’s movements.

### Rewards

The reward structure is defined in `robot_environment.py` and can be customized. The default rewards are:
- **Wall Collision**: -10
- **Equipment Collision**: -20
- **Worker Collision**: -50
- **Successful Delivery (Target)**: +100
- **Moving Closer to Target**: +5
- **Moving Further from Target**: -1

These rewards encourage the robot to avoid obstacles while optimizing for efficient deliveries.

## File Descriptions

- **robot_simulator.py**: Contains the core simulation logic for the robot’s movement and interactions within the warehouse.
  
- **robot_environment.py**: Defines the warehouse environment using the `gymnasium` interface. It includes the state transitions, reward function, and handling of worker movement.

- **robot_renderer.py**: Renders the warehouse, robot, workers, equipment, and target using Pygame. This allows for a visual representation of the robot's navigation.

- **robot_execution.py**: The main script to run experiments with Q-Learning and SARSA agents. It handles logging results to CSV files, plotting performance metrics, and saving Q-tables for analysis.

- **rl_agent.py**: Contains the implementation of the Q-Learning and SARSA agents, which learn policies based on their interactions with the environment.

## Customization

### Modifying Warehouse Layout

You can modify the warehouse layout (number of rows, columns, workers, and equipment) by editing the parameters in `robot_environment.py`:
- `n_rows` and `n_cols`: Define the grid size of the warehouse.
- `warehouse_workers`: List of worker positions represented as `(row, col)` tuples.
- `warehouse_equipment`: List of equipment positions represented as `(row, col)` tuples.

### Adjusting Agent Behavior

To adjust the behavior of the robot, you can modify the parameters in `robot_execution.py` or `rl_agent.py`:
- **Reward structure**: Customize the rewards for different events (e.g., collisions, successful deliveries).
- **Action success probability**: Adjust the `prob_success` parameter in `robot_environment.py` to make the robot’s movements more or less reliable.
- **Hyperparameters**: Modify the Q-Learning and SARSA parameters (e.g., discount factor \( \gamma \), learning rate \( alpha \), exploration rate \( \epsilon \)) in `rl_agent.py`.

### Logging Results

The results of each trial are logged in CSV files. By default, the files are saved in my personal directory. Feel free to change it to your own:
```
C:\GaTech\Fall Semester 2024\Reinforcement Learning (CS-7642)\Project 1\gymRobot-templates
```

The data includes:
- Agent name (Q-Learning or SARSA)
- Trial number
- Steps taken
- Rewards per step
- Cumulative rewards
- Steps to convergence (if applicable)

### Performance Plots

Performance results are plotted and saved as images in the `Graphs` folder. The plots show:
- Rewards per step
- Cumulative rewards per trial

The graphs help visualize the performance of each agent in navigating the warehouse.

## Known Issues

- **Rendering Performance**: The Pygame window may experience slower rendering when resized or when running on some systems. Adjust the window size or mode (`human` or `rgb_array`) in `robot_renderer.py` to improve performance.

- **Worker Movement**: The workers move randomly within the warehouse. This movement may lead to more frequent collisions in tighter spaces. You can modify their movement logic in `robot_simulator.py`.

## Conclusion

This project applies reinforcement learning techniques to a warehouse robot navigation task. The robot uses Q-Learning and SARSA to learn policies for optimal navigation and efficient deliveries. The environment provides a challenging scenario with dynamic obstacles, encouraging the robot to balance exploration and exploitation to achieve its goal.