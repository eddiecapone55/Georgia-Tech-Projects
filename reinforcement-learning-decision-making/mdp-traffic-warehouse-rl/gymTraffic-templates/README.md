# Traffic Intersection Control - Reinforcement Learning Project

## Introduction

This project simulates traffic flow at a four-way intersection and uses reinforcement learning to optimize the control of traffic lights. The primary goal is to minimize congestion by dynamically managing traffic lights for North-South and East-West directions. The project explores different reinforcement learning techniques, including Value Iteration, Policy Iteration, Q-Learning, and SARSA, applied to various traffic flow scenarios.

## Requirements

### Dependencies

The following Python libraries are required to run the project:

- `numpy`
- `pygame`
- `scipy`
- `matplotlib`
- `gymnasium`

To install all required dependencies, run:

```bash
pip install numpy pygame scipy matplotlib gymnasium
```

### Installation

1. Clone or download this repository to your local machine.
2. Ensure that all dependencies are installed using the pip command provided above.
3. Navigate to the project directory in your terminal or preferred IDE.

## How to Run the Traffic Intersection Problem

### Running the Simulation

1. **Traffic Simulation**:
   The traffic simulation is defined in `traffic_simulator.py` and rendered using `Pygame`. The simulation models the flow of cars arriving at the intersection, with the goal of optimizing the traffic lights to minimize wait times.

2. **Environment**:
   The environment for this simulation is defined in `traffic_environment.py` using the `gymnasium` framework. It handles state transitions, rewards, and interactions with the agent.

3. **Execution Script**:
   The main script to run the experiments is `traffic_execution.py`, which simulates traffic flow for different agents under various traffic conditions. The following agents are implemented:
   - **Value Iteration Planner** (from `rl_planners.py`)
   - **Policy Iteration Planner** (from `rl_planners.py`)
   - **Q-Learning Agent** (from `rl_agents.py`)
   - **SARSA Agent** (from `rl_agents.py`)

To run the experiment, execute the following command:
```bash
python traffic_execution.py
```

### Traffic Scenarios

The script runs experiments for different traffic scenarios defined in `traffic_execution.py`:
- **Uniform Traffic Flow**: Equal number of cars arriving from both North-South and East-West directions.
- **Unequal Traffic Flow (NS-heavy)**: More cars arriving from the North-South direction.
- **Unequal Traffic Flow (EW-heavy)**: More cars arriving from the East-West direction.
- **Heavy Traffic**: Large volumes of cars arriving from both directions.
- **Sparse Traffic**: Minimal traffic from both directions.

These scenarios are customizable by modifying the `scenarios` list in `traffic_execution.py`.

## File Descriptions

- **traffic_simulator.py**: Contains the core traffic simulation logic and rendering. This includes updating the number of cars waiting and handling the traffic light transitions.
  
- **traffic_environment.py**: Defines the environment using the `gymnasium` interface. This includes the MDP formulation (states, actions, rewards) and step-by-step interactions with the agents.

- **traffic_execution.py**: Main script to run traffic flow simulations for different agents and scenarios. Results are logged and plotted for performance analysis.

- **rl_planners.py**: Contains the implementation of Value Iteration and Policy Iteration algorithms, which compute optimal policies based on the environment's MDP.

- **rl_agents.py**: Contains the implementation of Q-Learning and SARSA agents, which learn policies through interaction with the environment.

## Results and Logs

### Logging Results

The results from each trial are logged into a CSV file. The default file path is - feel free to change it to your own:
```
C:\GaTech\Fall Semester 2024\Reinforcement Learning (CS-7642)\Project 1\gymTraffic-templates\Trial Data\all_agents_all_trials_results.csv
```

Each entry in the CSV file includes:
- The agent used (Value Iteration, Policy Iteration, Q-Learning, SARSA)
- The traffic scenario (e.g., uniform traffic)
- Rewards per step
- Cumulative discounted rewards
- Steps taken until convergence (if applicable)

### Performance Plots

After each trial, the performance results (rewards and cumulative discounted rewards) are plotted and saved as images in the `Graphs` folder. These plots visualize the performance of each agent in the traffic simulation.

## How to Customize

### Modify Traffic Scenarios

To add or modify traffic scenarios, update the `scenarios` list in `traffic_execution.py`. Each scenario is a dictionary that defines traffic parameters (e.g., `lambda_ns` for North-South traffic, `lambda_ew` for East-West traffic).

### Adjust Agent Behavior

You can modify the agent behavior by adjusting the hyperparameters in `traffic_execution.py` or in the respective agent classes (`rl_planners.py` or `rl_agents.py`). For example:
- **Discount factor \( \gamma \)**: Adjust how much the agent values future rewards.
- **Exploration rate \( \epsilon \)**: For Q-Learning and SARSA, adjust the exploration vs. exploitation trade-off.
- **Learning rate \( alpha \)**: Modify how quickly Q-Learning and SARSA agents update their Q-values.

## Known Issues

- **Rendering Performance**: The Pygame window may experience slower rendering on some systems. You can adjust the window size or rendering mode (`human` or `rgb_array`) in `traffic_simulator.py` to improve performance.

## Conclusion

This project applies reinforcement learning techniques to optimize traffic flow at an intersection. Various algorithms, including Value Iteration, Policy Iteration, Q-Learning, and SARSA, are tested under different traffic scenarios. The results, including performance logs and plots, help evaluate each agent's ability to minimize traffic congestion through learned policies.