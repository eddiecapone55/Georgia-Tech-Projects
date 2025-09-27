import sys
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
from robot_simulator import RobotSim
from robot_simulator import RobotRenderer


class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, n_rows=10, n_cols=10, warehouse_workers=None, warehouse_equipment=None, start=(0, 0), goal=(8, 2), rewards=None, prob_success=1.0, max_steps=1000, renderer=None):
        """
        Initializes the RobotEnv environment.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.warehouse_workers = warehouse_workers or [(2, 6), (7, 6)]
        self.warehouse_equipment = warehouse_equipment or [(4, 2), (4, 3), (4, 4), (4, 5)]
        self.start = start
        self.goal = goal
        self.rewards = rewards or {
            "wall": -1, 
            "equipment": -2, 
            "worker": -5, 
            "target": 10, 
            "move_closer": 0.5, 
            "move_further": -0.1
        }
        self.prob_success = prob_success
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        n_positions = n_rows * n_cols
        self.observation_space = spaces.Dict({
            "robot_position": spaces.Discrete(n_positions),
            "bumped_status": spaces.Discrete(5),  # bump statuses ("", "wall", "equipment", "worker", "target")
            "curr_target": spaces.Discrete(n_positions),
        })

        self.prob_fail = (1.0 - self.prob_success) / (self.action_space.n - 1)
        self.sim = RobotSim(n_rows, n_cols, self.warehouse_workers, self.warehouse_equipment, self.goal, self.start)
        self.s = self.sim.get_world_state()
        self.renderer = None

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        """
        Resets the environment to the initial state.
        """
        self.sim.reset(self.n_rows, self.n_cols, self.warehouse_workers, self.warehouse_equipment, self.goal, self.start)
        self.current_step = 0
        self.renderer = RobotRenderer(self.sim)
        self.s = self.sim.get_world_state()

        if return_info:
            return self.encode_observation(self.s), {}
        return self.encode_observation(self.s)

    def sample_action(self, action):
        """
        Samples an action based on transition probabilities.
        """
        prob = np.random.rand()
        if prob < self.prob_success:
            return action
        else:
            failed_actions = [a for a in range(self.action_space.n) if a != action]
            return np.random.choice(failed_actions)

    def step(self, action):
        """
        Takes a step in the environment based on the action.
        """
        action = self.sample_action(action)
        bumped = self.sim.advance(action)  # Simulate the action and get bumped status
        self.s = self.sim.get_world_state()

        reward = self.get_rewards(bumped)
        done = self.is_terminal(bumped)
        truncated = self.is_truncated()

        self.current_step += 1

        # Update worker positions randomly (simulating human movement)
        self._move_workers_randomly()

        return self.encode_observation(self.s), reward, done, truncated, {"bumped": bumped}

    def _move_workers_randomly(self):
        """
        Move warehouse workers randomly within the grid to simulate dynamic obstacles.
        """
        for idx, (row, col) in enumerate(self.warehouse_workers):
            # Random movement: up, down, left, or right
            move = np.random.choice(["up", "down", "left", "right"])
            if move == "up" and row > 0:
                row -= 1
            elif move == "down" and row < self.n_rows - 1:
                row += 1
            elif move == "left" and col > 0:
                col -= 1
            elif move == "right" and col < self.n_cols - 1:
                col += 1
            self.warehouse_workers[idx] = (row, col)

    def get_rewards(self, bumped):
        """
        Calculates the reward for the current state.
        """
        if bumped == "wall":
            return self.rewards["wall"]
        elif bumped == "equipment":
            return self.rewards["equipment"]
        elif bumped == "worker":
            return self.rewards["worker"]
        elif bumped == "target":
            return self.rewards["target"]

        # Get the current position of the robot from the simulator
        current_position = np.array([self.sim.row, self.sim.col])
        target_position = np.array(self.sim.curr_target)

        # Calculate the distance to the target
        prev_distance = np.linalg.norm(current_position - target_position)
        
        # After the move, calculate the new distance to the target
        new_position = np.array([self.sim.row, self.sim.col])
        new_distance = np.linalg.norm(new_position - target_position)

        # Reward if the robot is getting closer to the target
        if new_distance < prev_distance:
            return self.rewards["move_closer"]
        else:
            return self.rewards["move_further"]

    def is_truncated(self):
        """
        Checks if the episode has reached the maximum number of steps.
        """
        return self.current_step >= self.max_steps

    def is_terminal(self, bumped):
        """
        Checks if the current state is terminal (reaching the target).
        """
        return bumped == "target"

    def render(self, close=False):
        """
        Render the environment.
        """
        if close and self.renderer:
            if self.renderer:
                self.renderer.close()
            return

        if self.renderer:
            return self.renderer.render()

    def encode_observation(self, world_state):
        """
        Encodes the world state using linear indices.
        """
        row, col, bumped, n_boxes, prev_target, curr_target, workers_tuplist = world_state
        robot_position = np.array([row * self.n_cols + col])
        bump_mapping = {"": 0, "wall": 1, "equipment": 2, "worker": 3, "target": 4}
        bumped_encoded = np.array([bump_mapping[bumped]])
        curr_target_encoded = np.array([curr_target[0] * self.n_cols + curr_target[1]])
        
        encoded_state = np.concatenate([
            robot_position,
            bumped_encoded,
            curr_target_encoded,
        ])

        return encoded_state