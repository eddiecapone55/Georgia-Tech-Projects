""""""  	  
"""  	  
Template for implementing QLearner  (c) 2015 Tucker Balch  	  
  	  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  	  
Atlanta, Georgia 30332  	  
All Rights Reserved  	  
  	  
Template code for CS 4646/7646  	  
  	  
Georgia Tech asserts copyright ownership of this template and all derivative  	  
works, including solutions to the projects assigned in this course. Students  	  
and other users of this template code are advised not to share it with others  	  
or to make it available on publicly viewable websites including repositories  	  
such as github and gitlab.  This copyright statement should not be removed  	  
or edited.  	  
  	  
We do grant permission to share solutions privately with non-students such  	  
as potential employers. However, sharing with other current or future  	  
students of CS 4646 is prohibited and subject to being investigated as a  	  
GT honor code violation.  	  
  	  
-----do not edit anything above this line---  	  
  	  
Student Name: Edward Capone  
GT User ID: ecapone3  
GT ID: 904057332  
"""

import random as rand
import matplotlib.pyplot as plt
import numpy as np

class QLearner(object):
    """
    This is a Q learner object.
    
    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available.
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step.
                Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr.
                 Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        # Parameters and initializations
        self.verbose = verbose
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        # Initialize Q-table to all zeros.
        self.Q = np.zeros((num_states, num_actions))

        # Keep track of the last state and last action taken.
        self.s = 0
        self.a = 0

        # For Dyna-Q: a list to store experience tuples (s, a, s_prime, r)
        self.experiences = []

        # Initialize a dictionary for counting state visits (for heatmap visualization)
        self.state_counts = {}

    def _update_state_count(self, state):
        """
        Helper method to update the state visit count.
        """
        if state in self.state_counts:
            self.state_counts[state] += 1
        else:
            self.state_counts[state] = 1

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table.
        
        :param s: The new state.
        :type s: int
        :return: The selected action.
        :rtype: int
        """
        self.s = s
        # Update state visit count.
        self._update_state_count(s)
        # Epsilon-greedy action selection (but do NOT update rar)
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = int(np.argmax(self.Q[s, :]))
        if self.verbose:
            print(f"querysetstate: s = {s}, a = {action}")
        # Update last action for consistency
        self.a = action
        return action

    def query(self, s_prime, r):
        """
        Update the Q table with the experience tuple and return the selected action.
        
        :param s_prime: The new state.
        :type s_prime: int
        :param r: The immediate reward.
        :type r: float
        :return: The selected action.
        :rtype: int
        """
        # Q-Learning update for the observed experience (s, a, s_prime, r)
        old_value = self.Q[self.s, self.a]
        best_next = np.max(self.Q[s_prime, :])
        self.Q[self.s, self.a] = old_value + self.alpha * (r + self.gamma * best_next - old_value)

        # For Dyna-Q: store the experience and perform hallucinated updates if enabled.
        if self.dyna > 0:
            # Store the actual experience
            self.experiences.append((self.s, self.a, s_prime, r))
            # Perform dyna updates by sampling random experiences.
            for _ in range(self.dyna):
                s_d, a_d, s_prime_d, r_d = self.experiences[rand.randint(0, len(self.experiences) - 1)]
                old_val_d = self.Q[s_d, a_d]
                best_next_d = np.max(self.Q[s_prime_d, :])
                self.Q[s_d, a_d] = old_val_d + self.alpha * (r_d + self.gamma * best_next_d - old_val_d)

        # Epsilon-greedy action selection for the next action.
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = int(np.argmax(self.Q[s_prime, :]))

        if self.verbose:
            print(f"query: s = {s_prime}, a = {action}, r = {r}")

        # Update state counts for the new state
        self._update_state_count(s_prime)

        # Update state and action
        self.s = s_prime
        self.a = action

        # Decay the random action rate.
        self.rar *= self.radr

        return action

    def plot_state_heatmap(self, grid_shape=None, filename="qlearner_heatmap.png"):
        """
        Plots a heatmap of state visit frequencies from the training phase.
        
        :param grid_shape: Optional tuple (rows, cols) defining the grid dimensions.
                           If None, the grid is computed as roughly square.
        :param filename: The filename where the heatmap image will be saved.
        """
        total_states = self.Q.shape[0]
        # If grid shape not provided, compute one.
        if grid_shape is None:
            rows = int(np.ceil(np.sqrt(total_states)))
            cols = int(np.ceil(total_states / rows))
            grid_shape = (rows, cols)
        
        # Create an array of state visit counts.
        counts = np.zeros(total_states)
        for state in range(total_states):
            counts[state] = self.state_counts.get(state, 0)
        
        # If grid does not perfectly fit total_states, pad the array with zeros.
        grid_size = grid_shape[0] * grid_shape[1]
        if grid_size > total_states:
            counts = np.concatenate([counts, np.zeros(grid_size - total_states)])
        
        counts_grid = counts.reshape(grid_shape)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(counts_grid, cmap="hot", interpolation="nearest")
        plt.title("State Visit Frequency Heatmap")
        plt.xlabel("Grid Column")
        plt.ylabel("Grid Row")
        plt.colorbar(label="Visits")
        plt.savefig(filename)
        plt.close()
    
    def author(self):
        """
        Returns the GT username of the student.
        
        :return: GT username.
        :rtype: str
        """
        return "ecapone3"

    def study_group(self):
        """
        Returns a comma separated string of GT usernames of each member of your study group.
        
        :return: Comma separated GT usernames.
        :rtype: str
        """
        return "ecapone3"

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
