# Imports
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import numpy as np
import os
from PIL import Image

### Environment Setup ###
def setup_environment(layouts=["cramped_room", "asymmetric_advantages", "forced_coordination"], 
                    horizon=400, reward_shaping=None):
    """
    Sets up the Overcooked environments for the specified layouts, with a fixed episode horizon 
    and customizable reward shaping.
    
    Parameters:
        layouts (list): List of layout names to initialize (default includes all required layouts).
        horizon (int): Maximum steps per episode.
        reward_shaping (dict): Optional reward shaping parameters to encourage specific behaviors.
    
    Returns:
        environments (dict): Dictionary with layout names as keys and (environment, mdp) pairs as values.
    """
    if reward_shaping is None:
        # Default reward shaping encouraging specific tasks for multi-agent coordination
        reward_shaping = {
            "PLACEMENT_IN_POT_REW": 3,      # Reward for placing onions in pots
            "DISH_PICKUP_REWARD": 3,        # Reward for picking up a dish
            "SOUP_PICKUP_REWARD": 5         # Reward for delivering soup
        }

    environments = {}
    for layout in layouts:
        try:
            # Initialize MDP and environment for each layout
            mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
            base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
            environments[layout] = (base_env, mdp)
            print(f"Environment successfully created for layout '{layout}' with horizon {horizon}")
        except Exception as e:
            print(f"Failed to create environment for layout '{layout}': {e}")
    
    return environments

def visualize_state(env, state):
    """
    Visualizes the current state of the Overcooked environment for debugging or analysis.
    
    Parameters:
        env (OvercookedEnv): The Overcooked environment instance.
        state (OvercookedState): The current state to visualize.
    """
    try:
        img = StateVisualizer().render_from_layout_and_state(env.mdp, state)
        img.show()
    except Exception as e:
        print(f"Failed to visualize state: {e}")