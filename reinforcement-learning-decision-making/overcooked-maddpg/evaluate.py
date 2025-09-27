import numpy as np
from maddpg import MADDPG
from environment_setup import setup_environment
from plotting import plot_soups_delivered_evaluation

def evaluate_agent(horizon=400, n_games=100, render=False):
    """
    Evaluates the MADDPG agent across multiple layouts in the Overcooked environment.

    Parameters:
        horizon (int): Maximum steps per episode.
        n_games (int): Number of episodes to evaluate per layout.
        render (bool): If True, renders the environment for visual verification.
    """
    layouts = ["cramped_room", "asymmetric_advantages", "forced_coordination"]
    n_agents = 2
    n_actions = 6
    results = {}

    for layout in layouts:
        print(f"Evaluating layout: {layout}")

        # Initialize the Overcooked environment for the current layout
        base_env, _, _ = setup_environment(layout=layout, horizon=horizon)
        
        # Dynamic observation dimensions based on environment
        single_actor_dim = base_env.observation_space.shape[0]
        actor_dims = [single_actor_dim] * n_agents
        critic_dims = single_actor_dim * n_agents

        # Initialize MADDPG agents and load trained parameters
        maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                               fc1=64, fc2=64,  
                               alpha=0.01, beta=0.01, scenario=layout,
                               chkpt_dir='tmp/maddpg/')
        maddpg_agents.load_checkpoint()

        # Track scores, soups delivered, and intermediate metrics
        score_history = []
        soup_delivery_history = []

        for game in range(n_games):
            obs = base_env.reset()
            score = 0
            soups_delivered = 0
            done = [False] * n_agents

            while not any(done):
                # Render environment for visual confirmation if needed
                if render:
                    base_env.render()

                # Choose actions based on current observation
                actions = maddpg_agents.choose_action(obs)
                
                # Take action in the environment
                obs_, reward, done, info = base_env.step(actions)
                if "soup_delivered" in info:
                    soups_delivered += info["soup_delivered"]

                # Update cumulative reward
                score += sum(reward)
                obs = obs_

            score_history.append(score)
            soup_delivery_history.append(soups_delivered)
            print(f"Layout: {layout} | Game {game + 1}/{n_games} - Score: {score}, Soups Delivered: {soups_delivered}")

        # Store results for this layout
        results[layout] = {
            "avg_score": np.mean(score_history),
            "avg_soups": np.mean(soup_delivery_history),
            "score_history": score_history,
            "soup_delivery_history": soup_delivery_history
        }

        # Plot results for each layout
        plot_soups_delivered_evaluation(soup_delivery_history, filename=f"evaluation_soup_delivery_{layout}.png")

    # Print summary for each layout
    for layout in layouts:
        avg_score = results[layout]["avg_score"]
        avg_soups = results[layout]["avg_soups"]
        print(f"Layout {layout} - Average Score: {avg_score:.2f}, Average Soups Delivered: {avg_soups:.2f}")

if __name__ == "__main__":
    evaluate_agent(horizon=400, n_games=10, render=False)