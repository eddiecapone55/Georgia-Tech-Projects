import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from environment_setup import setup_environment
from plotting import plot_onion_pickups, plot_dish_pickups, plot_avg_soups_delivered, plot_soups_delivered_evaluation

def process_observations(obs):
    """Flattens and concatenates observations for use in the centralized critic."""
    if obs is None or len(obs) == 0:
        print("Warning: Observation is None or empty.")
        return None
    processed_obs = [o.flatten() if isinstance(o, np.ndarray) else None for o in obs]
    processed_obs = [o for o in processed_obs if o is not None]
    if not processed_obs:
        print("Warning: No valid observations to concatenate.")
        return None
    return np.concatenate(processed_obs)

def obs_list_to_state_vector(observation):
    """Convert a list of observations into a single state vector."""
    if observation is None or len(observation) == 0:
        print("Warning: Observation list is None or empty.")
        return None
    state = np.concatenate([obs for obs in observation if isinstance(obs, np.ndarray)], axis=None)
    return state

def calculate_inter_agent_distance(agent_positions):
    """Calculate Euclidean distance between two agents."""
    pos_a, pos_b = agent_positions
    return np.linalg.norm(np.array(pos_a) - np.array(pos_b))

def continuous_to_discrete_action(continuous_action):
    """Converts a continuous action output into a discrete action expected by Overcooked environment."""
    discrete_actions = ["stay", "up", "down", "left", "right", "interact"]
    discrete_action_index = np.argmax(continuous_action)  # Choose action based on highest activation
    if discrete_action_index >= len(discrete_actions):
        return 0  # Fallback to "stay" if out of bounds
    return discrete_action_index

if __name__ == '__main__':
    layouts = ["cramped_room", "asymmetric_advantages", "forced_coordination"]
    horizon = 400
    reward_shaping = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5
    }
    n_agents = 2
    n_actions = 6

    for layout in layouts:
        base_env, mdp, horizon = setup_environment(layout=layout, horizon=horizon, reward_shaping=reward_shaping)
        
        # Dynamic observation dimensions
        single_actor_dim = base_env.observation_space.shape[0]
        actor_dims = [single_actor_dim] * n_agents
        critic_dims = single_actor_dim * n_agents
        
        maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, fc1=64, fc2=64, alpha=0.01, beta=0.01, scenario=layout, chkpt_dir=f'tmp/maddpg/{layout}/')
        memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)
        
        score_history, soup_delivery_history = [], []
        onion_pickups, dish_pickups = [], []

        PRINT_INTERVAL = 500
        N_GAMES = 60000 
        total_steps = 0
        best_score = 0
        evaluate = False

        if evaluate:
            maddpg_agents.load_checkpoint()

        for i in range(N_GAMES):
            obs = base_env.reset()
            processed_obs = process_observations(obs)
            if processed_obs is None:
                continue  # Skip if initial observation is invalid

            score, soups_delivered = 0, 0
            onions_picked, dishes_picked = 0, 0
            done = [False] * n_agents

            while not any(done):
                if evaluate:
                    base_env.render()

                continuous_actions = maddpg_agents.choose_action(obs)
                actions = [continuous_to_discrete_action(action) for action in continuous_actions]
                
                obs_, reward, done, info = base_env.step(actions)

                if "soup_delivered" in info:
                    soups_delivered += info["soup_delivered"]
                if "onion_picked" in info:
                    onions_picked += info["onion_picked"]
                if "dish_picked" in info:
                    dishes_picked += info["dish_picked"]

                processed_obs_ = process_observations(obs_)
                if processed_obs_ is None:
                    break

                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)
                if state is None or state_ is None:
                    break

                collaborative_reward = sum(reward) + soups_delivered * 2
                memory.store_transition(processed_obs, processed_obs_, actions, collaborative_reward, done)

                if total_steps % 50 == 0 and not evaluate:
                    maddpg_agents.learn(memory)

                obs, processed_obs = obs_, processed_obs_
                score += sum(reward)
                
                total_steps += 1

            score_history.append(score)
            soup_delivery_history.append(soups_delivered)
            onion_pickups.append(onions_picked)
            dish_pickups.append(dishes_picked)

            avg_score = np.mean(score_history[-100:])
            avg_soups = np.mean(soup_delivery_history[-100:])

            if not evaluate and avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score

            if i % PRINT_INTERVAL == 0 and i > 0:
                print(f"Layout: {layout} | Episode {i}, Avg Score: {avg_score:.1f}, Avg Soups: {avg_soups:.1f}, Avg Onions Picked: {np.mean(onion_pickups[-100:]):.1f}, Avg Dishes Picked: {np.mean(dish_pickups[-100:]):.1f}")

        # Plot the required four metrics
        plot_onion_pickups(onion_pickups, filename=f"{layout}_onion_pickups.png")
        plot_dish_pickups(dish_pickups, filename=f"{layout}_dish_pickups.png")
        plot_avg_soups_delivered(soup_delivery_history, filename=f"{layout}_avg_soups_delivered.png")