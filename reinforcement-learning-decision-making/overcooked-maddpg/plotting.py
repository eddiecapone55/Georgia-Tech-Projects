import matplotlib.pyplot as plt
import numpy as np

def plot_onion_pickups(onion_pickups, filename="onion_pickups.png", window=100):
    """Plots the moving average of onion pickups per 1000 episodes."""
    avg_onions = np.convolve(onion_pickups, np.ones(window)/window, mode='valid')
    plt.figure()
    plt.plot(avg_onions)
    plt.title("Onion Pickups per 1000 Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Onions Picked Up")
    plt.savefig(filename)
    plt.close()

def plot_dish_pickups(dish_pickups, filename="dish_pickups.png", window=100):
    """Plots the moving average of dish pickups per 1000 episodes."""
    avg_dishes = np.convolve(dish_pickups, np.ones(window)/window, mode='valid')
    plt.figure()
    plt.plot(avg_dishes)
    plt.title("Dish Pickups per 1000 Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Dishes Picked Up")
    plt.savefig(filename)
    plt.close()

def plot_avg_soups_delivered(soups, filename="avg_soups_delivered.png", window=100):
    """Plots the moving average of soups delivered per episode."""
    avg_soups = np.convolve(soups, np.ones(window)/window, mode='valid')
    plt.figure()
    plt.plot(avg_soups)
    plt.title("Average Soups Delivered per 1000 Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Soups Delivered")
    plt.savefig(filename)
    plt.close()

def plot_soups_delivered_evaluation(soups, filename="soups_delivered_evaluation.png", window=100):
    """Plots the evaluation of soups delivered per episode."""
    avg_soups = np.convolve(soups, np.ones(window)/window, mode='valid')
    plt.figure()
    plt.plot(avg_soups)
    plt.title("Soups Delivered per 1000 Episodes (Evaluation)")
    plt.xlabel("Episode")
    plt.ylabel("Soups Delivered")
    plt.savefig(filename)
    plt.close()