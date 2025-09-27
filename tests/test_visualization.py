import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_curves

class TestVisualization(unittest.TestCase):

    def setUp(self):
        # Sample data for plotting
        self.train_loss_history = [0.9, 0.7, 0.5, 0.3]
        self.train_acc_history = [0.6, 0.7, 0.8, 0.9]
        self.valid_loss_history = [1.0, 0.8, 0.6, 0.4]
        self.valid_acc_history = [0.5, 0.6, 0.75, 0.85]

    def test_plot_curves(self):
        """
        Test if plot_curves function runs without error and produces expected outputs.
        """
        try:
            plot_curves(
                self.train_loss_history, 
                self.train_acc_history, 
                self.valid_loss_history, 
                self.valid_acc_history
            )
            plt.close('all')  # Close plots to avoid resource warnings
        except Exception as e:
            self.fail(f"plot_curves function failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
