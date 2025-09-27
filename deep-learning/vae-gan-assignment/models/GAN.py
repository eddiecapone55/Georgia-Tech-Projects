import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import BasicDecoder as BasicGenerator

class BasicDiscriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, output_dim=1, leaky=False):
        super(BasicDiscriminator, self).__init__()
        #############################################################################
        # Implement a basic discriminator with one hidden layer:
        #   - A linear layer to map the input (pre-flattened) to a hidden representation.
        #   - An activation function: ReLU if leaky is False, or LeakyReLU (with negative slope 0.2) if leaky is True.
        #   - A final linear layer that outputs a single value.
        #   - A sigmoid activation on the output to produce a probability.
        #############################################################################
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if leaky:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):   
        #############################################################################
        # Implement forward pass:
        #   1. x is already pre-flattened, so pass it through the first linear layer.
        #   2. Apply the chosen activation function.
        #   3. Pass the result through the output layer.
        #   4. Apply sigmoid to obtain a probability.
        #############################################################################
        h = self.activation(self.fc1(x))
        out = self.sigmoid(self.fc2(h))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

class BasicLeakyGenerator(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(BasicLeakyGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.Sigmoid()

    def forward(self, z):
        h = self.act1(self.fc1(z))
        return self.act2(self.fc2(h))
