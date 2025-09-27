import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import BasicDecoder  

class BasicEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(BasicEncoder, self).__init__()
        """
            args:
                input_dim: dim of input image (flattened, e.g., 28x28 = 784)
                hidden_dim: dim of hidden layer
                latent_dim: dim of latent vectors mu and logvar
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        #############################################################################
        # 1. Create a linear layer to map the input into the hidden representation.
        # 2. Create two linear layers (heads) from the hidden representation:
        #    one to produce mu and one to produce logvar (both of dimension latent_dim).
        #############################################################################
        self.fc1 = nn.Linear(input_dim, hidden_dim)      # Input to hidden
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)     # Hidden to mean vector
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # Hidden to log variance vector
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        """ 
        Forward pass for VAE encoder.
        args: 
            x: [N, input_dim]
        outputs:
            mu: [N, latent_dim]
            logvar: [N, latent_dim]
        """
        #############################################################################
        # 1. Pass the input through the first linear layer followed by a ReLU activation 
        #    to obtain a hidden representation.
        # 2. Compute mu and logvar from the hidden representation using two separate heads.
        #############################################################################
        hidden = F.relu(self.fc1(x))         # Hidden representation with ReLU activation
        mu = self.fc_mu(hidden)              # Mean vector (mu)
        logvar = self.fc_logvar(hidden)      # Log variance vector (logvar)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=256):
        """
            args:
                input_dim: dim of input image (flattened, e.g., 784)
                hidden_dim: dim of hidden layer
                latent_dim: dim of latents (for mu and logvar)
            Students implement Basic VAE using encoder and decoder. 
            1. Instantiate encoder and pass in variables.
            2. Instantiate decoder (provided) with appropriate args.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        #############################################################################
        # Instantiate encoder and decoder.
        # Note: For the decoder, we assume the provided BasicDecoder accepts:
        #       (latent_dim, hidden_dim, output_dim) where output_dim equals input_dim.
        #############################################################################
        self.encoder = BasicEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = BasicDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def reparameterize(self, mu, logvar):
        """
            args:
                mu: [N, latent_dim]
                logvar: [N, latent_dim]
            outputs:
                z: reparameterized latent representation [N, latent_dim]
            Implements the reparameterization trick:
                z = mu + eps * sigma, where sigma = exp(0.5 * logvar)
        """
        #############################################################################
        # 1. Compute standard deviation from logvar.
        # 2. Sample epsilon from a standard normal distribution (matching shape).
        # 3. Compute z using: z = mu + epsilon * std.
        #############################################################################
        std = torch.exp(0.5 * logvar)           # Standard deviation: sigma = exp(0.5 * logvar)
        eps = torch.randn_like(std)             # Sample epsilon ~ N(0, I)
        z = mu + eps * std                      # Reparameterization trick
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return z

    def encode(self, x):
        """
            args:
                x: input [N, C, H, W]
            outputs:
                z: latent representation [N, latent_dim]
                mu: mean latent [N, latent_dim]
                logvar: log variance latent [N, latent_dim]
            Note: The VAE takes a flattened input, so we flatten x before encoding.
        """
        #############################################################################
        # 1. Flatten the input image.
        # 2. Pass through the encoder to obtain mu and logvar.
        # 3. Compute the reparameterized latent vector z.
        #############################################################################
        x = x.view(x.size(0), -1)               # Flatten the image to shape [N, input_dim]
        mu, logvar = self.encoder(x)            # Encode to get mu and logvar
        z = self.reparameterize(mu, logvar)     # Reparameterize to get latent vector z
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return (z, mu, logvar)
    
    def forward(self, x):
        """
            args:
                x: input [N, C, H, W]
            outputs:
                out: reconstructed image [N, input_dim]
                mu: mean latent [N, latent_dim]
                logvar: log variance latent [N, latent_dim]
        """
        #############################################################################
        # 1. Encode the input to obtain z, mu, and logvar.
        # 2. Decode z to obtain the reconstructed image.
        #############################################################################
        z, mu, logvar = self.encode(x)  # Get latent representation and statistics
        out = self.decoder(z)           # Reconstruct the image from z
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return (out, mu, logvar)

    @torch.no_grad()
    def generate(self, z):
        """
            args:
                z: latent vector [N, latent_dim]
            outputs:
                out: reconstructed image [N, input_dim]
        """
        self.eval()
        out = self.decoder(z)
        return out
