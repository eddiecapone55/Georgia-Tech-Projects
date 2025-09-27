import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, beta=1, recon_loss='l2', reduction='mean', return_losses=False):
        super(VAELoss, self).__init__()
        self.beta = beta
        # Set up the reconstruction loss based on the provided string.
        if recon_loss.lower() == 'l2':
            self.reconstruction_loss = nn.MSELoss(reduction=reduction)
        if recon_loss.lower() == 'l1':
            self.reconstruction_loss = nn.L1Loss(reduction=reduction)
        if recon_loss.lower() == 'bce':
            self.reconstruction_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.return_losses = return_losses

    def forward(self, reconstructed, original, mu, logvar, kl_weight=None):
        """
        Computes the VAE loss with optional KL annealing.

        Args:
            reconstructed: The reconstructed output from the decoder.
            original: The original input image.
            mu: The mean from the encoder.
            logvar: The log variance from the encoder.
            kl_weight: Optional weight for the KL divergence term. If not provided, defaults to self.beta.

        Returns:
            The total VAE loss (and optionally, the individual loss components).
        """
        # Use the provided KL weight or default to self.beta.
        if kl_weight is None:
            kl_weight = self.beta

        #############################################################################
        # 1. Compute the reconstruction loss and scale it by 784 (28x28).
        #############################################################################
        loss_recon = self.reconstruction_loss(reconstructed, original) * 784
        
        #############################################################################
        # 2. Compute the KL divergence loss:
        #    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        #############################################################################
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        #############################################################################
        # 3. Compute the total loss as:
        #    total_loss = reconstruction_loss + kl_weight * KL_loss
        #############################################################################
        loss = loss_recon + kl_weight * loss_kl
        
        #############################################################################
        # 4. Return the losses based on the flag.
        #############################################################################
        if self.return_losses:
            return loss, loss_recon, loss_kl
        else:
            return loss
