from models.UNet import UNet
from losses import VAELoss
from utils.data_utils import set_seed, get_device, AverageMeter
from utils.trainer import Trainer
import os, torch, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import yaml
import numpy as np

class DiffusionTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)

        self.net = self._init_diffuser(config)
        self.net.to(device=self.device)
        self.optimizer = self._init_optimizer(self.net)
        self.timesteps = self.config.diffusion.timesteps
    
        # Initialize fixed_eval_batch for visualization
        self.fixed_eval_batch = torch.randn((8, 1, self.height, self.width), device=self.device)

        # Provided criterion for MSE loss.
        self.criterion = nn.MSELoss(reduction='mean')

        self.noise_start = self.config.diffusion.noise_start
        self.noise_end = self.config.diffusion.noise_end

        #############################################################################
        # Noise Schedule Initialization:
        # 1. Create a linear schedule for beta from noise_start to noise_end over timesteps.
        # 2. Compute alphas as (1 - beta).
        # 3. Compute cumulative product alpha_bars = prod_{s=1}^{t} alpha_s.
        #############################################################################
        self.betas = torch.linspace(self.noise_start, self.noise_end, self.timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        #############################################################################
        #                              END OF NOISE SCHEDULE INIT                   #
        #############################################################################

    @property
    def beta(self):
        return self.betas

    @property
    def alpha(self):
        return self.alphas

    @property
    def alphas_bar(self):
        return self.alpha_bars

    @staticmethod
    def _build_diffuser(cfg):
        return UNet(cfg)

    def _init_diffuser(self, cfg):
        net = self._build_diffuser(cfg)
        return net

    def forward_diffusion(self, x_0, t):
        """
        Applies forward diffusion (q_sample) on the clean image x_0 at timestep t.
        
        Args:
            x_0: Clean image tensor of shape [B, C, H, W].
            t: Timestep tensor of shape [B, 1] (or [B]).
        
        Returns:
            x_t: Noised image at timestep t.
            noise: The noise that was added.
        """
        t_flat = t.view(-1)  # Ensure shape is [B]
        alpha_bar_t = self.alpha_bars[t_flat].view(-1, 1, 1, 1)  # Expand to [B, 1, 1, 1]
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return x_t, noise

    def train(self):
        start_train = time.time()
        loss_meter = AverageMeter()
        iter_meter = AverageMeter()

        # Define the selected epochs at which to generate images.
        selected_epochs = {0, self.n_epochs // 2, self.n_epochs - 1}

        for epoch in range(self.n_epochs):
            for i, (data, _) in enumerate(self.train_loader):
                self.net.train()
                start = time.time()

                data = data.to(self.device)
                data = 2 * data - 1  # Normalize to [-1, 1]
                self.batch_size = data.size(0)

                #############################################################################
                # Training Step:
                # 1. Sample a random timestep for each image.
                # 2. Apply forward diffusion: obtain x_t and the added noise.
                # 3. Predict the noise using the model: pass (x_t, t) into the UNet.
                # 4. Compute the MSE loss between predicted noise and true noise.
                # 5. Backpropagate and update the model parameters.
                #############################################################################
                self.optimizer.zero_grad()
                t = torch.randint(0, self.timesteps, (data.size(0),), device=self.device, dtype=torch.long)
                t = t.view(-1, 1)  # Reshape to [B, 1]
                x_t, noise = self.forward_diffusion(data, t)
                predicted_noise = self.net(x_t, t)
                loss = F.mse_loss(predicted_noise, noise)
                loss.backward()
                self.optimizer.step()
                #############################################################################
                #                              END OF TRAINING STEP                       #
                #############################################################################

                loss_meter.update(loss.item(), data.size(0))
                iter_meter.update(time.time() - start)

            # At the end of each epoch, if the epoch is in the selected set, generate images.
            if epoch in selected_epochs:
                self.visualize_forward_diffusion(data, epoch=epoch)
                self.visualize_reverse_diffusion(epoch=epoch)
                print(
                    f'Epoch: [{epoch}/{self.n_epochs}]\t'
                    f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                    f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
                )

        # After training, generate final reverse diffusion images.
        self.visualize_reverse_diffusion(epoch=self.n_epochs - 1)
        print(f"Completed in {(time.time() - start_train):.3f} seconds")
        self.save_model(self.net, f'{self.output_dir}/diffusion_net_{self.dataset.lower()}.pth')

    @torch.no_grad()
    def sample_timestep(self, x, t):
        t_flat = t.view(-1)
        beta_t = self.betas[t_flat].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t_flat].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t_flat].view(-1, 1, 1, 1)
        
        predicted_noise = self.net(x, t)
        one_over_sqrt_alpha_t = 1.0 / torch.sqrt(alpha_t)
        term1 = one_over_sqrt_alpha_t * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
        
        if t.min().item() > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x_prev = term1 + sigma_t * z
        else:
            x_prev = term1
        return x_prev

    @torch.no_grad()
    def sample(self, epoch, x, save_freq=500):
        self.net.eval()
        for t_val in reversed(range(self.timesteps)):
            t_tensor = torch.full((x.size(0), 1), t_val, device=self.device, dtype=torch.long)
            x = self.sample_timestep(x, t_tensor)
        normalized_x = (x.clone() + 1) / 2
        return normalized_x

    @torch.no_grad()
    def generate(self, n):
        self.batch_size = 16
        x = torch.randn((n, 1, self.height, self.width), device=self.device)
        saved_images = []
        for img_idx in range(0, x.size(0), self.batch_size):
            x_current = x[img_idx:img_idx+self.batch_size, :, :, :]
            for i in reversed(range(self.timesteps)):
                t = torch.full((x_current.size(0),), i, device=self.device, dtype=torch.long).unsqueeze(1)
                x_current = self.sample_timestep(x_current, t)

            normalized_x = (x_current.clone() + 1) / 2
            saved_images.extend(normalized_x)
            
        saved_images = torch.stack(saved_images)
        return saved_images

    @torch.no_grad()
    def visualize_forward_diffusion(self, data, epoch, steps_to_plot=None, max_n=8):
        if steps_to_plot is None:
            steps_to_plot = torch.linspace(0, self.timesteps - 1, steps=50, dtype=torch.int32, device=self.device)
        
        x0 = data[:max_n].to(self.device)
        self.batch_size = x0.size(0)
        
        timestep_images = []
        for t_int in steps_to_plot:
            t = torch.full((self.batch_size,), t_int, device=self.device, dtype=torch.long).view(-1, 1)
            x_t, _ = self.forward_diffusion(x0, t)
            x_t = (x_t + 1) / 2  # Normalize to [0,1]
            timestep_images.append(x_t)
        
        timestep_images = torch.stack(timestep_images)
        grid = timestep_images.transpose(0, 1).reshape(-1, *x0.shape[1:])
        
        vutils.save_image(
            grid,
            f"{self.output_dir}/epoch{epoch}_forward_diffusion_steps.png",
            nrow=len(steps_to_plot),
            padding=2
        )

    @torch.no_grad()
    def visualize_reverse_diffusion(self, x=None, epoch=0, max_n=8):
        if x is None:
            x = torch.randn((max_n, 1, self.height, self.width), device=self.device)
        else:
            x = x[:max_n].to(self.device)
        
        self.batch_size = x.size(0)
        x_current = x.clone()

        save_steps = torch.linspace(self.timesteps - 1, 0, steps=50, dtype=torch.int32, device=self.device)
        saved_images = []
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((self.batch_size,), i, device=self.device, dtype=torch.long).unsqueeze(1)
            x_current = self.sample_timestep(x_current, t)
            if i in save_steps:
                normalized_x = (x_current.clone() + 1) / 2
                saved_images.append(normalized_x)
        
        saved_images = torch.stack(saved_images)
        grid = saved_images.transpose(0, 1).reshape(-1, *x.shape[1:])
        
        vutils.save_image(
            grid,
            f"{self.output_dir}/reverse_diffusion_epoch_{epoch}.png",
            nrow=len(save_steps),
            padding=2
        )
