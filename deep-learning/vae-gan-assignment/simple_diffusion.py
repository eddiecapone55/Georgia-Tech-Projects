import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
import os
from utils.data_utils import load_pt_data, get_device
from models.noise_predictor import SimpleNoisePredictor

os.makedirs('outputs', exist_ok=True)

class SimpleDiffusionTrainer:
    def __init__(self, image, device='cuda'):
        self.device = device
        self.model = SimpleNoisePredictor().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        self.image = image.to(device)
        self.output_dir = f"./outputs/simple_diffusion"
        os.makedirs(self.output_dir, exist_ok=True,)
        
        # sample fixed noise
        num_noises = 1
        noise_amplitude = 8
        self.fixed_noises = [torch.rand_like(self.image) * noise_amplitude for i in range(num_noises)]

    def sample_noise(self, idx=None):
        if idx is None:
            idx = torch.randint(0, len(self.fixed_noises), (1,)).item()
        return self.fixed_noises[idx]
        
    def train_step(self):
        self.model.train()
        image = self.image
        fixed_noise = self.sample_noise()  # Use the fixed noise generated in __init__
        torch.manual_seed(42) 
                
        # 1. Create a noisy image by adding the fixed noise to the clean image.
        noisy_image = image + fixed_noise
        self.optimizer.zero_grad()
        
        # 2. Flatten the noisy image and feed it to the model
        pred_noise = self.model(noisy_image.view(1, -1))
        # 3. Reshape the predicted noise back to the image shape
        pred_noise = pred_noise.view_as(noisy_image)
        
        # 4. Compute the loss between the predicted noise and the ground truth noise using MSE loss.
        loss = F.mse_loss(pred_noise, fixed_noise)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), pred_noise.detach()

    @torch.no_grad()
    def reverse_step(self, step):
        self.model.eval()
        # Create a noisy image by adding fixed noise to the original image.
        fixed_noise = self.sample_noise()
        noisy_image = self.image + fixed_noise
        denoised_image, pred_noise = None, None

        # 1. Predict the noise from the noisy image using the model.
        pred_noise = self.model(noisy_image.view(1, -1))
        pred_noise = pred_noise.view_as(noisy_image)
        
        # 2. Denoise the image by subtracting the predicted noise from the noisy image.
        denoised_image = noisy_image - pred_noise

        # Visualization: Concatenate original, noisy, fixed noise, predicted noise, and denoised image.
        grid = torch.cat([
            self.image,        # original image
            noisy_image,       # noisy image
            fixed_noise,       # fixed noise (target)
            pred_noise,        # predicted noise
            denoised_image     # denoised image
        ], dim=0)
        
        vutils.save_image(grid.unsqueeze(1), os.path.join(self.output_dir, f"noise_test_{step:04d}.png"), nrow=5, normalize=True)
        return grid

def train():
    images, _ = map(list, zip(*load_pt_data('mnist_1_shots.pt')))
    image = images[0]
    
    # Normalize to [-1,1]
    single_image = 2 * image - 1 

    device = get_device()
    trainer = SimpleDiffusionTrainer(single_image, device=device)
    
    trainer.reverse_step(step=0)
    # Training loop
    for step in range(100):
        loss, pred_noise = trainer.train_step()
        
        if (step + 1) % 25 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
            trainer.reverse_step(step)
    
if __name__ == "__main__":
    train()
