from models.GAN import BasicDiscriminator, BasicGenerator, BasicLeakyGenerator
from utils.data_utils import set_seed, get_device, AverageMeter
from utils.trainer import Trainer
import os, torch, time, argparse
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import yaml

class GANTrainer(Trainer):
    def __init__(self, config, output_dir=None, device=None):
        super().__init__(config, output_dir=output_dir, device=device)

        self.generator = self._init_generator()
        self.discriminator = self._init_discriminator()

        self.generator.to(device=self.device)
        self.discriminator.to(device=self.device)
        self.optimizer_gen = self._init_optimizer(self.generator)
        self.optimizer_disc = self._init_optimizer(self.discriminator)
        self.criterion = torch.nn.BCELoss(reduction='mean')

        self.fixed_eval_latents = torch.randn((64, self.config.network.latent_dim), device=self.device)

    @staticmethod
    def _build_generator(output_dim, hidden_dim, latent_dim, leaky=False):
        """
        Build basic generator given inputs.
        """
        if leaky:
            return BasicLeakyGenerator(output_dim=output_dim, 
                                         hidden_dim=hidden_dim,
                                         latent_dim=latent_dim)
        else:
            return BasicGenerator(output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  latent_dim=latent_dim)

    def _init_generator(self):
        """
        Instantiate generator.
        """
        generator = self._build_generator(output_dim=self.input_dim, 
                                          hidden_dim=self.config.network.hidden_dim, 
                                          latent_dim=self.config.network.latent_dim,
                                          leaky=self.config.gan.leaky)
        return generator
    
    @staticmethod
    def _build_discriminator(input_dim, hidden_dim, leaky=True):
        """
        Build basic discriminator given inputs.
        """
        return BasicDiscriminator(input_dim=input_dim, 
                                  hidden_dim=hidden_dim,
                                  output_dim=1,
                                  leaky=leaky)

    def _init_discriminator(self):
        """
        Instantiate discriminator.
        """
        disc = self._build_discriminator(input_dim=self.input_dim, 
                                         hidden_dim=self.config.network.hidden_dim, 
                                         leaky=self.config.gan.leaky)
        return disc

    def compute_loss_real(self, data, batch_size):
        #############################################################################
        # 1. Flatten the real data and move to device.
        # 2. Create real labels (all ones).
        # 3. Pass real images through the discriminator and compute BCELoss.
        #############################################################################
        real_images = data.view(batch_size, -1).to(self.device)
        D_out = self.discriminator(real_images)
        real_labels = torch.ones_like(D_out, device=self.device)
        loss_real = self.criterion(D_out, real_labels)
        return loss_real

    def compute_loss_fake(self, batch_size, latent_dim):
        #############################################################################
        # 1. Sample latents from a standard normal distribution.
        # 2. Generate fake images using the generator.
        # 3. Detach fake images from the graph to prevent backpropagation into the generator.
        # 4. Create fake labels (all zeros).
        # 5. Pass fake images through the discriminator and compute BCELoss.
        #############################################################################
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_images = fake_images.view(batch_size, -1).detach()
        D_out_fake = self.discriminator(fake_images)
        fake_labels = torch.zeros_like(D_out_fake, device=self.device)
        loss_fake = self.criterion(D_out_fake, fake_labels)
        return loss_fake

    def compute_loss_gen(self, batch_size, latent_dim):
        #############################################################################
        # 1. Sample new latents.
        # 2. Generate fake images using the generator.
        # 3. Create real labels (all ones) for these fake images.
        # 4. Pass fake images through the discriminator and compute BCELoss.
        #############################################################################
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_images = fake_images.view(batch_size, -1)
        D_out_fake = self.discriminator(fake_images)
        real_labels = torch.ones_like(D_out_fake, device=self.device)
        loss_gen = self.criterion(D_out_fake, real_labels)
        return loss_gen

    def train(self):
        start = time.time()

        loss_meter_gen = AverageMeter()
        loss_meter_fake = AverageMeter()
        loss_meter_real = AverageMeter()
        iter_meter = AverageMeter()

        hidden_dim = self.config.network.hidden_dim
        latent_dim = self.config.network.latent_dim
        loss_func = self.criterion

        # Define selected epochs for generating images:
        selected_epochs = {0, self.n_epochs // 2, self.n_epochs - 1}

        for epoch in range(self.n_epochs):
            for i, (data, _) in enumerate(self.train_loader):
                start_iter = time.time()
                batch_size = data.size(0)

                # A. Train discriminator
                self.optimizer_disc.zero_grad()
                loss_real = self.compute_loss_real(data, batch_size)
                loss_fake = self.compute_loss_fake(batch_size, latent_dim)
                d_loss = loss_real + loss_fake
                d_loss.backward()
                self.optimizer_disc.step()

                # B. Train generator
                self.optimizer_gen.zero_grad()
                loss_gen = self.compute_loss_gen(batch_size, latent_dim)
                loss_gen.backward()
                self.optimizer_gen.step()

                loss_meter_real.update(loss_real.item(), batch_size)
                loss_meter_fake.update(loss_fake.item(), batch_size)
                loss_meter_gen.update(loss_gen.item(), batch_size)
                iter_meter.update(time.time() - start_iter)

                if i % 1000 == 0:
                    print(
                        f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                        f'D_Loss_Real {loss_meter_real.val:.3f} ({loss_meter_real.avg:.3f})\t'
                        f'D_Loss_Fake {loss_meter_fake.val:.3f} ({loss_meter_fake.avg:.3f})\t'
                        f'G_Loss {loss_meter_gen.val:.3f} ({loss_meter_gen.avg:.3f})\t'
                        f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
                    )

            # After each epoch, if it's one of the selected ones, generate sample images.
            if epoch in selected_epochs:
                filename = f"{self.output_dir}/samples_{epoch}.png"
                self.sample_generator(self.fixed_eval_latents, filename, n=self.fixed_eval_latents.size(0))

        filename = f"{self.output_dir}/final_samples.png"
        self.sample_generator(self.fixed_eval_latents, filename, n=self.fixed_eval_latents.size(0))
        print(f"Completed in {(time.time()-start):.3f} seconds.")
        self.save_model(self.discriminator, f'{self.output_dir}/discriminator_{self.dataset.lower()}.pth')
        self.save_model(self.generator, f'{self.output_dir}/generator_{self.dataset.lower()}.pth')

    def sample_generator(self, data, filename, n):
        self.generator.eval()
        with torch.no_grad():
            res = self.generator(data.to(self.device))
        vutils.save_image(res.view(-1, 1, 28, 28), filename, nrow=n)
        self.generator.train()
