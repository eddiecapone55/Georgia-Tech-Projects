import os, torch, time, argparse
import yaml
from trainer_vae import VAETrainer
from trainer_gan import GANTrainer
from trainer_diffusion import DiffusionTrainer
from config import Config
import torchvision.utils as vutils

# Import fid_score from pytorch_fid
from pytorch_fid import fid_score

def compute_fid(generated_dir, real_dir, device, batch_size=50):
    """
    Computes the FID score given two directories of images.
    
    Args:
        generated_dir (str): Path to the directory with generated images.
        real_dir (str): Path to the directory with real images.
        device (str): Device to run FID computation ('cuda' or 'cpu').
        batch_size (int): Batch size for feature extraction.
        
    Returns:
        float: The computed FID score.
    """
    fid_value = fid_score.calculate_fid_given_paths(
        [generated_dir, real_dir],
        batch_size,
        device,
        dims=2048  # use features from Inception V3
    )
    return fid_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images and compute FID")
    parser.add_argument('--config_file', type=str, default='configs/config.yaml',
                        help="path to YAML config")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="path to output directory (optional); defaults to outputs/model_name")
    parser.add_argument('--real_dir', type=str, default=None,
                        help="directory containing real images for FID evaluation")
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
        config = Config(config_dict=config_dict)

    if config.network.model.lower() == 'vae':
        net_trainer = VAETrainer(config=config, output_dir=args.output_dir)
    elif config.network.model.lower() == 'gan':
        net_trainer = GANTrainer(config=config, output_dir=args.output_dir)
    elif config.network.model.lower() == 'diffusion':
        net_trainer = DiffusionTrainer(config=config, output_dir=args.output_dir)
    else:
        raise ValueError(f"{config.network.model} not supported.")
    
    args.output_dir = net_trainer.output_dir
    
    fixed_eval_latents = torch.randn((2000, config.network.latent_dim), device=net_trainer.device)

    if config.network.model.lower() == 'gan':
        model = net_trainer.load_model(f"{args.output_dir}/generator_{net_trainer.dataset.lower()}.pth",
                                       map_location=net_trainer.device)
        model.eval()
        with torch.no_grad():
            res = model(fixed_eval_latents.to(net_trainer.device))
        all_images = res.view(-1, 1, 28, 28)

    elif config.network.model.lower() == 'vae':
        model = net_trainer.load_model(f"{args.output_dir}/vae_{net_trainer.dataset.lower()}.pth",
                                       map_location=net_trainer.device)
        model.eval()
        with torch.no_grad():
            res = model.decoder(fixed_eval_latents.to(net_trainer.device))
        all_images = res.view(-1, 1, 28, 28)

    elif config.network.model.lower() == 'diffusion':
        model = net_trainer.load_model(f"{args.output_dir}/diffusion_net_{net_trainer.dataset.lower()}.pth",
                                       map_location=net_trainer.device)
        net_trainer.net = model
        model.eval()
        # For diffusion models, create an initial noise tensor with shape [n, 1, H, W].
        # Ensure that net_trainer.height and net_trainer.width are defined.
        noise = torch.randn((2000, 1, net_trainer.height, net_trainer.width), device=net_trainer.device)
        with torch.no_grad():
            # Call the sample() method (which implements the full reverse diffusion process)
            res = net_trainer.sample(epoch=0, x=noise)
        all_images = res.view(-1, 1, 28, 28)

    # Save a grid of the first 144 images.
    vutils.save_image(all_images[0:144, :, :, :], f"{args.output_dir}/grid.png", nrow=12)

    # Save each generated image into a folder.
    os.makedirs(f"{args.output_dir}/images/", exist_ok=True)
    for i in range(fixed_eval_latents.size(0)):
        vutils.save_image(all_images[i, :, :, :], f"{args.output_dir}/images/output{i}.png")

    # If a directory containing real test images is provided, compute FID score.
    if args.real_dir is not None:
        fid_value = compute_fid(generated_dir=f"{args.output_dir}/images/",
                                real_dir=args.real_dir,
                                device=net_trainer.device)
        print(f"FID Score: {fid_value:.2f}")
