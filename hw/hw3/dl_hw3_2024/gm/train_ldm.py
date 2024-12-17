import click
import torch
from trainers import LDMTrainer
from model import VAE, UNet
from diffusion import GaussianDiffusion
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from configs import TrainingConfig


def train(vae_ckpt, batch_size, exp_name):
    device = 'cuda'
    cfg = TrainingConfig()
    cfg.num_epochs = 100
    cfg.batch_size = batch_size
    cfg.exp_name = exp_name

    model = UNet()
    diffusion = GaussianDiffusion()

    vae_model = VAE()
    vae_model.load_state_dict(torch.load(vae_ckpt)["model_state_dict"], strict=False)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train_ds = CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_ds = CIFAR10(root="./data", train=False, transform=transform, download=True)
    trainer = LDMTrainer(cfg, device, model, diffusion, vae_model, train_ds, test_ds)
    trainer.fit()


@click.command()
@click.option('--vae_ckpt', '-c', default=None)
@click.option('--batch-size', '-b', default=64)
@click.option('--exp-name', '-n', default="default")
def main(vae_ckpt, batch_size, exp_name):
    torch.manual_seed(1234)
    train(vae_ckpt, batch_size, exp_name)


if __name__ == "__main__":
    main()
