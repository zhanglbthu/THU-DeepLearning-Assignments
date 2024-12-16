import click
import torch
from trainers import VAETrainer
from model import VAE
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from configs import TrainingConfig


def train(ckpt, batch_size, exp_name):
    device = 'cuda'
    cfg = TrainingConfig()
    cfg.num_epochs = 10
    cfg.batch_size = batch_size
    cfg.ckpt = ckpt
    cfg.exp_name = exp_name

    model = VAE()

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train_ds = CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_ds = CIFAR10(root="./data", train=False, transform=transform, download=True)
    trainer = VAETrainer(cfg, device, model, train_ds, test_ds)
    trainer.fit()


@click.command()
@click.option('--ckpt', '-c', default="./checkpoints/gpt2_pretrained.pt")
@click.option('--batch-size', '-b', default=64)
@click.option('--exp-name', '-n', default="default")
def main(ckpt, batch_size, exp_name):
    torch.manual_seed(1234)
    train(ckpt, batch_size, exp_name)


if __name__ == "__main__":
    main()
