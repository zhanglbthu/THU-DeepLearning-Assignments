import click
import torch
from trainers import DPOTrainer
from gpt import GPT
from configs import get_configs
from dataset import DahoasRMStaticDataset


def train(ckpt, batch_size, exp_name):
    device = 'cuda'
    cfg = get_configs("gpt2")
    cfg.num_epochs = 5
    cfg.batch_size = batch_size
    cfg.ckpt = ckpt
    cfg.exp_name = exp_name

    model = GPT.from_pretrained(cfg, ckpt)
    sft_model = GPT.from_pretrained(cfg, ckpt)

    train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=None)
    test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=100)
    trainer = DPOTrainer(cfg, device, model, sft_model, train_ds, test_ds)
    trainer.fit()


@click.command()
@click.option('--ckpt', '-c', default="/path/to/sft/model")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
def main(ckpt, batch_size, exp_name):
    torch.manual_seed(1234)
    train(ckpt, batch_size, exp_name)


if __name__ == "__main__":
    main()
