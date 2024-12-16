import torch
from torch import nn
from torch.utils.data import DataLoader
from loss import KLDivergenceLoss, ReconstructionLoss, GANLossD, GANLossG
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm, trange
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
from torchinfo import summary
from configs import TrainingConfig
from model import Discriminator
from diffusion import GaussianDiffusion
from torchvision.utils import save_image


class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        with open(f'./runs/{self.run_name}/metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.model.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')


class VAETrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"vae_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.num_epochs = cfg.num_epochs
        self.num_visualization = cfg.num_visualization
        self.eval_freq = 1000
        self.save_freq = 2000
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=8,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=cfg.batch_size,
                                          num_workers=8,
                                          pin_memory=True)
        self.model = model
        self.kl_weight = cfg.kl_weight
        self.gan_weight = cfg.gan_weight
        self.gan_loss_start = cfg.gan_loss_start

        self.criterion_kl = KLDivergenceLoss()
        self.criterion_recon = ReconstructionLoss()
        
        if self.gan_weight > 0.0:
            self.criterion_gan_d = GANLossD()
            self.criterion_gan_g = GANLossG()
            self.discriminator = Discriminator()
            self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=cfg.lr)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float32

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        self.model.to(self.device)
        summary(self.model, input_data=torch.ones(1, 3, 32, 32, device=self.device))

        if self.gan_weight > 0.0:
            self.discriminator.to(self.device)
            summary(self.discriminator, input_data=torch.ones(1, 3, 32, 32, device=self.device))
        
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        steps = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            for (x, _) in (pbar := tqdm(self.train_dataloader)):
                steps += 1
                x = x.to(self.device)

                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    x_recon, mu, logvar = self.model(x)
                    kl_loss = self.criterion_kl(mu, logvar)
                    recon_loss = self.criterion_recon(x, x_recon)
                    loss = recon_loss + self.kl_weight * kl_loss

                    if self.gan_weight > 0.0 and steps >= self.gan_loss_start:
                        fake = self.discriminator(x_recon)
                        loss_gan_g = self.criterion_gan_g(fake)
                        loss += self.gan_weight * loss_gan_g

                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)

                self.optimizer.zero_grad(set_to_none=True)
                # scaler.scale(loss).backward(retain_graph=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                lossf = loss.item()
                writer.add_scalar('Loss/train/step', lossf, steps)
                pbar.set_description(f"Epoch {epoch}, batch loss {round(lossf, 3)}, kl loss {round(kl_loss.item(), 3)}, recon loss {round(recon_loss.item(), 3)}")

                if self.gan_weight > 0.0 and steps >= self.gan_loss_start:
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        real = self.discriminator(x)
                        fake = self.discriminator(x_recon.detach())
                        loss_gan_d = self.gan_weight * self.criterion_gan_d(real, fake)
                    self.optimizer_d.zero_grad(set_to_none=True)
                    scaler.scale(loss_gan_d).backward()
                    scaler.step(self.optimizer_d)
                    scaler.update()
                    loss_gan_df = loss_gan_d.item()
                    writer.add_scalar('Loss/train/gan_d/step', loss_gan_df, steps)

                if steps % self.save_freq == 0:
                    self.save_states(steps)

                if steps % self.eval_freq == 0:
                    self.model.eval()
                    with torch.no_grad():
                        for idx, (x, _) in zip(trange(self.num_visualization), self.test_dataloader):
                            x = x.to(self.device)
                            z = self.model.encode(x)
                            x_recon = self.model.decode(z)
                            x_sample = self.model.decode(torch.randn_like(z))
                            x_concat = torch.stack([x, x_recon, x_sample], dim=1).reshape(-1, *x.shape[1:])
                            if not os.path.exists(f'./runs/{self.run_name}/{steps}'):
                                os.makedirs(f'./runs/{self.run_name}/{steps}')
                            save_image(x_concat * 0.5 + 0.5, f'./runs/{self.run_name}/{steps}/samples_{idx}.png', nrow=3)
                    self.model.train()

        self.save_states(steps, True)


class LDMTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module, diffusion: GaussianDiffusion, vae_model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"ldm_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.num_epochs = cfg.num_epochs
        self.num_visualization = cfg.num_visualization
        self.eval_freq = 1000
        self.save_freq = 5000
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=8,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=cfg.batch_size,
                                          num_workers=8,
                                          pin_memory=True)
        self.model = model
        self.diffusion = diffusion
        self.vae_model = vae_model

        self.vae_scale, self.vae_shift = None, None

        self.criterion_recon = ReconstructionLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float32

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        self.model.to(self.device)
        self.diffusion.to(self.device)

        self.vae_model.to(self.device)
        
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        steps = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            for x, y in (pbar := tqdm(self.train_dataloader)):
                steps += 1
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    with torch.no_grad():
                        z = self.vae_model.encode(x)
                        if self.vae_scale is None:
                            self.vae_scale = torch.std(z, dim=(0, 2, 3), keepdim=True)
                            self.vae_shift = torch.mean(z, dim=(0, 2, 3), keepdim=True)
                        z = (z - self.vae_shift) / self.vae_scale
                    loss = self.diffusion(self.model, z, y)

                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)

                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                lossf = loss.item()
                writer.add_scalar('Loss/train/step', lossf, steps)
                pbar.set_description(f"Epoch {epoch}, batch loss {round(lossf, 3)}")

                if steps % self.save_freq == 0:
                    self.save_states(steps)

                if steps % self.eval_freq == 0:
                    self.model.eval()
                    with torch.no_grad():
                        for idx, (x, y) in zip(trange(self.num_visualization), self.test_dataloader):
                            x = x.to(self.device)
                            y = y.to(self.device)
                            z = self.vae_model.encode(x)
                            z_sample = self.diffusion.sample(self.model, z.shape, y)
                            z_sample = z_sample * self.vae_scale + self.vae_shift
                            x_recon = self.vae_model.decode(z)
                            x_sample = self.vae_model.decode(z_sample)
                            x_concat = torch.stack([x, x_recon, x_sample], dim=1).reshape(-1, *x.shape[1:])
                            if not os.path.exists(f'./runs/{self.run_name}/{steps}'):
                                os.makedirs(f'./runs/{self.run_name}/{steps}')
                            save_image(x_concat * 0.5 + 0.5, f'./runs/{self.run_name}/{steps}/samples_{idx}.png', nrow=3)
                    self.model.train()

        self.save_states(steps, True)
