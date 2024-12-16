import torch
from torch import nn
from torch.utils.data import DataLoader
from loss import CrossEntropyLoss, DPOLoss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
import statistics
from tqdm import tqdm, trange
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
from torchinfo import summary
from configs import TrainingConfig


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


class SFTTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"sft_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.num_steps = cfg.num_steps
        self.save_freq = 20000
        self.train_dataloader =  iter(
            DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=8,
                       pin_memory=True))
        self.test_dataloader = iter(
            DataLoader(test_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=8,
                       pin_memory=True))
        self.model = model
        self.criterion = CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

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
        summary(self.model, input_data=torch.ones(1, 1024).long())

        self.model.to(self.device)
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        self.model.train()

        with trange(1, self.num_steps + 1) as pbar:
            for step in pbar:
                x, y = next(self.train_dataloader)
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)

                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.grad_clip)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                lossf = loss.item()
                writer.add_scalar('Loss/train/step', lossf, step)

                pbar.set_description(f"Step {step}, batch loss {round(lossf, 3)}")

                if step % self.save_freq == 0:
                    self.save_states(step)

        self.save_states(step, True)


class DPOTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module, sft_model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"dpo_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.num_epochs = cfg.num_epochs
        self.eval_freq = 5000
        self.save_freq = 20000
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=8,
                                           shuffle=True,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=cfg.batch_size,
                                          num_workers=8,
                                          pin_memory=True)
        self.model = model
        self.sft_model = sft_model
        self.criterion = DPOLoss(kl_beta=cfg.kl_beta)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)
    
    def shared_step(self, completions, attention_masks):
        ########################################################################
        # TODO: Implement a single step of DPO trainer
        # TODO: You should implement the model call (including self.model and self.model_sft)
        # TODO: After calling the model and obtaining log_p, calculate the loss and accuracy
        # TODO: Hint: (completions[:, 0], attention_masks[:, 0]) is the positive sample
        #         and (completions[:, 1], attention_masks[:, 1]) is the negative sample
        ############################ Your code here ############################
        loss, acc = None, None
        ########################################################################
        return loss, acc

    def fit(self):
        summary(self.model, input_data=torch.ones(1, 1024).long())

        self.model.to(self.device)
        self.model_sft.to(self.device)
        self.model_sft.eval()
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        self.model.train()

        steps = 0

        for epoch in range(1, self.num_epochs + 1):
            with tqdm(self.train_dataloader) as pbar:
                for completions, attention_masks in pbar:
                    steps += 1
                    completions = completions.to(self.device)
                    attention_masks = attention_masks.to(self.device)

                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        loss, acc = self.shared_step(self.model, self.model_sft, completions, attention_masks)

                    if self.grad_clip != 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.grad_clip)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    lossf = loss.item()
                    accf = acc.item()
                    writer.add_scalar('Loss/train/step', lossf, steps)
                    writer.add_scalar('Acc/train/step', accf, steps)
                    pbar.set_description(f"Epoch {epoch}, batch loss {round(lossf, 3)}, acc {round(accf, 3)}")

                    if steps != 0 and steps % self.save_freq == 0:
                        self.save_states(steps)

                    if steps % self.eval_freq == 0:
                        with torch.no_grad():
                            self.model.eval()
                            losses, accs = [], []
                            for completions, attention_masks in tqdm(self.test_dataloader):
                                completions = completions.to(self.device)
                                attention_masks = attention_masks.to(self.device)
                                loss, acc = self.shared_step(self.model, self.model_sft, completions, attention_masks)
                                lossf, accf = loss.item(), acc.item()
                                losses.append(lossf)
                                accs.append(accf)
                            total_loss, total_acc = statistics.mean(losses), statistics.mean(accs)
                            writer.add_scalar('Loss/test/step', total_loss, steps)
                            writer.add_scalar('Acc/test/step', total_acc, steps)
                            print(f'Step: {steps + 1}, Test Loss: {total_loss}, Acc: {total_acc}')

            self.save_states(steps, True)
