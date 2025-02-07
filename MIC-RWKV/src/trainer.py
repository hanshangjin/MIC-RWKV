########################################################################################################
# This part modified from RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
import torch.nn as nn
from torch.optim import Adam
import torch
from tqdm.auto import tqdm
import numpy as np

from src.dna_utils import collate_fn


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class Trainer:

    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.avg_loss = -1
        self.steps = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def run_epoch(self, split, optimizer=None):
        is_train = split == 'train'
        model = self.model
        model.train(is_train)
        data = self.train_dataset if is_train else self.val_dataset

        loader = DataLoader(
            data,
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
            collate_fn=lambda batch: collate_fn(batch, self.config.n_amr, self.config.ctx_len)
        )

        pbar = tqdm(enumerate(loader), total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)

        total_loss = 0
        total_cls_loss = 0
        total_contra_loss = 0

        for it, (x, y, mask, num_sequence_mask) in pbar:
            x, y, mask, num_sequence_mask = x.to(self.device), y.to(self.device), mask.to(self.device), num_sequence_mask.to(self.device)

            with torch.set_grad_enabled(is_train):
                _, _, cls_loss, contra_loss = model(x, y, mask, num_sequence_mask)
                contra_loss = 10 * contra_loss
                loss = cls_loss + contra_loss

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    if self.config.grad_norm_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                    optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_contra_loss += contra_loss.item()

            if is_train:
                self.steps += 1
                avg_loss = total_loss / (it + 1)
                pbar.set_description(f"Iteration: {self.steps} | Avg Loss: {avg_loss:.4f} | CLS: {cls_loss:.4f} | Contra: {contra_loss:.4f}")

        avg_loss = total_loss / len(loader)
        avg_cls_loss = total_cls_loss / len(loader)
        avg_contra_loss = total_contra_loss / len(loader)

        return avg_loss, avg_cls_loss, avg_contra_loss

    def train(self):
        model, config = self.model, self.config
        optimizer = Adam(model.parameters(), lr=config.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=config.eta_min)

        for epoch in range(config.max_epochs):
            print(f"Epoch {epoch+1}/{config.max_epochs}")

            # Training
            train_loss, train_cls, train_contra = self.run_epoch('train', optimizer)
            scheduler.step()

            # Validation
            val_loss, val_cls, val_contra = self.run_epoch('val')

            print(f"Validation Loss - Iteration {epoch+1}: {val_loss:.4f}, CLS: {val_cls:.4f}, Contra: {val_contra:.4f}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, self.config.epoch_save_path + str(epoch+1) + '.pth')
