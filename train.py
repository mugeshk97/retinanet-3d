import json
import os
import random
import sys
import time
import warnings

import numpy as np
from easydict import EasyDict as edict

warnings.filterwarnings("ignore", category=UserWarning)

import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import data_config
from data_process.data_loader import (create_train_dataloader,
                                      create_val_dataloader)
from loss.loss import Compute_Loss
from model.callbacks import EarlyStopping
from model.retinanet import retinanet


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
            
def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch):
    losses = AverageMeter('Loss', ':.4e')

    criterion = Compute_Loss(device=device)
    
    # switch to train mode
    
    model.train()
    
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
    loop.set_description(desc=f"Epoch {epoch}/{data_config.epochs}")

    for batch_idx, batch_data in loop:
        imgs, targets, ts = batch_data


        batch_size = imgs.size(0)

        for k in targets.keys():
            targets[k] = targets[k].to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True).float()
        outputs = model(imgs)

        total_loss, loss_stats = criterion(outputs, targets)

        total_loss.backward()

        if (batch_idx + 1) % data_config.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        reduced_loss = total_loss.data

        losses.update(to_python_float(reduced_loss), batch_size)

        loop.set_postfix(loss=losses.avg) 

    if lr_scheduler:
        lr_scheduler.step()
    
    return losses.avg


def validate(val_dataloader, model):
    losses = AverageMeter('Loss', ':.4e')
    criterion = Compute_Loss(device=device)
    model.eval()
    
    loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=True)
    loop.set_description(desc=f"Validation")
    with torch.no_grad():
        for batch_idx, batch_data in loop:
            imgs, targets, ts = batch_data
            batch_size = imgs.size(0)
            for k in targets.keys():
                targets[k] = targets[k].to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True).float()
            outputs = model(imgs)
            total_loss, loss_stats = criterion(outputs, targets)
            reduced_loss = total_loss.data
            losses.update(to_python_float(reduced_loss), batch_size)
            loop.set_postfix(val_loss=losses.avg)

    return losses.avg



if os.path.exists(data_config.checkpoints_dir):
    print("[INFO] directory already exist change the checkpoint dir name")
    sys.exit()
else:
    os.mkdir(data_config.checkpoints_dir)

# Re-produce results
if data_config.seed is not None:
    random.seed(data_config.seed)
    np.random.seed(data_config.seed)
    torch.manual_seed(data_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cpu' if 0 == None else 'cuda:{}'.format(0))

model = retinanet(data_config.arch)

model.cuda(0)
torch.cuda.set_device(0)

train_params = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.Adam(train_params, lr = data_config.lr, weight_decay = data_config.weight_decay)

# scheduler
lf = lambda x: (((1 + math.cos(x * math.pi / data_config.epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)

early_stoping =EarlyStopping(patience=data_config.patience)

# Create dataloader
train_dataloader, train_sampler = create_train_dataloader(data_config.batch_size)
val_dataloader = create_val_dataloader(data_config.batch_size)


for epoch in range(1, data_config.epochs + 1):
    # if early_stoping.early_stop:
    #     break
    train_loss = train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch)

    val_loss = validate(val_dataloader, model)

    early_stoping(train_loss, val_loss, model, data_config.checkpoints_dir, epoch)
