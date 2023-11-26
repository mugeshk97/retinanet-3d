import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


from config import data_config
from data_process.dataset import Dataset

# from data_process.transformation import OneOf, Random_Rotation, Random_Scaling,Random_Translation,Random_Flipping


num_workers = 8
pin_memory = True

def create_train_dataloader(batch_size):
    """Create dataloader for training"""
    # train_lidar_aug = OneOf([
    #     Random_Rotation(limit_angle=np.pi / 4, p=1.0),
    #     Random_Scaling(scaling_range=(0.95, 1.05), p=1.0),
    #     Random_Translation(limit_translation=0.5, p=0.5),
    #     Random_Flipping(p=0.5),

    # ], p=0.66)
    train_dataset = Dataset(mode='train', augmentation=None)
    train_sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True,
                                  pin_memory=data_config.pin_memory, num_workers=data_config.num_workers, sampler=train_sampler)

    return train_dataloader, train_sampler


def create_val_dataloader(batch_size):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = Dataset(mode='val', augmentation=None)
 
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                pin_memory=data_config.pin_memory, num_workers=data_config.num_workers, sampler=val_sampler)
    return val_dataloader


def create_test_dataloader(batch_size):
    """Create dataloader for testing phase"""
    test_dataset = Dataset(mode='test', augmentation=None)
    test_sampler = None
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 pin_memory=data_config.pin_memory, num_workers=data_config.num_workers, sampler=test_sampler)

    return test_dataloader



if __name__ == "__main__":

    from config import data_config

    train_dataloader, train_sampler = create_train_dataloader(batch_size=4)

    for batch_idx, batch_data in enumerate(train_dataloader):
        bev , target, ts = batch_data
        print(bev.shape, target['hm_cen'].shape)
        break