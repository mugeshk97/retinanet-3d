import math
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils import data


from config import data_config
from data_process.bev_utils import (drawRotatedBox, get_filtered_lidar,
                                    makeBEVMap)
from data_process.common_utils import get_image, get_label, get_lidar
from data_process.target_utils import build_targets


class Dataset(data.Dataset):
    def __init__(self, mode="train", augmentation=None):

        self.dataset_dir = data_config.dataset
        # self.input_size = data_config.input_size
        # self.hm_size = data_config.hm_size

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        self.augmentation = augmentation


        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.lidar_dir = os.path.join(self.dataset_dir, "pcds")
        self.label_dir = os.path.join(self.dataset_dir, "labels")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()]
        
    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            return self.load_img_with_targets(index)

    def load_img_only(self, index):

        sample_id = int(self.sample_id_list[index])
    
        lidar_file = os.path.join(self.lidar_dir, f'{sample_id}.npy')
        label_path = os.path.join(self.label_dir, f'{sample_id}.txt')
        labels, has_labels = get_label(label_path)

        lidarData = get_lidar(lidar_file)
        lidarData, labels = get_filtered_lidar(lidarData, labels)

        bev_map = makeBEVMap(lidarData)
        bev_map = torch.from_numpy(bev_map)

        return bev_map, labels, sample_id

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""
        sample_id = int(self.sample_id_list[index])

        # img_path = os.path.join(self.image_dir, f'{sample_id}.png')

        lidar_file = os.path.join(self.lidar_dir, f'{sample_id}.npy')
        label_path = os.path.join(self.label_dir, f'{sample_id}.txt')

        lidarData = get_lidar(lidar_file)
        labels, has_labels = get_label(label_path)

        if self.augmentation:
            lidarData, labels[:, 1:] = self.augmentation(lidarData, labels[:, 1:])

        lidarData, labels = get_filtered_lidar(lidarData, labels)

        bev_map = makeBEVMap(lidarData)
        bev_map = torch.from_numpy(bev_map)

        # hflipped = False
        # if np.random.random() < self.hflip_prob:
        #     hflipped = True
        #     bev_map = torch.flip(bev_map, [-1])

        targets = build_targets(labels)
        # print(bev_map.shape, targets)
        return bev_map, targets, sample_id


    def draw_img_with_label(self, index):
        
        sample_id = int(self.sample_id_list[index])
        print(sample_id)

        lidar_file = os.path.join(self.lidar_dir, f'{sample_id}.npy')
        label_path = os.path.join(self.label_dir, f'{sample_id}.txt')

        lidarData = get_lidar(lidar_file)
        labels, has_labels = get_label(label_path)


        if self.augmentation:
            lidarData, labels[:, 1:] = self.augmentation(lidarData, labels[:, 1:])

 
        lidarData, labels = get_filtered_lidar(lidarData, labels)
        bev_map = makeBEVMap(lidarData)

        return bev_map, labels




if __name__ == "__main__":
    

    dataset = Dataset(mode='test')
    boundary = {
        "minX": 0,
        "maxX": 60,
        "minY": -30,
        "maxY": 30,
        "minZ": -2.73,
        "maxZ": 3.00
    }

    bound_size_x = boundary['maxX'] - boundary['minX']
    bound_size_y = boundary['maxY'] - boundary['minY']
    bound_size_z = boundary['maxZ'] - boundary['minZ']

    BEV_WIDTH = 608  # across y axis -25m ~ 25m
    BEV_HEIGHT = 608  # across x axis 0m ~ 50m

    DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT
    
    for idx in range(len(dataset)):
        # print(idx)
        bev_map, labels = dataset.draw_img_with_label(idx)
        # print(bev_map.shape)
        # print(labels.shape)

        bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
        bev_map = cv2.cvtColor(bev_map, cv2.COLOR_BGR2RGB)
            
        for box_idx, (cls_id, x, y, z, w, l, h, yaw) in enumerate(labels):

            # if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
            #     continue
            # if (h <= 0) or (w <= 0) or (l <= 0):
            #     continue
            
            # Draw rotated box
            # yaw = -yaw

            y1 = int((x - boundary['minX']) / DISCRETIZATION)
            x1 = int((y - boundary['minY']) / DISCRETIZATION)
            w1 = int(w / DISCRETIZATION)
            l1 = int(l / DISCRETIZATION)

            drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, (0,255,0))
        # Rotate the bev_map
        # bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

        cv2.imshow('bev_map', bev_map)
        if cv2.waitKey(0) & 0xff == 27:
            break
        cv2.destroyAllWindows()