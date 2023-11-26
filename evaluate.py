import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

import math

import matplotlib.pyplot as plt
import open3d as o3d

from config import data_config
from data_process.data_loader import create_test_dataloader
from metrics.metricsv2 import ConfusionMatrix
from model.retinanet import BasicBlock, Bottleneck, ResNet
from utils.inference_utils import (compute_3d_bbox_corners,
                                   convert_det_to_real_values, decode,
                                   draw_predictions, post_processing)
from utils.viz import visualize_point_cloud


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


configs = edict()

configs.pretrained_path = "/home/user/Project/new_3d/checkpoints_101/model_val_loss_ 0.635011_epoch_486.pth"
configs.K = 50
configs.down_ratio = 4
configs.peak_thresh = 0.2
configs.num_classes = 3
configs.iou_thres = 0.1

confusion_matrix = ConfusionMatrix(configs.num_classes, conf=0.5, iou_thres=configs.iou_thres)

# model = ResNet(BasicBlock, [2, 2, 2, 2])
model = ResNet(Bottleneck, [3, 4, 23, 3]) #resnet101

assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
print('Loaded weights from {}\n'.format(configs.pretrained_path))


model = model.to(device="cuda:0")

model.eval()

prediction_corner = []
prediction_classes = []

ground_truth_corner = []
ground_truth_classes = []


test_dataloader = create_test_dataloader(1)

loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=True)

with torch.no_grad():
    for batch_idx, batch_data in loop:
        confusion_matrix = ConfusionMatrix(configs.num_classes, conf=0.5, iou_thres=configs.iou_thres)
        bev_map , labels,  time_stamp = batch_data



        if labels.shape[1] == 0:
            continue


        # ts_pcd = "/home/user/Project/SFA3D/bmw_data/rear_bin/"+str(time_stamp.numpy()[0])+'.npy'
        # lidar_point = np.load(ts_pcd)[:, :3]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np.asarray(lidar_point))
        # visualize_point_cloud(pcd)

        for i in labels[0]:
            
            cat_id, x, y, z, h, w, l, ry = i

            if cat_id < 0:
                continue

            
            gt_corner = compute_3d_bbox_corners(x, y, z, l, w, h, ry)
            
            ground_truth_corner.append(gt_corner)
            ground_truth_classes.append(cat_id)
        
        
        input_bev_maps = bev_map.to("cuda:0", non_blocking=True).float()

        outputs = model(input_bev_maps)

        outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
        outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
        
        detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                            outputs['dim'], K=configs.K)
        detections = detections.cpu().numpy().astype(np.float32)

        detections = post_processing(detections, data_config.num_classes, configs.down_ratio, configs.peak_thresh)

        detections = detections[0]


        detects_3d = convert_det_to_real_values(detections, num_classes= data_config.num_classes)
        if len(detects_3d) > 0:
            for i in detects_3d:
                cls_id = i[0]
                x = i[1]
                y = i[2]
                z = i[3]
                w = i[4]
                l = i[5]
                h = i[6]
                yaw = i[7]
                corner = compute_3d_bbox_corners(x, y, z, h, l, w, yaw)

                prediction_classes.append(cls_id)
                prediction_corner.append(corner)

        
        # print(ground_truth_classes, prediction_classes)
        # print(prediction_corner)
        # print(ground_truth_corner)
        # bb = []
        # for j in prediction_corner:
        #     bb.append(j)
        # for j in ground_truth_corner:
        #     bb.append(j)

        # visualize_point_cloud(point_cloud=pcd, bbox=bb)
        
        # break

ground_truth_corner = np.asarray(ground_truth_corner)
ground_truth_classes = np.asarray(ground_truth_classes)
prediction_classes = np.asarray(prediction_classes)
prediction_corner = np.asarray(prediction_corner)
    
confusion_matrix.process_batch(ground_truth_corner, prediction_corner, ground_truth_classes, prediction_classes)
print(confusion_matrix.matrix)


