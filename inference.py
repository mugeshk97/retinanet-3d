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

import math

import matplotlib.pyplot as plt
import open3d as o3d

from config import data_config
from data_process.data_loader import create_test_dataloader
from model.retinanet import BasicBlock, ResNet, Bottleneck
from utils.inference_utils import (convert_det_to_real_values, decode,
                                   draw_predictions, post_processing,
                                   yaw_rotation_matrix)
from utils.viz import visualize_point_cloud


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


configs = edict()

configs.pretrained_path = "/home/user/Project/new_3d/checkpoints_101/model_val_loss_ 0.635011_epoch_486.pth"
configs.K = 50
configs.down_ratio = 4
configs.peak_thresh = 0.2


# model = ResNet(BasicBlock, [2, 2, 2, 2])
model = ResNet(Bottleneck, [3, 4, 23, 3]) #resnet101


assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
print('Loaded weights from {}\n'.format(configs.pretrained_path))


model = model.to(device="cuda:0")

model.eval()

test_dataloader = create_test_dataloader(1)
with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_dataloader):
        bev_maps,labels,  time_stamp = batch_data
        print(time_stamp)
        
        ts_pcd = "/home/user/Project/new_3d/dataset/rear_bin/"+str(time_stamp.numpy()[0])+'.npy'
        lidar_point = np.load(ts_pcd)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(lidar_point))
        # visualize_point_cloud(pcd)


        input_bev_maps = bev_maps.to("cuda:0", non_blocking=True).float()


        outputs = model(input_bev_maps)
    
        outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
        outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
    
        
        detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                            outputs['dim'], K=configs.K)
        detections = detections.cpu().numpy().astype(np.float32)
        detections = post_processing(detections, data_config.num_classes, configs.down_ratio, configs.peak_thresh)

        detections = detections[0]
        # print(detections)  # only first batch
        


        bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (data_config.BEV_WIDTH, data_config.BEV_HEIGHT))
        bev_map = draw_predictions(bev_map, detections.copy(), data_config.num_classes)

        # Rotate the bev_map
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
        

        boxes_3d = []
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

                R = yaw_rotation_matrix(yaw)
                box_3d = o3d.geometry.OrientedBoundingBox([x, y, z], R, [h, l, w])
                line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box_3d)
                lines = np.asarray(line_set.lines)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.paint_uniform_color((0.16862745098039217, 0.6588235294117647, 0.8392156862745098))


                boxes_3d.append(line_set)
            

        cv2.imshow("Image", bev_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # visualize_point_cloud(pcd, boxes_3d)

