import math

import numpy as np
from scipy import signal

from config import data_config


def compute_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)

def gaussian2D(kernel_size, sigma=1):
    gaussian_kernel = signal.gaussian(kernel_size[0], std=sigma)[:, np.newaxis] * signal.gaussian(kernel_size[1], std=sigma)[np.newaxis, :]
    return gaussian_kernel


def gen_hm_radius(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return masked_heatmap



def build_targets(labels):
    
    max_objects = data_config.max_objects
    hm_size = data_config.hm_size
    
    minX = data_config.boundary['minX']
    maxX = data_config.boundary['maxX']
    minY = data_config.boundary['minY']
    maxY = data_config.boundary['maxY']
    minZ = data_config.boundary['minZ']
    maxZ = data_config.boundary['maxZ']


    num_objects = min(len(labels), max_objects)
    hm_l, hm_w = hm_size

    hm_main_center = np.zeros((data_config.num_classes, hm_l, hm_w), dtype=np.float32)
    cen_offset = np.zeros((max_objects, data_config.num_center_offset), dtype=np.float32)
    direction = np.zeros((max_objects, data_config.num_direction), dtype=np.float32)
    z_coor = np.zeros((max_objects, data_config.num_z), dtype=np.float32)
    dimension = np.zeros((max_objects, data_config.num_dim), dtype=np.float32)

    indices_center = np.zeros((max_objects), dtype=np.int64)
    obj_mask = np.zeros((max_objects), dtype=np.uint8)

    for k in range(num_objects):
        cls_id, x, y, z, w, l, h, yaw = labels[k]
        cls_id = int(cls_id)
        yaw = -yaw
        if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
            continue
        if (h <= 0) or (w <= 0) or (l <= 0):
            continue

        bbox_l = l / data_config.bound_size_x * hm_l
        bbox_w = w / data_config.bound_size_y * hm_w

        radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))

        radius = max(0, int(radius))

        center_y = (x - minX) / data_config.bound_size_x * hm_l  # x --> y (invert to 2D image space)
        center_x = (y - minY) / data_config.bound_size_y * hm_w  # y --> x

        center = np.array([center_x, center_y], dtype=np.float32)

        center_int = center.astype(np.int32) 

        # if cls_id < 0:
        #     ignore_ids = [_ for _ in range(num_classes)] if cls_id == - 1 else [- cls_id - 2]
        #     # Consider to make mask ignore
        #     for cls_ig in ignore_ids:
        #         gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
        #     hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999
        #     continue

        gen_hm_radius(hm_main_center[cls_id], center, radius)
        indices_center[k] = center_int[1] * hm_w + center_int[0]
        
        cen_offset[k] = center - center_int

        dimension[k, 0] = w
        dimension[k, 1] = l
        dimension[k, 2] = h

        direction[k, 0] = math.sin(float(yaw))  # im
        direction[k, 1] = math.cos(float(yaw))  # re

#         if hflipped:
#             direction[k, 0] = - direction[k, 0]

        z_coor[k] = z - minZ

        obj_mask[k] = 1

    targets = {
        'hm_cen': hm_main_center,
        'cen_offset': cen_offset,
        'direction': direction,
        'z_coor': z_coor,
        'dim': dimension,
        'indices_center': indices_center,
        'obj_mask': obj_mask,
    }
    return targets