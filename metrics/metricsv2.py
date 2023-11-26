import math
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd


import torch
import math

import torch
import math



def box_iou(box1, box2, eps=1e-7):
    """
    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)



def box_iou_3d(box1, box2, eps=1e-7):
    """
    Args:
        box1 (torch.Tensor): A tensor of shape (N, 8, 3) representing N 3D bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 8, 3) representing M 3D bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # Calculate the minimum and maximum coordinates for each set of corners
    a_min = torch.min(box1, dim=1).values
    a_max = torch.max(box1, dim=1).values
    b_min = torch.min(box2, dim=1).values
    b_max = torch.max(box2, dim=1).values

    # Calculate the intersection volume
    min_max = torch.min(a_max[:, None], b_max)
    max_min = torch.max(a_min[:, None], b_min)
    inter = torch.prod(torch.clamp(min_max - max_min, min=0), dim=2)

    # Calculate the volumes of the boxes
    volume_a = torch.prod(a_max - a_min, dim=1)
    volume_b = torch.prod(b_max - b_min, dim=1)

    # Calculate IoU
    iou = inter / (volume_a[:, None] + volume_b - inter + eps)

    return iou



class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.array): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task='detect'):
        """Initialize attributes for the YOLO model."""
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) #if self.task == 'detect' else np.zeros((nc, nc))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels, true_classes,  detection_classes):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class).
            labels (Array[M, 5]): Ground truth bounding boxes and their associated class labels.
                                  Each row should contain (class, x1, y1, x2, y2).
        """
        detections = torch.Tensor(detections)
        labels = torch.Tensor(labels) 

        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                print(gc)
                self.matrix[self.nc, gc] += 1  # background FN
            return

        # detections = detections[detections[:, 4] > self.conf]
        gt_classes = detection_classes.astype(int)
        detection_classes = true_classes.astype(int)
        iou = box_iou_3d(labels, detections)
        print(iou)

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        """Returns the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp[:-1], fp[:-1], fn[:-1]) #if self.task == 'detect' else (tp, fp)  # remove background class if task=detect

    def plot(self, class_labels, suffix_label):
        cm = self.matrix
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        sns.heatmap(cm, cmap=sns.light_palette("navy", 12),
                    annot=annot, fmt='', cbar=False,
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    annot_kws={"fontsize":14})
        if suffix_label:
            plt.title(f"Confusion Matrix - {suffix_label}", size=20, weight='bold')
        else:
            plt.title(f"Confusion Matrix", size=20, weight='bold')
        
        plt.show()