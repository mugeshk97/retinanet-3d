import numpy as np
from config import data_config


def get_lidar(lidar_file):
    return np.load(lidar_file)


def cls_type_to_id(cls_type):
    if cls_type not in data_config.CLASS_NAME_TO_ID.keys():
        return -1
    return data_config.CLASS_NAME_TO_ID[cls_type]


def get_image(self, img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return img

def get_label(label_path, occluded_threshold=25):
    labels = []
    for line in open(label_path, 'r'):
        line = line.rstrip()
        line_parts = line.split(' ')
        obj_name = line_parts[0]
        cat_id = cls_type_to_id(obj_name)

        truncated = int(float(line_parts[1])) 
        occluded = int(line_parts[2])  
        alpha = float(line_parts[3])  
        bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])])

        if cat_id <= -1 and occluded > occluded_threshold:
            continue
            
        h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10])
        x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13])
        ry = float(line_parts[14]) 
        object_label = [cat_id, x, y, z, w, l, h, ry]

        labels.append(object_label)

    if len(labels) == 0:
        labels = np.zeros((1, 8), dtype=np.float32)
        has_labels = False

    else:
        labels = np.array(labels, dtype=np.float32)
        has_labels = True
    return labels, has_labels