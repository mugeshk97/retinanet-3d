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

 
CLASS_NAME_TO_ID = {
    'CAR': 0,
    'PERSON': 1,
    'TRUCK': 2,
    'VAN': 0
}

dataset = "/home/mugesh/Project/retinanet-3d/dataset"

input_size = (608, 608)
hm_size = (152, 152)
max_objects = 50

BEV_WIDTH = input_size[0]  # across y axis -25m ~ 25m
BEV_HEIGHT = input_size[1]  # across x axis 0m ~ 50m

DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT


head_conv = 64


num_classes = 3
num_center_offset = 2
num_z = 1
num_dim = 3
num_direction = 2  # sin, cos

heads = {
    'hm_cen': num_classes,
    'cen_offset': num_center_offset,
    'direction': num_direction,
    'z_coor': num_z,
    'dim': num_dim
}

arch = 18

num_workers = 8
pin_memory = True


# train configs


accumulation_steps = 4
epochs = 500
checkpoints_dir = "checkpoints/"
seed = 402
patience = 50
batch_size = 4
lr = 0.001
weight_decay = 0
