import numpy as np
import os
import sys
import torch
RING_2_SIZE = 35
RING_3_SIZE = 60
RING_1_SIZE_SPHERE = 6
RING_2_SIZE_SPHERE = 18
# RING_2_SIZE_SPHERE = 40
# RING_3_SIZE_SPHERE = 36
RING_3_SIZE_SPHERE = 90
RING_4_SIZE_SPHERE = 60
RING_2_SIZE_SADDLE = 24
# RING_3_SIZE_SADDLE = 45
RING_3_SIZE_SADDLE = 90
RING_2_SIZE_POLYNOM = 18
RING_3_SIZE_POLYNOM = 36
# RING_3_SIZE_POLYNOM = 90
RING_2_SIZE_TOSCA = 43
RING_3_SIZE_TOSCA = 91
ring_size_mapping = {2: RING_2_SIZE, 3: RING_3_SIZE}
ring_size_mapping_sphere = {1: RING_1_SIZE_SPHERE, 2: RING_2_SIZE_SPHERE, 3: RING_3_SIZE_SPHERE, 4: RING_4_SIZE_SPHERE}
ring_size_mapping_saddle = {2: RING_2_SIZE_SADDLE, 3: RING_3_SIZE_SADDLE}
ring_size_mapping_polynom = {2: RING_2_SIZE_POLYNOM, 3: RING_3_SIZE_POLYNOM}
ring_size_mapping_tosca = {2: RING_2_SIZE_TOSCA, 3: RING_3_SIZE_TOSCA}

torch_configs = dict(
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    type=torch.float64,
    float32=torch.float32,
    float64=torch.float64,
    float16=torch.float16,
    int32=torch.int32,
    int64=torch.int64
)

def set_torch_configs(device: str, dtype: str = 'float64') -> None:
    """
    Sets torch device and dtype according to user's request.
    :param device:
    :return:
    """
    global torch_configs

    torch_configs['type'] = torch_configs[dtype]

    if device == 'cpu':
        torch_configs['dev'] = torch.device('cpu')

    if device == 'cuda' and not torch.cuda.is_available():
        print('cuda is not available, using device: cpu.')


def torch_(x, t=None):
    """
    Converts torch object to device and dtype.
    :param x: any torch object.
    :param t: object type.
    :return: x in the current available device and dtype.
    """
    if t:
        return x.to(torch_configs['dev']).type(t)
    return x.to(torch_configs['dev']).type(torch_configs['type'])


def canonical_rotation(a, b=None):
    if b is None:
        b = np.array([0., 1., 0.])
    if np.linalg.norm(a) == 0:
        return np.eye(3)
    a = a / np.linalg.norm(a)
    if np.linalg.norm(b - np.dot(a, b) * a) == 0:
        return np.eye(3)

    g = np.array([[np.dot(a, b), - np.linalg.norm(np.cross(a, b)), 0.],
                  [np.linalg.norm(np.cross(a, b)), np.dot(a, b), 0.],
                  [0., 0., 1.]])

    f = np.zeros((3, 3))
    f[:, 0] = a
    f[:, 1] = (b - np.dot(a, b) * a) / np.linalg.norm(b - np.dot(a, b) * a)
    f[:, 2] = np.cross(b, a)

    r = f @ g @ np.linalg.inv(f)

    return r
