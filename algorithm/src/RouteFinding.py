import os
import numpy as np
import heapq
import matplotlib.pyplot as plt


folder_path = '../../mapset/'


def map_handler(path, obs_threshold=2.5):
    map_set = []
    for file in os.listdir(path):
        if file.endswith('.npy'):
            file_path = os.path.join(path, file)
            map_data = np.load(file_path)
            map_set.append(map_data)
    high_map = []
    bin_map = []
    for map in map_set:
        high_map.append(map[:, :, 3])
        bin_map.append(np.where(map[:, :, 3] > obs_threshold, 1, 0))
    return high_map, bin_map


