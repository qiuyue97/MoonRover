import os
import numpy as np

def build_temp_environment():
    current_directory = os.getcwd()
    new_directory = os.path.join(current_directory, 'algorithm', 'temp')
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    img_directory = os.path.join(new_directory, 'img')
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)
    sub_dirs_1 = ['interest_target', 'obstacles', 'route_plan']
    sub_dirs_2 = ['rgb_f', 'rgb_b', 'd_f', 'd_b', 'res']
    for sub_dir in sub_dirs_1:
        new_sub_dir = os.path.join(img_directory, sub_dir)
        if not os.path.exists(new_sub_dir):
            os.makedirs(new_sub_dir)
    for sub_dir in sub_dirs_2:
        new_sub_dir = os.path.join(img_directory, 'interest_target', sub_dir)
        if not os.path.exists(new_sub_dir):
            os.makedirs(new_sub_dir)
