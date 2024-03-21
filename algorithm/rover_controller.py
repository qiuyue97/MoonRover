"""
@author: 15201622364
"""

import numpy as np
from algorithm.builder import build_temp_environment
from algorithm.map_handler import map_handler
from algorithm.route_planer import Dijkstra_main, action_judge
from algorithm.obstacles_avoider import avoider_main
from algorithm.box_finder import finder_main, point_clustering

class rover_controller(object):
    def __init__(self, moon_map, pos_A, area_B):
        self.map = moon_map
        self.pos_A = pos_A
        self.area_B = area_B

    def step(self, rgb_image_f, rgb_image_b, d_image_f, d_image_b, world_time, car_status, rotation_flag):
        """
        功能：进行一步决策推理，请根据输入自行判断是否、如何进行决策推理
        输入：
            rgb_image_f: 前相机RGB数据，分辨率1280*720
            rgb_image_b: 后相机RGB数据，分辨率1280*720
            d_image_f: 前相机D数据，与RGB对齐，分辨率1280*720
            d_image_b: 后相机D数据，与RGB对齐，分辨率1280*720
            world_time: 仿真器的时间，单位ms
        输出：
            motor_velocity: 四轮转速[左前，右前，左后，右后]，范围[-1,1]，如不进行推理则输出[]
            done: 任务是否结束，结束为True，不结束False，自行判断何时结束
            target_pos: 发现的目标的二维坐标，维度n*2的list，n为发现的目标个数
                        当done为True时输出该list，当done为False时输出[]
        """
        done = False
        tar_all = []
        target_pos = []
        now_step = int(world_time / 960) + 1
        if world_time == 0:
            build_temp_environment()
            map_handler(self.map)
            path = Dijkstra_main(car_status, self.area_B) # 迪杰斯特拉寻路
            rest_path, actions = action_judge(path, car_status, scaling_ratio=25)
            np.save('./algorithm/temp/rest_path', rest_path)
            np.save('./algorithm/temp/actions', actions)
            np.save('./algorithm/temp/tar_all', tar_all)
        else:
            rest_path = np.load('./algorithm/temp/rest_path.npy')
            actions = np.load('./algorithm/temp/actions.npy')
            tar_all = np.load('./algorithm/temp/tar_all.npy').tolist()
        if len(actions) == 0 and len(rest_path) > 0:
            rest_path, actions = action_judge(rest_path, car_status, scaling_ratio=25)
            np.save('./algorithm/temp/rest_path', rest_path)
        # 根据行动表行动
        if len(actions) > 0:
            if d_image_f is not None and rotation_flag is False:
                o_flag = avoider_main(car_status, d_image_f, now_step, self.area_B)
            # 若发现障碍物完成避障算法，重新加载行动表
                if o_flag is True:
                    actions = np.load('./algorithm/temp/actions.npy')
            # 调用兴趣目标发现主程序
            tar_all.extend(finder_main(rgb_image_f, d_image_f, car_status, now_step, 'f'))
            tar_all.extend(finder_main(rgb_image_b, d_image_b, car_status, now_step, 'b'))
            np.save('./algorithm/temp/tar_all', tar_all)
            if actions[0] == 'left':
                motor_velocity = [-1, 1, -1, 1]
                np.save('./algorithm/temp/actions', actions[1:])
                print(f'现在是第{now_step}步，执行左转命令，小车与Y轴夹角{np.degrees(car_status[0]):.2f}度，'
                      f'位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
            elif actions[0] == 'right':
                motor_velocity = [1, -1, 1, -1]
                np.save('./algorithm/temp/actions', actions[1:])
                print(f'现在是第{now_step}步，执行右转命令，小车与Y轴夹角{np.degrees(car_status[0]):.2f}度，'
                      f'位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
            elif actions[0] == 'ahead':
                motor_velocity = [1, 1, 1, 1]
                np.save('./algorithm/temp/actions', actions[1:])
                print(f'现在是第{now_step}步，执行直行命令，小车与Y轴夹角{np.degrees(car_status[0]):.2f}度，'
                      f'位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
        else:
            if rotation_flag is False:
                motor_velocity = [-1, 1, -1, 1]
                actions = np.concatenate((actions, np.array(['left'] * 19)))
                np.save('./algorithm/temp/actions', actions[1:])
                rotation_flag = True
            else:
                motor_velocity = [0, 0, 0, 0]
                target_pos = point_clustering(tar_all)
                print('任务完成，发现的兴趣目标list如下：\n', target_pos)
                done = True

        return motor_velocity, done, target_pos, rotation_flag