"""
@author: 15201622364
"""

import numpy as np
from algorithm.builder import build_temp_environment
from algorithm.map_handler import map_handler
from algorithm.route_planer import Dijkstra_main, Baka_main
from algorithm.obstacles_avoider import avoider_main
from algorithm.box_finder import finder_main, point_clustering

class rover_controller(object):
    def __init__(self, moon_map, pos_A, area_B):
        self.map = moon_map
        self.pos_A = pos_A
        self.area_B = area_B

    def step(self, rgb_image_f, rgb_image_b, d_image_f, d_image_b, world_time):
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
            car_status = np.array([0, np.array(self.pos_A).astype(np.float64)], dtype=object)
            # actions = Baka_main(car_status, self.area_B)  # 八嘎寻路
            actions = Dijkstra_main(car_status, self.area_B) # 迪杰斯特拉寻路
            # car_status = np.array([0, np.array(self.area_B[:2]).astype(np.float64)], dtype=object) # 测试用
            # actions = ['ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead', 'ahead'] # 测试用
            np.save('./algorithm/temp/car_status', car_status)
            np.save('./algorithm/temp/actions', actions)
            np.save('./algorithm/temp/tar_all', tar_all)
        else:
            car_status = np.load('./algorithm/temp/car_status.npy', allow_pickle=True)
            actions = np.load('./algorithm/temp/actions.npy')
            tar_all = np.load('./algorithm/temp/tar_all.npy').tolist()
        if now_step <= len(actions):
            # 若行动表后续步数大于2且后3步均为前进，则启动避障判断器
            if (len(actions) - now_step >= 2) & (all(a==b for a, b in zip(actions[now_step - 1:now_step + 2], ['ahead', 'ahead', 'ahead']))):
                o_flag = avoider_main(d_image_f, now_step, self.area_B)
            else:
                o_flag = False
            # 若发现障碍物完成避障算法，重新加载行动表
            if o_flag is True:
                actions = np.load('./algorithm/temp/actions.npy')
            # 调用兴趣目标发现主程序
            tar_all.extend(finder_main(rgb_image_f, d_image_f, car_status, now_step, 'f'))
            tar_all.extend(finder_main(rgb_image_b, d_image_b, car_status, now_step, 'b'))
            np.save('./algorithm/temp/tar_all', tar_all)
            # 根据行动表行动
            if actions[int(now_step - 1)] == 'left':
                motor_velocity = [-1, 1, -1, 1]
                car_status[0] += np.radians(18.95)
                np.save('./algorithm/temp/car_status', car_status)
                print(f'预计需要{len(actions)}步，现在是第{now_step}步，执行左转命令，本步完成后小车与Y轴夹角{np.degrees(car_status[0]):.2f}度，'
                      f'位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
            elif actions[int(now_step - 1)] == 'right':
                motor_velocity = [1, -1, 1, -1]
                car_status[0] -= np.radians(18.95)
                np.save('./algorithm/temp/car_status', car_status)
                print(f'预计需要{len(actions)}步，现在是第{now_step}步，执行右转命令，本步完成后小车与Y轴夹角{np.degrees(car_status[0]):.2f}度，'
                      f'位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
            elif actions[int(now_step - 1)] == 'ahead':
                motor_velocity = [1, 1, 1, 1]
                car_status[1] += (1 / 11) * np.array([-np.sin(car_status[0]), np.cos(car_status[0])])
                np.save('./algorithm/temp/car_status', car_status)
                print(f'预计需要{len(actions)}步，现在是第{now_step}步，执行直行命令，本步完成后小车与Y轴夹角{np.degrees(car_status[0]):.2f}度，'
                      f'位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
        elif now_step <= len(actions) + 20:
            motor_velocity = [-1, 1, -1, 1]
            car_status[0] += np.radians(18.95)
            np.save('./algorithm/temp/car_status', car_status)
            tar_all.extend(finder_main(rgb_image_f, d_image_f, car_status, now_step, 'f'))
            tar_all.extend(finder_main(rgb_image_b, d_image_b, car_status, now_step, 'b'))
            np.save('./algorithm/temp/tar_all', tar_all)
            print(f'正在检测目标区域周围兴趣点({now_step - len(actions)} / 20)...')
        else:
            motor_velocity = [0, 0, 0, 0]
            target_pos = point_clustering(tar_all)
            print('任务完成，发现的兴趣目标list如下：\n', target_pos)
            done = True

        return motor_velocity, done, target_pos