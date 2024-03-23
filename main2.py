"""
@author: 15201622364
"""

from controller import Camera, RangeFinder, Supervisor
from algorithm.rover_controller import rover_controller
import numpy as np
from algorithm.result_saver import result_saver

if __name__ == '__main__':
    file_dicts = [
        {
            "file": 'map.npy',
            "start": [10, -15],
            "goal": [-15, 15],
        },
        {
            "file": 'TestMap-3.npy',
            "start": [11, -14],
            "goal": [15, 15],
        },
        {
            "file": 'TestMap4.npy',
            "start": [-11, 18],
            "goal": [15, -15],
        },
        {
            "file": 'TestMap5.npy',
            "start": [-15, -20],
            "goal": [0, 10],
        },
    ]
    map_id = 0
    map_path = file_dicts[map_id]["file"]
    sup = Supervisor()
    # sup.simulationReset()
    # sup.step(1000)

    robot = sup.getFromDef('Rover')  # 连接巡视器

    motor_lf = sup.getDevice('lf_motor')  # 连接左前轮驱动
    motor_lb = sup.getDevice('lb_motor')  # 连接左后轮驱动
    motor_rf = sup.getDevice('rf_motor')  # 连接右前轮驱动
    motor_rb = sup.getDevice('rb_motor')  # 连接右后轮驱动

    motor_lf.setPosition(float('inf'))  # 左前轮初始化
    motor_rf.setPosition(float('inf'))
    motor_lb.setPosition(float('inf'))
    motor_rb.setPosition(float('inf'))
    motor_lf.setVelocity(0)  # 给定左前轮转速，不超过1
    motor_rf.setVelocity(0)
    motor_lb.setVelocity(0)
    motor_rb.setVelocity(0)

    cam_f = Camera('cam_f')  # 连接前相机-RGB
    cam_b = Camera('cam_b')
    cam_f.enable(30)  # 初始化
    cam_b.enable(30)

    dep_f = RangeFinder('depth_f')  # 连接前相机D
    dep_b = RangeFinder('depth_b')
    dep_f.enable(30)  # 初始化
    dep_b.enable(30)

    """先验信息"""
    moon_map = np.load(f'./mapset/{map_path}')
    pos_A = file_dicts[map_id]['start']  # 初始点的x\y坐标
    area_B = np.concatenate((file_dicts[map_id]["goal"], [10, 10]), axis=0).tolist()  # 目标区域的中心点x\y坐标、x\y宽度

    con = rover_controller(moon_map, pos_A, area_B)  # 决策程序实例化


    world_time = 0
    done = False
    rotation_flag = False
    while not done:
        """获取RGBD数据"""
        rgb_image_f = cam_f.getImage()  # 前相机获取RGB图像
        rgb_image_b = cam_b.getImage()
        d_image_f = dep_f.getRangeImage()  # 前相机获取D图像
        d_image_b = dep_b.getRangeImage()

        """获取巡视器状态"""
        translation_field = robot.getField('translation')
        rotation_field = robot.getField('rotation')
        position = translation_field.getSFVec3f()
        rotation = rotation_field.getSFRotation()
        if map_path in ['TestMap-3.npy', 'TestMap5.npy', 'TestMap4.npy']:
            car_angle = rotation[3] + np.pi / 2
        elif map_path in ['map.npy']:
            car_angle = -rotation[3] + np.pi / 2
        else:
            raise NameError
        car_status = np.array([np.arctan2(np.sin(car_angle), np.cos(car_angle)), np.array([position[0], -position[2]]).astype(np.float64)], dtype=object)

        """决策主程序"""
        motor_velocity, done, target_pos, rotation_flag = con.step(rgb_image_f, rgb_image_b, d_image_f, d_image_b, world_time, car_status, rotation_flag)

        """巡视器轮速控制"""
        if motor_velocity != []:
            motor_lf.setVelocity(motor_velocity[0])  # 给定左前轮转速，-1~1
            motor_rf.setVelocity(motor_velocity[1])
            motor_lb.setVelocity(motor_velocity[2])
            motor_rb.setVelocity(motor_velocity[3])

        """仿真环境运行"""
        for _ in range(30):
            # now_step = int(world_time / 960) + 1
            # sup.exportImage(f"E:/py_program/simenv/real_world/{now_step}.png", 100)
            sup.step(32)
            world_time += 32  # 32ms
    result_saver(map_id, './algorithm/temp/tar_all.npy')
